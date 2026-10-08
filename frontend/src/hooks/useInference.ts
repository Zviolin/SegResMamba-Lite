/**
 * ============================================================
 *  useInference —— 模型推理流程编排 Hook
 * ============================================================
 * 职责：
 *  1. 加载可用模型列表（loadModels），并负责失败时的默认配置回退；
 *  2. 编排完整推理链路（runModelInference）：
 *       根据当前模式获取推理引擎（在线 API / 离线 ONNX）→
 *       执行分割 → 写入 store 并计算肿瘤统计。
 * 对外暴露：modelList、inference 状态、loadModels、runModelInference、isReadyForInference。
 *
 * 【M1 改造说明】推理的具体执行已下沉到 services/inference/ 下的
 * Provider 实现（在线 / 离线），本 Hook 只负责「流程编排 + 状态管理」，
 * 不再直接依赖 apiClient，从而支持离/在线一键切换。
 */
import { useCallback, useEffect, useRef } from 'react';
import { useViewerStore } from '../store/viewerStore';
import { useModeStore } from '../store/modeStore';
import { onlineProvider } from '../services/inference/onlineProvider';
import { offlineProvider } from '../services/inference/offlineProvider';
import { releaseSessions } from '../services/inference/offline/modelSessionCache';
import { saveHistoryRecord } from '../services/storage/historyGateway';
import { buildNiftiGzBlob } from '../utils/niftiWriter';
import type { ModelInfo } from '../types';

/**
 * 离线历史允许保存的原始 MRI 上限（字节，默认 80MB）。
 * 离线模式会把整份原始 MRI 复制进 IndexedDB 用于回放，若不设上限，
 * 几十 MB 一条的记录很快会撑爆浏览器配额（且失败时用户无感知）。
 * 超过上限时只存分割结果，详情页会自动降级为「伪 MRI 灰度」回放。
 */
const OFFLINE_MRI_SAVE_LIMIT = 80 * 1024 * 1024;

/**
 * 推理功能 Hook
 * @returns 模型列表、推理状态、加载模型方法、运行推理方法、是否可开始推理
 */
export function useInference() {
  // 从 store 读取需要的数据与状态
  const modelList = useViewerStore((s) => s.modelList);
  const inference = useViewerStore((s) => s.inference);
  const mriFileInfo = useViewerStore((s) => s.mriFileInfo);
  // 一次性取出需要用到的全部 action
  const {
    setModelList,
    setSelectedModel,
    setInferenceStatus,
    setInferenceResult,
    setInferenceError,
    setInferenceProgress,
    setSegData,
    setSegFileInfo
  } = useViewerStore((s) => s.actions);

  // 当前模式 + 推理引擎：
  // 直接订阅 mode 而非依赖 useModeProvider() 返回的单例引用，
  // 因为 onlineProvider/offlineProvider 是模块常量对象，React 浅比较不会重渲染。
  const mode = useModeStore((s) => s.mode);
  const provider = mode === 'online' ? onlineProvider : offlineProvider;

  // 优化点③：离开离线模式时主动释放 ORT session，防止 WASM 内存泄露/手机 OOM
  useEffect(() => {
    if (mode !== 'offline') {
      releaseSessions().catch(() => { /* 释放失败不阻断 */ });
    }
  }, [mode]);

  /**
   * 加载可用模型列表
   *
   * 通过当前模式的引擎获取模型列表（在线走后端 /api/models，离线返回内置配置）；
   * 若获取失败（例如后端未启动），回退到内置的默认模型配置（SegResNet-Mamba），
   * 保证界面仍可使用。
   */
  const loadModels = useCallback(async () => {
    // 在函数内部按需读取当前选中模型，而不是把它放进依赖数组：
    // 否则「setSelectedModel → selectedModel 变化 → loadModels 重建 → effect 重跑」
    // 会让在线模式每次挂载都多打一次 /api/models。
    const currentModel = useViewerStore.getState().inference.selectedModel;
    try {
      const models: ModelInfo[] = await provider.getModels();
      setModelList(models);
      // 若用户尚未选择模型，则默认选中列表第一个
      if (models.length > 0 && !currentModel) {
        setSelectedModel(models[0].modelType);
      }
    } catch (error) {
      // 网络/后端异常时输出警告并使用默认模型兜底
      console.warn('无法获取模型列表，使用默认配置:', error);
      setModelList([
        {
          modelType: 'SEGRESNET_MAMBA',
          modelName: 'SegResNet-Mamba',
          modelVersion: '1.0.0',
          description: '基于 Mamba 的脑肿瘤分割模型',
          supportedModalities: ['T1ce', 'FLAIR', 'T1', 'T2']
        }
      ]);
      if (!currentModel) {
        setSelectedModel('SEGRESNET_MAMBA');
      }
    }
  }, [provider, setModelList, setSelectedModel]);

  // 组件挂载后自动加载一次模型列表（模式切换后也会因 provider 变化重新加载）
  useEffect(() => {
    loadModels();
  }, [loadModels]);

  /**
   * 运行完整推理流程
   *
   * @param mriFile 需要推理的原始 MRI 文件（来自上传输入框）
   *
   * 流程：
   *   - 交给当前模式的推理引擎执行分割（在线：上传→推理→下载→解析；
   *     离线：本地 ONNX 推理），引擎内部通过 onProgress 上报进度；
   *   - 进度 <40% 视为上传/准备阶段（展示"上传中"），≥40% 视为推理阶段；
   *   - 拿到掩码后写入 store，若 MRI 已就绪则计算肿瘤统计；
   *   - 写入推理结果，进度 100%，状态置为 completed。
   * 任何一步失败都会置为 error 状态并记录错误信息。
   */
  /**
   * 每次推理分配的递增序号，用于判定异步结果返回时「本次调用是否已被取代」。
   *
   * 背景：推理是长异步流程，途中若发生模式切换（在线↔离线）或再次发起推理，
   * 旧流程仍在后台执行，它的进度与结果会覆盖新流程，
   * UI 表现为「进度条乱跳 / 显示的是上一次的结果」。
   * 这里沿用 App.tsx 探活逻辑里同样的过期守卫写法。
   */
  const runIdRef = useRef(0);

  const runModelInference = useCallback(async (mriFile: File, modalityFiles?: Record<string, File>) => {
    const runId = ++runIdRef.current;
    /** 本次调用是否已被后续调用取代（取代后立刻停止写 store） */
    const isStale = () => runId !== runIdRef.current;
    try {
      // ---- 第 0 步：进入推理流程 ----
      setInferenceStatus('processing');
      setInferenceProgress(0);
      setInferenceError(null);

      // 进度回调：根据进度映射"上传中 / 推理中"语义（与改造前一致）
      const handleProgress = (p: number) => {
        setInferenceProgress(p);
        setInferenceStatus(p < 40 ? 'uploading' : 'processing');
      };

      // ---- 第 1 步：调用当前模式的推理引擎执行分割 ----
      const { segData, result } = await provider.runSegmentation(
        {
          mriFile,
          modalityFiles,   // 离线模式 4 模态（可选）
          modality: inference.selectedModality,
          modelType: inference.selectedModel
        },
        handleProgress
      );

      // 推理耗时较长（尤其离线 ONNX），返回时可能已被新任务取代 → 丢弃旧结果
      if (isStale()) return;

      // ---- 第 2 步：分割掩码写入 store ----
      setSegData(segData);
      setSegFileInfo({
        name: 'segmentation_result.nii.gz',
        size: segData.image.byteLength,
        status: 'ready'
      });

      // 肿瘤统计不再在这里算：<AnalysisReport /> 内部对同一对 (segData, mriData)
      // 做了去重计算，这里再算一遍等于把 900 万体素白扫一次。
      // （报告面板折叠未挂载时不会算，等用户展开时再算，结果一致。）

      // ---- 第 3 步：收尾，写入结果 ----
      setInferenceResult(result);
      setInferenceProgress(100);
      setInferenceStatus('completed');

      // ---- 第 4 步（M4）：推理成功后写入历史记录（模式判断在 historyGateway 内）----
      // 离线模式：同时把分割结果 + 原始 MRI 存进 IndexedDB，
      // 历史记录详情页可以像主页一样回放三平面视图（含灰度底图与亮度调节）
      let segGz: Blob | undefined;
      let mriGz: Blob | undefined;
      if (mode === 'offline' && segData) {
        try {
          segGz = await buildNiftiGzBlob(segData);
        } catch (e) {
          console.warn('[offline] 分割压缩包生成失败（历史将不含可导出文件）:', e);
        }
        // 原始 MRI 原样保存（File 本身是 Blob），回放时需要灰度底图才能调亮度。
        // 超过体积上限时跳过：只存分割结果，详情页会降级为「伪 MRI 灰度」回放。
        if (mriFile.size > OFFLINE_MRI_SAVE_LIMIT) {
          console.warn(
            `[offline] 原始 MRI ${(mriFile.size / 1024 / 1024).toFixed(1)}MB 超过 ` +
            `${OFFLINE_MRI_SAVE_LIMIT / 1024 / 1024}MB 上限，本次历史只保存分割结果`
          );
        } else {
          // File 本身就是 Blob 的子类，直接交给 IndexedDB 保存即可。
          // 原先这里先 arrayBuffer() 再 new Blob() —— 等于把整份文件在内存里
          // 完整拷贝一遍（80MB 上限即峰值 160MB+），移动端 WebView 容易 OOM。
          mriGz = mriFile;
        }
      }

      // 压缩包生成是异步的，期间同样可能已被取代
      if (isStale()) return;

      // 历史持久化单独兜错：它失败只意味着「这条记录没存下来」，
      // 而推理本身已经成功、结果也已写入 store 展示给用户。
      // 若让它抛到外层 catch，UI 会显示「推理失败」，与画面上的分割结果自相矛盾。
      try {
        await saveHistoryRecord({
          fileName: mriFile.name,
          modelType: inference.selectedModel,
          modelUsed: result.modelUsed,
          executionTimeMs: result.executionTimeMs,
          volumes: result.volumes,
          status: 'COMPLETED',
          error: null,
          createdAt: new Date().toISOString(),
          jobId: null,
        }, segGz, mriGz);
      } catch (e) {
        console.warn('[history] 历史记录保存失败（推理结果不受影响）:', e);
      }

    } catch (error) {
      // 统一错误处理：仅写推理错误（推理面板红 Alert 专属展示，避免与顶部 Banner 同文案重复两遍）
      const message = error instanceof Error ? error.message : '推理过程中发生错误';
      setInferenceError(message);
      setInferenceStatus('error');
    }
  }, [
    provider,
    // ★ mode 必须进依赖：函数体内用它判断是否把结果写入本地历史（第 4 步）。
    // 缺失时闭包会一直持有挂载时的旧 mode，推理途中切换模式会把历史存错地方。
    mode,
    inference.selectedModality,
    inference.selectedModel,
    setInferenceStatus,
    setInferenceProgress,
    setInferenceError,
    setSegData,
    setSegFileInfo,
    setInferenceResult
  ]);

  /**
   * 是否允许开始推理的布尔标志
   * 需同时满足：MRI 已就绪、已选择模型、且当前不在上传/推理中（避免重复触发）。
   */
  const isReadyForInference = mriFileInfo.status === 'ready' &&
    inference.selectedModel &&
    inference.status !== 'processing' &&
    inference.status !== 'uploading';

  // 对外暴露接口
  return {
    modelList,
    inference,
    loadModels,
    runModelInference,
    isReadyForInference
  };
}
