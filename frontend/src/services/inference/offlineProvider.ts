/**
 * ============================================================
 *  inference/offlineProvider —— 离线推理引擎（编排层）
 * ============================================================
 * 职责：实现 InferenceProvider 接口，调度下层子模块完成推理流程。
 * 本文件不写大段算法代码，复杂逻辑已下沉到 offline/ 子模块：
 *
 *   offline/ortEnv.ts             ORT 环境配置 + SIMD 降级
 *   offline/modelSessionCache.ts  模型懒加载 / session 缓存 / 释放
 *   offline/preprocess.ts          预处理（重采样/裁剪/填充/归一化，从 www 移植）
 *   offline/postprocess.ts         后处理（softmax+argmax，从 www 移植）
 *
 * 编排流程：
 *   解析 MRI → 转 D,H,W 顺序 → 多模态/单模态填通道 → prepareInput →
 *   getSession → session.run → processMultiClassOutput → 逆变换回填原始分辨率
 */
import { parseNiftiFile, getVolumeDims } from '../../utils/niftiParser';
import type { ModelInfo, InferenceResult } from '../../types';
import type { InferenceProvider, SegmentationInput, SegmentationResult } from './types';
import { OFFLINE_MODELS_META } from '../../config/appConfig';
import { getSession } from './offline/modelSessionCache';
import {
  prepareInput,
  mapModalityToChannel,
  TARGET_SIZE,
  TARGET_SPACING,
  type VolumeData,
  type OfflineModality,
} from './offline/preprocess';
import { processMultiClassOutput, countClasses, upsampleMaskToOriginal } from './offline/postprocess';

/* ───────────── 模型列表 ───────────── */

/** 暴露给推理面板的模型列表（内置配置，来自 appConfig） */
// 注意：这里从元信息里解构 `file` 字段，丢弃（不在 ModelInfo 上暴露，但模型文件路径在 OFFLINE_MODEL_FILES 表里查）。
// 用 `_file` 命名告诉 ESLint 这只是丢弃字段，避开 no-unused-vars 报警。
const OFFLINE_MODELS: ModelInfo[] = OFFLINE_MODELS_META.map(({ file: _file, ...info }) => info);

/* ───────────── 数据转换 ───────────── */

/**
 * 将 React 端 NiftiData（layout x+y*nx+z*nx*ny）转换为
 * www 格式 VolumeData（layout d*H*W+h*W+w，shape [nz,ny,nx]）。
 */
function niftiToVolume(mriData: Awaited<ReturnType<typeof parseNiftiFile>>): VolumeData {
  const dims = getVolumeDims(mriData);
  const nx = dims[1];
  const ny = dims[2];
  const nz = dims[3];
  const pix = mriData.header.pixDims;
  const sx = pix[1] || 1.0;
  const sy = pix[2] || 1.0;
  const sz = pix[3] || 1.0;

  const src = mriData.typedArray;
  const data = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        data[z * ny * nx + y * nx + x] = src[x + y * nx + z * nx * ny];
      }
    }
  }
  return { data, shape: [nz, ny, nx], spacing: [sz, sy, sx] };
}

/* ───────────── 离线引擎实现（编排） ───────────── */

export const offlineProvider: InferenceProvider = {
  id: 'offline',
  label: '离线模式',

  /** 模型列表：返回内置的本地模型配置 */
  async getModels(): Promise<ModelInfo[]> {
    return OFFLINE_MODELS;
  },

  /**
   * 本地 ONNX 分割完整流程（编排，算法细节在 offline/ 子模块）
   */
  async runSegmentation(
    input: SegmentationInput,
    onProgress: (p: number) => void
  ): Promise<SegmentationResult> {
    const startTime = Date.now();

    // ── 第 1 步：解析主 MRI 文件（用于 Viewer 底图与 segData 尺寸）──
    onProgress(5);
    const mriData = await parseNiftiFile(input.mriFile);
    const dims = getVolumeDims(mriData);
    const nx = dims[1];
    const ny = dims[2];
    const nz = dims[3];

    // ── 第 2 步：预处理（offline/preprocess）──
    onProgress(20);
    // 多模态输入：优先使用 modalityFiles（BraTS 4 模态，缺失通道置零）
    const volumes: Partial<Record<OfflineModality, VolumeData>> = {};
    const modFiles = input.modalityFiles || {};
    const modKeys = Object.keys(modFiles).filter(k => modFiles[k]);
    if (modKeys.length > 0) {
      for (const modKey of modKeys) {
        const mod = modKey as OfflineModality;
        const modData = await parseNiftiFile(modFiles[modKey]);
        volumes[mod] = niftiToVolume(modData);
      }
    }
    // 未提供多模态 → 回退单模态（主文件 + 所选模态）
    if (Object.keys(volumes).length === 0) {
      const channel = mapModalityToChannel(input.modality);
      volumes[channel] = niftiToVolume(mriData);
    }
    const { tensor: inputTensor, transform } = prepareInput(volumes);
    onProgress(35);

    // ── 第 3 步：加载模型并推理（offline/modelSessionCache）──
    const session = await getSession(input.modelType);
    onProgress(45);

    // 输入 / 输出张量名以模型实际声明为准（此前写死 'input' / 'output'，
    // 换成 IO 名不同的模型会直接抛 undefined）。
    const inputName = session.inputNames?.[0] ?? 'input';
    const outputName = session.outputNames?.[0] ?? 'output';

    const tensor = new (window.ort!.Tensor)('float32', inputTensor, [1, 4, TARGET_SIZE, TARGET_SIZE, TARGET_SIZE]);
    const outputs = await session.run({ [inputName]: tensor });
    onProgress(80);

    const outputData = outputs[outputName]?.data as Float32Array | undefined;
    if (!outputData) {
      throw new Error(`模型输出 "${outputName}" 不存在（实际输出：${session.outputNames?.join(', ') || '未知'}）`);
    }

    // ── 第 4 步：后处理（offline/postprocess）──
    const labels = processMultiClassOutput(outputData);   // Uint8Array 64³
    onProgress(88);

    // 只解构用到的三类（背景 c0 不需要）
    const [, c1, c2, c3] = countClasses(labels);
    const voxelCm3 = (TARGET_SPACING * TARGET_SPACING * TARGET_SPACING) / 1000; // 8mm³ = 0.008cm³
    const wt = (c1 + c2 + c3) * voxelCm3;
    const tc = (c1 + c3) * voxelCm3;
    const et = c3 * voxelCm3;

    // ── 第 5 步：逆变换回填原始分辨率，构造 segData ──
    const segArray = upsampleMaskToOriginal(labels, nx, ny, nz, transform);
    const segData = {
      header: {
        dims: [3, nx, ny, nz],
        pixDims: mriData.header.pixDims,
        datatypeCode: 2, // UINT8
      },
      image: segArray.buffer as ArrayBuffer,
      typedArray: segArray,
    };
    onProgress(95);

    const result: InferenceResult = {
      maskFilePath: 'local://offline-segmentation',
      modelUsed: OFFLINE_MODELS_META.find(m => m.modelType === input.modelType)?.modelName ?? 'SegResNet-DF（本地）',
      executionTimeMs: Date.now() - startTime,
      volumes: {
        WT: Number(wt.toFixed(2)),
        TC: Number(tc.toFixed(2)),
        ET: Number(et.toFixed(2)),
      },
    };
    onProgress(100);

    return { segData, result };
  },
};
