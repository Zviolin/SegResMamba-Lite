/**
 * ============================================================
 *  inference/types —— 推理 Provider 统一接口
 * ============================================================
 * 在线（后端 API）与离线（本地 ONNX）两种推理引擎实现同一接口，
 * 业务层（useInference）只依赖本接口，从而实现「一个前端，离/在线切换」。
 *
 * M1 里程碑：接口定义 + 在线实现 + 离线占位；
 * M2 里程碑：离线实现接入 onnxruntime-web。
 */
import type { NiftiData, ModelInfo, InferenceResult } from '../../types';

/**
 * 分割请求参数
 * 由业务层（useInference）根据当前选择组装，两种引擎共用。
 * 在线引擎使用 mriFile（单文件走后端）；离线引擎优先使用
 * modalityFiles（BraTS 4 模态，键为 t1n/t1c/t2w/t2f），
 * 未提供时回退为 mriFile + modality 单模态。
 */
export interface SegmentationInput {
  /** 主 MRI 文件（在线：后端上传用；离线：Viewer 显示底图） */
  mriFile: File;
  /** 离线模式：按模态的文件集合（键 t1n/t1c/t2w/t2f，缺失通道置零） */
  modalityFiles?: Record<string, File>;
  /** 当前选中的 MRI 模态（如 'T1ce'） */
  modality: string;
  /** 当前选中的模型类型标识（如 'SEGRESNET_MAMBA'） */
  modelType: string;
}

/**
 * 分割结果
 * 引擎执行完成后统一返回，业务层据此写入 store 并计算统计。
 */
export interface SegmentationResult {
  /** 解析后的分割掩码体数据（写入 store，用于三平面叠加显示） */
  segData: NiftiData;
  /** 推理结果信息（模型、耗时、体积等，展示在结果卡片） */
  result: InferenceResult;
}

/**
 * 推理引擎统一接口
 * 新增推理方式（例如云端 GPU 推理）时，只需再实现一个该接口的类。
 */
export interface InferenceProvider {
  /** 模式标识 */
  id: 'online' | 'offline';
  /** 模式展示名称 */
  label: string;
  /** 获取可用模型列表（在线走后端接口，离线返回内置配置） */
  getModels(): Promise<ModelInfo[]>;
  /**
   * 执行分割
   * @param input      输入参数（MRI 文件 + 模态 + 模型）
   * @param onProgress 进度回调（0-100），供 UI 进度条展示
   * @returns 分割掩码数据 + 推理结果信息
   */
  runSegmentation(
    input: SegmentationInput,
    onProgress: (p: number) => void
  ): Promise<SegmentationResult>;
}
