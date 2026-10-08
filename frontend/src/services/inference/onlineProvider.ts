/**
 * ============================================================
 *  inference/onlineProvider —— 在线推理引擎（后端 API）
 * ============================================================
 * 实现 InferenceProvider 接口，内部复用原有 apiClient 流程：
 *
 *   上传 MRI → 提交推理 → 下载掩码 → 解析为 NIfTI
 *
 * 逻辑与改造前 useInference.ts 的 runModelInference() 完全一致，
 * 仅将「流程编排」从业务 Hook 中下沉到引擎实现，达到解耦目的。
 * 因此在线模式的用户行为与改造前 100% 一致，无回归风险。
 */
import {
  getAvailableModels,
  uploadMriFile,
  runInference,
  downloadMaskFile
} from '../api/apiClient';
import { parseNiftiFile } from '../../utils/niftiParser';
import type { InferenceProvider, SegmentationInput, SegmentationResult } from './types';

/**
 * 在线推理引擎实例（单例）
 * 通过 modeStore 的 useModeProvider() 在业务层获取。
 */
export const onlineProvider: InferenceProvider = {
  id: 'online',
  label: '在线模式',

  /** 模型列表：直接请求后端 /api/models */
  async getModels() {
    return getAvailableModels();
  },

  /**
   * 在线分割完整流程（进度语义与改造前保持一致）：
   *   10% 开始上传 → 30% 上传完成 → 40% 开始推理 → 80% 推理完成 → 90% 解析完成
   */
  async runSegmentation(
    input: SegmentationInput,
    onProgress: (p: number) => void
  ): Promise<SegmentationResult> {
    // ---- 第 1 步：上传阶段 ----
    onProgress(10);
    const uploadResult = await uploadMriFile(input.mriFile);
    onProgress(30);

    // ---- 第 2 步：推理阶段 ----
    onProgress(40);
    const inferenceResult = await runInference({
      mriFilePath: uploadResult.filePath,
      modality: input.modality,
      modelType: input.modelType
    });
    onProgress(80);

    // ---- 第 3 步：下载并解析分割掩码 ----
    // 从后端下载分割结果 Blob，包装成 File（掩码为 gzip 压缩的 .nii.gz）
    const maskBlob = await downloadMaskFile(inferenceResult.maskFilePath);
    const maskFile = new File([maskBlob], 'inference_mask.nii.gz', { type: 'application/gzip' });
    const segData = await parseNiftiFile(maskFile);
    onProgress(90);

    return { segData, result: inferenceResult };
  }
};
