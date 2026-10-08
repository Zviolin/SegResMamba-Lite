/**
 * ============================================================
 *  config/appConfig —— 应用级配置（消除魔法数字）
 * ============================================================
 * 集中管理：移动端断点、ORT 运行时、离线模型元信息。
 * 业务代码引用此处常量，避免散落的硬编码数字/字符串。
 */
import type { ModelInfo } from '../types';

/** ── 响应式断点 ── */
/** 移动端断点宽度（px），< 该值视为移动端布局 */
export const MOBILE_BREAKPOINT = 900;

/**
 * ── 体数据兜底维度 ──
 * 未加载 NIfTI 时各处 UI 需要一个占位形状（切片滑块上限、网格布局等），
 * 取 BraTS 常见尺寸 240×240×155，格式与 NIfTI 头部同构：[维度数, nx, ny, nz]。
 * 原先这个数组在 6 个文件里各写一遍，改尺寸要挨个改，此处收口。
 */
export const DEFAULT_VOLUME_DIMS = [0, 240, 240, 155];

/** ── ORT 运行时配置（offline/ortEnv.ts 使用）── */
export const ORT_CONFIG = {
  /** 推理线程数：安卓 WebView 无 COOP/COEP，必须单线程 */
  numThreads: 1,
  /** 是否启用 SIMD（低端设备加载 ort-wasm-simd.wasm 失败时自动降级 false） */
  simd: true,
  /** 推理移入 Worker，主线程不冻结 */
  proxy: true,
  /** WASM 静态资源目录（public 下，绝对路径） */
  wasmDir: '/vendor/ort/',
};

/** ── 离线内置模型元信息 ── */
export interface OfflineModelMeta extends ModelInfo {
  /** 模型文件路径（public 下绝对 URL） */
  file: string;
}

/** 离线可用模型列表（M2 内置，不依赖后端） */
export const OFFLINE_MODELS_META: OfflineModelMeta[] = [
  {
    modelType: 'OFFLINE_SEGRESNET_DF',
    file: '/models/segresnet_df_ema.onnx',
    modelName: 'SegResNet-DF（本地）',
    modelVersion: '1.0.0',
    description: 'DiceFocal+EMA · Dice≈0.91 · 推荐',
    supportedModalities: ['T1ce', 'FLAIR', 'T1', 'T2'],
  },
  {
    modelType: 'OFFLINE_SEGRESNET_CE',
    file: '/models/segresnet_dicece_baseline.onnx',
    modelName: 'SegResNet-CE（本地）',
    modelVersion: '1.0.0',
    description: 'DiceCE 基线 · Dice≈0.92',
    supportedModalities: ['T1ce', 'FLAIR', 'T1', 'T2'],
  },
];

/** 离线默认模型 */
export const OFFLINE_DEFAULT_MODEL = 'OFFLINE_SEGRESNET_DF';

/** modelType → 模型文件 映射（供 session 缓存按文件复用） */
export const OFFLINE_MODEL_FILES: Record<string, string> = Object.fromEntries(
  OFFLINE_MODELS_META.map((m) => [m.modelType, m.file])
);
