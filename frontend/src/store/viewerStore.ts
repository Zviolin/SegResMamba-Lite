/** viewerStore：MRI/标签体数据 / 视图 / 切片 / 统计 / 全局状态 / 模型 / 推理；组件与 hooks 通过 useViewerStore 统一读写。 */
import { create } from 'zustand';
import type { NiftiData, ViewSettings, SlicePosition, TumorStatistics, FileInfo, ModelInfo, InferenceState, InferenceResult } from '../types';

/** 状态结构定义：上半是数据字段，下半 actions 是更新方法。 */
interface ViewerState {
  /** 解析后的 MRI 体数据（未加载时为 null） */
  mriData: NiftiData | null;
  /** 解析后的分割标签体数据（未加载时为 null） */
  segData: NiftiData | null;
  /** 用户选中的原始 MRI File（仅内存，供运行推理；不可序列化） */
  mriFile: File | null;
  /** MRI 文件信息（文件名、大小、加载状态） */
  mriFileInfo: FileInfo;
  /** 分割标签文件信息 */
  segFileInfo: FileInfo;

  /** 视图显示设置（窗宽窗位、透明度、区域显隐） */
  viewSettings: ViewSettings;
  /** 三平面当前切片位置（轴向/冠状/矢状） */
  slicePosition: SlicePosition;

  /** 肿瘤统计结果（未计算时为 null） */
  statistics: TumorStatistics | null;

  /** 是否处于全局加载中（用于遮罩/禁用） */
  isLoading: boolean;
  /** 全局错误信息（无错误时为 null） */
  error: string | null;

  /** 可用模型列表 */
  modelList: ModelInfo[];
  /** 推理状态机（阶段、模型、模态、结果、进度） */
  inference: InferenceState;

  /** 全部状态更新方法集合 */
  actions: {
    /** 写入 MRI 体数据 */
    setMriData: (data: NiftiData) => void;
    /** 写入分割标签体数据 */
    setSegData: (data: NiftiData) => void;
    /** 写入用户选择的 MRI File 对象（不持久化） */
    setMriFile: (file: File | null) => void;
    /** 部分更新 MRI 文件信息（合并旧值） */
    setMriFileInfo: (info: Partial<FileInfo>) => void;
    /** 部分更新分割标签文件信息（合并旧值） */
    setSegFileInfo: (info: Partial<FileInfo>) => void;

    /** 部分更新视图设置（合并旧值） */
    updateViewSettings: (settings: Partial<ViewSettings>) => void;
    /** 部分更新三平面切片位置（合并旧值） */
    setSlicePosition: (position: Partial<SlicePosition>) => void;

    /** 写入肿瘤统计结果 */
    setStatistics: (stats: TumorStatistics) => void;

    /** 设置全局加载状态 */
    setLoading: (loading: boolean) => void;
    /** 设置全局错误信息 */
    setError: (error: string | null) => void;

    /** 写入模型列表 */
    setModelList: (models: ModelInfo[]) => void;
    /** 设置当前选中的模型类型 */
    setSelectedModel: (modelType: string) => void;
    /** 设置当前选中的 MRI 模态 */
    setSelectedModality: (modality: string) => void;
    /** 设置推理阶段状态 */
    setInferenceStatus: (status: InferenceState['status']) => void;
    /** 写入推理结果 */
    setInferenceResult: (result: InferenceResult | null) => void;
    /** 写入推理错误信息 */
    setInferenceError: (error: string | null) => void;
    /** 设置推理进度（0-100） */
    setInferenceProgress: (progress: number) => void;
    /** 重置推理状态为初始值 */
    resetInference: () => void;

  };
}

/** 视图设置的初始值：默认窗宽 800、窗位 400，透明度 50%，三个区域全部显示 */
const initialViewSettings: ViewSettings = {
  windowWidth: 800,
  windowLevel: 400,
  opacity: 50,
  showNecrosis: true,
  showEdema: true,
  showEnhancing: true
};

/** 视图设置持久化键名（M4：跨会话保留窗宽窗位等显示偏好） */
const VIEW_SETTINGS_KEY = 'brainseg-view-settings';

/** 读取初始视图设置：优先取持久化值，否则用默认值 */
/**
 * 把数值夹取到 [min, max] 区间，非数值时回退到 fallback
 *
 * 用途：localStorage 里读回来的值不可信（可能是手改的、也可能是旧版本写错的）。
 * 典型坑：windowWidth 若为 0，窗宽窗位算法会除零得到 NaN，
 * 写入 Uint8ClampedArray 后变成 0 —— 整个视图全黑且不报任何错。
 */
function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === 'number' ? value : Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(Math.max(n, min), max);
}

function readInitialViewSettings(): ViewSettings {
  try {
    const saved = localStorage.getItem(VIEW_SETTINGS_KEY);
    if (saved) {
      const merged = { ...initialViewSettings, ...JSON.parse(saved) } as ViewSettings;
      // 逐项夹取到合法范围：窗宽至少 1（防除零），其余按 UI 滑块范围收敛
      return {
        ...merged,
        windowWidth: clampNumber(merged.windowWidth, 1, 4000, initialViewSettings.windowWidth),
        windowLevel: clampNumber(merged.windowLevel, 0, 2000, initialViewSettings.windowLevel),
        opacity: clampNumber(merged.opacity, 0, 100, initialViewSettings.opacity),
      };
    }
  } catch {
    // 持久化数据损坏时回退默认
  }
  return { ...initialViewSettings };
}

/** 切片位置的初始值：三个平面都从第 0 层开始（加载文件后会被重置到中间层） */
const initialSlicePosition: SlicePosition = {
  axial: 0,
  coronal: 0,
  sagittal: 0
};

/** 文件信息的初始值：无文件名、0 字节、idle 状态 */
const initialFileInfo: FileInfo = {
  name: '',
  size: 0,
  status: 'idle'
};

/** 推理状态的初始值：idle、未选模型、默认模态 T1ce、无结果、进度 0 */
const initialInferenceState: InferenceState = {
  status: 'idle',
  selectedModel: '',
  selectedModality: 'T1ce',
  result: null,
  error: null,
  progress: 0
};

/** 创建 Zustand store；actions 在 create() 内一次性创建、引用稳定，避免 selector 误判重渲染。 */
export const useViewerStore = create<ViewerState>((set) => ({
  // ---- 数据字段初始值 ----
  mriData: null,
  segData: null,
  mriFile: null,
  // 用展开运算符复制初始对象，避免多处以同一引用共享被误改
  mriFileInfo: { ...initialFileInfo },
  segFileInfo: { ...initialFileInfo },

  viewSettings: readInitialViewSettings(),
  slicePosition: { ...initialSlicePosition },

  statistics: null,

  isLoading: false,
  error: null,

  modelList: [],
  inference: { ...initialInferenceState },

  // ---- actions ----
  actions: {
    // 直接覆盖整个数据对象
    setMriData: (data) => set({ mriData: data }),
    setSegData: (data) => set({ segData: data }),
    setMriFile: (file) => set({ mriFile: file }),

    // 部分更新：合并旧状态与传入的新字段，保留未涉及的字段
    setMriFileInfo: (info) => set((state) => ({
      mriFileInfo: { ...state.mriFileInfo, ...info }
    })),

    setSegFileInfo: (info) => set((state) => ({
      segFileInfo: { ...state.segFileInfo, ...info }
    })),

    updateViewSettings: (settings) => set((state) => {
      const next = { ...state.viewSettings, ...settings };
      // M4：视图设置持久化到 localStorage，跨会话保留显示偏好
      try { localStorage.setItem(VIEW_SETTINGS_KEY, JSON.stringify(next)); } catch { /* 忽略 */ }
      return { viewSettings: next };
    }),

    setSlicePosition: (position) => set((state) => ({
      slicePosition: { ...state.slicePosition, ...position }
    })),

    setStatistics: (stats) => set({ statistics: stats }),

    setLoading: (loading) => set({ isLoading: loading }),
    setError: (error) => set({ error }),

    setModelList: (models) => set({ modelList: models }),

    // 以下均只更新 inference 对象内的某个字段（先展开再覆盖）
    setSelectedModel: (modelType) => set((state) => ({
      inference: { ...state.inference, selectedModel: modelType }
    })),
    setSelectedModality: (modality) => set((state) => ({
      inference: { ...state.inference, selectedModality: modality }
    })),
    setInferenceStatus: (status) => set((state) => ({
      inference: { ...state.inference, status }
    })),
    setInferenceResult: (result) => set((state) => ({
      inference: { ...state.inference, result }
    })),
    setInferenceError: (error) => set((state) => ({
      inference: { ...state.inference, error }
    })),
    setInferenceProgress: (progress) => set((state) => ({
      inference: { ...state.inference, progress }
    })),
    // 整个推理状态重置为初始值
    resetInference: () => set({ inference: { ...initialInferenceState } }),

  },
}));
