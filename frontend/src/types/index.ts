/**
 * ============================================================
 *  类型定义集中文件（全局共享类型）
 * ============================================================
 * 本文件集中定义了整个前端项目在 NIfTI 数据解析、视图状态管理、
 * 推理流程、历史记录与异步任务等模块中共享的 TypeScript 类型。
 * 所有组件 / hooks / store / utils 均从这里导入类型，保证类型一致。
 */

/**
 * NIfTI 文件头信息
 * 描述医学影像数据的元数据，包含数据维度、体素数据类型、体素物理间距等关键参数。
 * 由 nifti-reader-js 的 readHeader() 解析而来。
 */
export interface NiftiHeader {
  /**
   * 数据维度数组。
   * 约定下标含义：dims[0] 为总维度个数，dims[1]=X 方向宽度(nx)、
   * dims[2]=Y 方向高度(ny)、dims[3]=Z 方向层数(nz)。
   * 典型脑 MRI 为 [3, 240, 240, 155]。
   */
  dims: number[];
  /**
   * NIfTI 标准体素数据类型编码。
   * 例如：2=UINT8、4=INT16、8=INT32、16=FLOAT32、64=FLOAT64 等。
   * getTypedData() 根据该编码决定用哪种 TypedArray 解读原始字节。
   */
  datatypeCode: number;
  /**
   * 体素物理尺寸（体素间距），单位毫米。
   * pixDims[1]=X 方向、pixDims[2]=Y 方向、pixDims[3]=Z 方向。
   * 三个值相乘即单个体素的体积，用于计算肿瘤体积。
   */
  pixDims: number[];
  /* nifti-reader-js 解析出的其它常用头字段（按需强类型声明，禁止 [key: string]: any 这种全开放索引） */
  /** 仿射变换矩阵（nifti-reader-js 里为 4x4 嵌套数组） */
  affine?: number[][];
  /** qform / sform 编码（0=未知 1=scanner 等） */
  qform_code?: number;
  sform_code?: number;
  /** NIfTI 魔数（"n+1" / "ni1"），用于校验文件格式 */
  magic?: string;
  /** 头部字节数（NIfTI-1 恒为 348） */
  sizeof_hdr?: number;
}

/**
 * 类型化数组联合类型
 * 覆盖 NIfTI 文件可能使用的全部数值数据类型。
 * 统一用 TypedArray 别名，避免在每个函数里重复写联合类型。
 */
export type TypedArray = Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array;

/**
 * 解析完成后的 NIfTI 数据载体
 * 同时保存原始字节缓冲、头部信息和便于索引访问的类型化数组。
 */
export interface NiftiData {
  /** NIfTI 头部信息（维度 / 数据类型 / 体素间距等） */
  header: NiftiHeader;
  /** 原始图像字节缓冲（ArrayBuffer） */
  image: ArrayBuffer;
  /**
   * 依据 header.datatypeCode 构造的类型化数组，
   * 是后续按体素读取灰度值 / 标签值的主要数据源。
   */
  typedArray: TypedArray;
}

/**
 * 视图平面类型
 * 医学影像 MPR（多平面重建）的三种正交切面：
 * axial=轴位（横断面，沿 Z 轴切片）、coronal=冠状位（沿 Y 轴切片）、sagittal=矢状位（沿 X 轴切片）。
 */
export type PlaneType = 'axial' | 'coronal' | 'sagittal';

/**
 * 视图显示设置
 * 控制窗宽窗位、分割叠加透明度以及各肿瘤区域是否显示。
 */
export interface ViewSettings {
  /** 窗宽（Window Width）：决定对比度，值越小对比度越高 */
  windowWidth: number;
  /** 窗位（Window Level）：决定亮度，值越大图像越亮 */
  windowLevel: number;
  /** 分割叠加透明度（0-100，百分数），数值越大颜色覆盖越不透明 */
  opacity: number;
  /** 是否显示坏死核心（NCR，标签值 1） */
  showNecrosis: boolean;
  /** 是否显示水肿区（ED，标签值 2） */
  showEdema: boolean;
  /** 是否显示增强肿瘤（ET，标签值 3 或 4） */
  showEnhancing: boolean;
}

/**
 * 三平面当前切片位置
 * 记录三个正交视图各自的切片索引，用于三视图联动与十字线定位。
 */
export interface SlicePosition {
  /** 轴位切片索引（沿 Z 轴，0 ~ nz-1） */
  axial: number;
  /** 冠状位切片索引（沿 Y 轴，0 ~ ny-1） */
  coronal: number;
  /** 矢状位切片索引（沿 X 轴，0 ~ nx-1） */
  sagittal: number;
}

/**
 * 肿瘤统计信息
 * 由 calculateTumorStatistics() 对分割标签逐体素统计得出，
 * 供分析报告、体积展示等使用。
 */
export interface TumorStatistics {
  /** 肿瘤总体素数量（NCR + ED + ET） */
  totalVoxels: number;
  /** 坏死核心（标签值 1）体素数 */
  necrosisVoxels: number;
  /** 水肿区（标签值 2）体素数 */
  edemaVoxels: number;
  /** 增强肿瘤（标签值 3 或 4）体素数 */
  enhancingVoxels: number;
  /** 坏死核心体积（mm³ = 体素数 × 单个体素体积） */
  necrosisVolume: number;
  /** 水肿区体积（mm³） */
  edemaVolume: number;
  /** 增强肿瘤体积（mm³） */
  enhancingVolume: number;
  /** 肿瘤总体积（mm³） */
  totalVolume: number;
  /** 包含肿瘤的切片张数 */
  tumorSlices: number;
  /** 含肿瘤切片的起始与结束索引范围，例如 [10, 120] */
  sliceRange: [number, number];
}

/**
 * 文件加载状态信息
 * 用于跟踪 MRI / 分割标签文件的上传与解析状态。
 */
export interface FileInfo {
  /** 文件名 */
  name: string;
  /** 文件大小（字节） */
  size: number;
  /** 文件状态：idle=未选择、loading=解析中、ready=就绪、error=解析失败 */
  status: 'idle' | 'loading' | 'ready' | 'error';
}

/**
 * 可用模型信息
 * 由后端 /api/models 接口返回，描述可用的分割模型。
 */
export interface ModelInfo {
  /** 模型类型标识（作为唯一 key / 请求参数，如 'SEGRESNET_MAMBA'） */
  modelType: string;
  /** 模型显示名称（如 'SegResNet-Mamba'） */
  modelName: string;
  /** 模型版本号 */
  modelVersion: string;
  /** 模型功能描述 */
  description: string;
  /** 该模型支持的 MRI 模态列表（如 ['T1ce','FLAIR','T1','T2']） */
  supportedModalities: string[];
}

/**
 * 推理结果
 * 后端推理完成后返回的核心结果对象。
 */
export interface InferenceResult {
  /** 分割掩码文件路径（用于下载 / 解析） */
  maskFilePath: string;
  /** 实际使用的模型标识 */
  modelUsed: string;
  /** 推理耗时（毫秒） */
  executionTimeMs: number;
  /** 各区域体积（cm³）的映射，如 { 'WT': 123.45, 'TC': 45.6, 'ET': 12.3 } */
  volumes: Record<string, number>;
}

/**
 * 推理请求参数
 * 提交给后端进行模型推理所需的参数。
 */
export interface InferenceRequest {
  /** MRI 文件在服务端的路径（由上传接口返回） */
  mriFilePath: string;
  /** 使用的 MRI 模态（如 'T1ce'） */
  modality: string;
  /** 使用的模型类型（可选，缺省时由后端决定） */
  modelType?: string;
}

/**
 * 推理流程状态枚举
 * 描述前端推理按钮从点击到完成 / 失败的生命周期。
 */
export type InferenceStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';

/**
 * 推理状态机数据
 * 集中保存推理流程相关的全部状态。
 */
export interface InferenceState {
  /** 当前推理阶段状态 */
  status: InferenceStatus;
  /** 当前选中的模型类型 */
  selectedModel: string;
  /** 当前选中的 MRI 模态 */
  selectedModality: string;
  /** 推理结果（完成后非空） */
  result: InferenceResult | null;
  /** 推理错误信息（出错时非空） */
  error: string | null;
  /** 推理进度（0-100），用于进度条显示 */
  progress: number;
}

// ── Java 中间层新增类型 ──

/**
 * 历史记录
 * 由 Java 中间层 /api/history 接口返回，记录每次推理历史。
 */
export interface HistoryRecord {
  /** 历史记录主键 ID */
  id: number;
  /** 关联的异步任务 ID（同步推理可为 null） */
  jobId: string | null;
  /** 原始 MRI 文件名 */
  fileName: string;
  /** 模型类型 */
  modelType: string;
  /** 实际使用的模型名称 */
  modelUsed: string;
  /** 推理耗时（毫秒） */
  executionTimeMs: number;
  /** 各区域体积（cm³）映射，可空 */
  volumes: Record<string, number> | null;
  /** 状态（如 'COMPLETED' / 'FAILED'） */
  status: string;
  /** 失败原因（成功时为空） */
  error: string | null;
  /** 创建时间（ISO 字符串） */
  createdAt: string;
}

/**
 * 异步任务状态
 * 由 Java 中间层 /api/jobs 接口返回，描述异步推理任务的实时状态。
 */
export interface JobStatus {
  /** 任务唯一 ID */
  jobId: string;
  /** 任务状态：排队中 / 运行中 / 已完成 / 失败 / 超时 */
  status: 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'TIMEOUT';
  /** 任务进度（0-100） */
  progress: number;
  /** 任务结果（完成后非空） */
  result: InferenceResult | null;
  /** 错误信息（失败时非空） */
  error: string | null;
  /** 创建时间（ISO 字符串） */
  createdAt: string;
  /** 完成时间（尚未完成时为 null） */
  finishedAt: string | null;
}
