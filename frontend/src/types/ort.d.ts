/**
 * ============================================================
 *  ort —— ONNX Runtime Web 全局类型声明
 * ============================================================
 * ORT 运行时通过 index.html 中的 <script src="/vendor/ort/ort.min.js">
 * 以 UMD 方式加载（版本 1.17，与 www 移动端验证组合一致），
 * 挂载为 window.ort 全局对象。此处提供最小化的类型声明。
 */

/** ONNX Runtime Web 全局对象（1.17 版最小接口） */
interface OrtGlobal {
  version?: string;
  env: {
    versions?: { common?: string };
    wasm?: {
      numThreads?: number;
      simd?: boolean;
      proxy?: boolean;
      wasmPaths?: Record<string, string>;
    };
    [key: string]: unknown;
  };
  /** 创建推理会话（可传入 ArrayBuffer 或 URL） */
  InferenceSession: {
    create(
      buffer: ArrayBuffer | Uint8Array | string,
      options?: {
        executionProviders?: string[];
        graphOptimizationLevel?: string;
        [key: string]: unknown;
      }
    ): Promise<OrtSession>;
  };
  /** 张量构造器 */
  Tensor: new (
    type: string,
    data: Float32Array | number[],
    dims: number[]
  ) => OrtTensor;
  [key: string]: unknown;
}

/** 推理会话 */
export interface OrtSession {
  inputNames: string[];
  outputNames: string[];
  run(inputs: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  release?(): Promise<void>;
}

/** ONNX 张量 */
export interface OrtTensor {
  data: Float32Array | Uint8Array | number[];
  dims: number[];
  type: string;
}

declare global {
  interface Window {
    ort?: OrtGlobal;
  }
  /** 兼容直接引用（index.html 已挂载全局） */
  const ort: OrtGlobal | undefined;
}

export {};
