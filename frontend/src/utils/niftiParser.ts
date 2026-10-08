/**
 * ============================================================
 *  NIfTI 文件解析与切片处理工具
 * ============================================================
 * 本模块提供三大能力：
 *  1. parseNiftiFile —— 把用户选择的 .nii / .nii.gz 文件解析为 NiftiData；
 *  2. getTypedData —— 根据 NIfTI 头部的数据类型编码，把原始字节转成可索引的类型化数组；
 *  3. getSliceData —— 从三维体数据中按指定平面（轴向/冠状/矢状）抽取一张二维切片；
 *  4. applyWindowLevel —— 窗宽窗位算法，把原始体素值映射为 0~255 的灰度，用于 Canvas 渲染；
 *  5. getVolumeDims / getMaxSlices —— 体数据维度与三平面最大切片索引的统一取法。
 */
import * as nifti from 'nifti-reader-js';
import pako from 'pako';
import { DEFAULT_VOLUME_DIMS } from '../config/appConfig';
import type { NiftiData, NiftiHeader, TypedArray } from '../types';

/**
 * 解析 NIfTI 文件（支持 .nii 与 .nii.gz 压缩格式）
 *
 * @param file 用户选择的文件对象
 * @returns Promise<NiftiData> 解析成功后的数据（头部 + 原始字节 + 类型化数组）
 * @throws 当文件不是有效 NIfTI 或读取失败时抛出 Error
 *
 * 核心步骤：
 *  1. 用 FileReader 把文件读为 ArrayBuffer；
 *  2. 若为 gzip 压缩（.nii.gz），先解压成原始字节流；
 *  3. 校验是否为合法 NIfTI，读取头部与图像字节；
 *  4. 依据头部 datatypeCode 构造类型化数组，方便后续按体素取值。
 */
/**
 * 比较两个体数据的三维形状是否一致（只比对 dims[1..3]，忽略表示维度个数的 dims[0]）
 *
 * 用途：叠加显示前校验「分割标签」与「MRI」是否为同一套体数据的相同分辨率，
 * 形状不一致时叠加会把标签画到错误的解剖位置上（不报错，但结果是错的）。
 */
export function sameShape(a?: number[], b?: number[]): boolean {
  return !!a && !!b && a[1] === b[1] && a[2] === b[2] && a[3] === b[3];
}

/**
 * 把 sliceIndex 夹取到合法范围 [0, upper-1]
 *
 * 越界的切片索引会让 data[idx] 取到 undefined，经窗宽窗位运算变成 NaN，
 * 最终写入 Uint8ClampedArray 变成 0 —— 表现为「黑屏但不报错」，极难排查。
 * 这里在源头夹住，保证任何调用方传进来的索引都安全。
 */
function clampIndex(sliceIndex: number, upper: number): number {
  if (!Number.isFinite(sliceIndex)) return 0;
  const max = Math.max(0, upper - 1);
  return Math.min(Math.max(Math.trunc(sliceIndex), 0), max);
}

export async function parseNiftiFile(file: File): Promise<NiftiData> {
  // 空文件不必交给解析器兜圈子
  if (!file || file.size === 0) {
    throw new Error('文件为空，无法解析');
  }
  // 复用下面的统一解析入口（同一套三层解压兜底链），
  // 避免「上传走一层解压、历史回放走三层兜底」导致同一份文件两处行为不一致。
  const buf = await file.arrayBuffer();
  return parseNiftiFromGzData(new Uint8Array(buf));
}

/**
 * 从 Blob / Uint8Array（可能为 .nii.gz 压缩字节或裸 .nii 字节）解析为 NiftiData
 *
 * 主要用于：本地历史记录里保存的 segDataGz Blob（来自 IndexedDB），
 * 在历史预览/还原分割结果时直接拿到 NiftiData，无需再走 File + FileReader 流程。
 *
 * @param data  Blob / Uint8Array：gzip 压缩或裸 NIfTI 都可
 * @returns Promise<NiftiData> 解析成功后的数据
 *
 * 处理顺序：
 *  1. 统一把 Blob 转成 Uint8Array；
 *  2. 用 nifti-reader-js 的 isCompressed 判定是否为 gzip 并尝试解压；
 *  3. 若仍不是裸 NIfTI，则用 pako 兜底解压；
 *  4. 最终的字节流用 readHeader + readImage + getTypedData 解析。
 */
export async function parseNiftiFromGzData(data: Blob | Uint8Array): Promise<NiftiData> {
  // 1. Blob → Uint8Array 统一形态（保留 gzip 字节不动）
  let bytes: Uint8Array;
  if (data instanceof Blob) {
    const buf = await data.arrayBuffer();
    bytes = new Uint8Array(buf);
  } else {
    bytes = data;
  }

  // 把 Uint8Array 安全转成 ArrayBuffer（保留偏移，避开共享缓冲陷阱）
  const toAB = (u: Uint8Array): ArrayBuffer => {
    const out = new ArrayBuffer(u.byteLength);
    new Uint8Array(out).set(u);
    return out;
  };

  let raw: ArrayBuffer = toAB(bytes);

  // 2. 先按 nifti-reader-js 自带的判别逻辑（识别 gzip magic 1F 8B）
  const tryDecompressNiftiReader = (buf: ArrayBuffer): ArrayBuffer | null => {
    try {
      if (nifti.isCompressed(buf)) return nifti.decompress(buf);
    } catch { /* ignore */ }
    return null;
  };

  // 3. 兜底用 pako 解 gzip；pako 不接受 ArrayBuffer，故传 Uint8Array
  const tryPakoInflate = (u: Uint8Array): Uint8Array | null => {
    try {
      // 试 raw 模式（无头 2 字节）
      return pako.inflateRaw(u);
    } catch {
      try {
        // 退到 zlib 头/普通 gzip 头
        return pako.inflate(u);
      } catch { return null; }
    }
  };

  // 解压尝试链
  let decompressed = tryDecompressNiftiReader(raw);
  if (!decompressed) {
    const inflated = tryPakoInflate(bytes);
    if (inflated) decompressed = toAB(inflated);
  }

  // 判定：是否已经是裸 NIfTI���不是则用解压结果覆盖 raw
  if (nifti.isNIFTI(raw)) {
    // raw 即裸 NIfTI，什么也不做
  } else if (decompressed && nifti.isNIFTI(decompressed)) {
    raw = decompressed;
  } else {
    throw new Error('不是有效的 NIfTI 文件（含 gzip 解压后仍非合法 NIfTI）');
  }

  const header = nifti.readHeader(raw) as NiftiHeader;

  // 维度个数 dims[0] > 3 说明是 4D（含时间序列/多帧）数据。
  // 按 3D 解读会静默只读到第一帧而结果看似正常，因此显式拒绝并给出原因。
  if ((header?.dims?.[0] ?? 0) > 3) {
    throw new Error('暂不支持 4D NIfTI（含时间序列或多帧），请提供三维体数据');
  }

  const image = nifti.readImage(header as unknown as nifti.NIFTI1, raw);
  const typedArray = getTypedData(image, header);
  return { header, image, typedArray };
}

/**
 * 依据 NIfTI 头部数据类型编码，把图像原始字节包装成对应的类型化数组
 *
 * @param image  图像像素字节缓冲（ArrayBuffer）
 * @param header NIfTI 头部信息，主要用到 datatypeCode
 * @returns 与数据类型匹配的 TypedArray；未知类型时回退为 Uint8Array
 *
 * 为什么需要这一步：NIfTI 原始字节只是一串二进制，需要知道每个体素占几个字节、
 * 是有符号还是无符号、是整数还是浮点，才能正确解读灰度值 / 标签值。
 */
export function getTypedData(image: ArrayBuffer, header: NiftiHeader): TypedArray {
  // 空数据保护：图像或头部缺失时返回空数组，避免后续越界
  if (!image || !header) return new Uint8Array(0);

  // NIfTI 标准 datatypeCode -> 对应 TypedArray 构造器的映射表
  // 采用工厂函数（() => new XxxArray(image)）延迟创建，避免一次性构造所有类型
  const dataTypeMap: Record<number, () => TypedArray> = {
    [nifti.NIFTI1.TYPE_INT16]: () => new Int16Array(image),     // 4: 16 位有符号整数
    [nifti.NIFTI1.TYPE_INT32]: () => new Int32Array(image),     // 8: 32 位有符号整数
    [nifti.NIFTI1.TYPE_FLOAT32]: () => new Float32Array(image), // 16: 32 位浮点数（MRI 常用）
    [nifti.NIFTI1.TYPE_FLOAT64]: () => new Float64Array(image), // 64: 64 位浮点数
    [nifti.NIFTI1.TYPE_INT8]: () => new Int8Array(image),       // 256: 8 位有符号整数
    [nifti.NIFTI1.TYPE_UINT8]: () => new Uint8Array(image),     // 2: 8 位无符号整数
    [nifti.NIFTI1.TYPE_UINT16]: () => new Uint16Array(image),   // 512: 16 位无符号整数
    [nifti.NIFTI1.TYPE_UINT32]: () => new Uint32Array(image),   // 768: 32 位无符号整数
  };

  // 按头部记录的数据类型查找构造器；找不到则回退为 Uint8Array（按单字节解读）
  const converter = dataTypeMap[header.datatypeCode];
  return converter ? converter() : new Uint8Array(image);
}

/**
 * 窗宽窗位（Window / Level）映射算法
 *
 * 把原始体素灰度值线性映射到 0~255 的显示灰度范围，实现"调窗"效果。
 *
 * @param value        原始体素值
 * @param windowWidth  窗宽：决定映射的对比度区间宽度（值越大对比度越低）
 * @param windowLevel  窗位：决定映射区间中心（值越大图像越亮）
 * @returns 0~255 之间的灰度值，供 Canvas ImageData 使用
 *
 * 算法原理：
 *   - 映射区间为 [windowLevel - windowWidth/2, windowLevel + windowWidth/2]；
 *   - 区间下限 low 以下的体素全部压到 0（纯黑），上限 high 以上的全部压到 255（纯白）；
 *   - 区间内的体素按线性比例放大到 0~255。
 */
export function applyWindowLevel(
  value: number,
  windowWidth: number,
  windowLevel: number
): number {
  // 窗宽非法（0 / 负数 / NaN）时映射区间会退化，(value-low)/(high-low) 得到 NaN，
  // 写入 Uint8ClampedArray 后表现为整屏全黑且没有任何报错。
  // 这里兜一个最小窗宽，宁可显示成高对比的黑白图，也不要 silent black screen。
  const width = Number.isFinite(windowWidth) && windowWidth > 0 ? windowWidth : 1;
  // 同理兜住非法的窗位
  const level = Number.isFinite(windowLevel) ? windowLevel : 0;

  // 计算映射区间的下限与上限
  const low = level - width / 2;
  const high = level + width / 2;

  // 线性归一化：(value - low) / (high - low) 得到 0~1 比例，再乘 255 得到灰度
  const pixel = ((value - low) / (high - low)) * 255;
  // 钳位到 [0, 255]，保证 ImageData 通道取值合法
  return Math.max(0, Math.min(255, pixel));
}

/**
 * 从三维体数据中抽取一张指定平面的二维切片
 *
 * @param data        体数据的类型化数组（按 X 优先、再 Y、再 Z 的顺序存储，即 idx = z*ny*nx + y*nx + x）
 * @param dims        头部维度数组 [total, nx, ny, nz]
 * @param sliceIndex  要抽取的切片索引（在不同平面下代表不同的轴坐标，见各 case）
 * @param plane       平面类型：axial=轴位、coronal=冠状位、sagittal=矢状位
 * @returns 包含像素数组（按行优先排列）及切片宽高的对象，可直接喂给 Canvas
 *
 * 索引计算说明（以线性存储 idx = z*(nx*ny) + y*nx + x 为基础）：
 *   - axial（轴位）   ：固定 Z=sliceIndex，遍历 X、Y。此时每个像素 idx = sliceIndex*(nx*ny) + y*nx + x；
 *   - coronal（冠状位）：固定 Y=sliceIndex，遍历 X、Z。每个像素 idx = z*(nx*ny) + sliceIndex*nx + x；
 *   - sagittal（矢状位）：固定 X=sliceIndex，遍历 Y、Z。每个像素 idx = z*(nx*ny) + y*nx + sliceIndex。
 * 其中冠状/矢状位对 Z 从大到小（nz-1 → 0）遍历，是为了让图像上下方向符合医学影像显示习惯。
 */
export function getSliceData(
  data: TypedArray,
  dims: number[],
  sliceIndex: number,
  plane: 'axial' | 'coronal' | 'sagittal'
): { pixels: number[], width: number, height: number } {
  // 读取三维各方向尺寸；若头缺失则回退到常见的脑 MRI 尺寸
  const nx = dims[1] || DEFAULT_VOLUME_DIMS[1];   // X 方向宽度
  const ny = dims[2] || DEFAULT_VOLUME_DIMS[2];   // Y 方向高度
  const nz = dims[3] || DEFAULT_VOLUME_DIMS[3];   // Z 方向层数

  let width = 0;
  let height = 0;
  const pixels: number[] = [];

  // 按不同平面分别计算切片尺寸与像素索引
  switch (plane) {
    case 'axial': {
      // 轴位切片：宽度 = X 方向，高度 = Y 方向
      width = nx;
      height = ny;
      // 轴位沿 Z 轴切片：先把索引夹到 [0, nz-1]，避免越界读到 undefined
      const z = clampIndex(sliceIndex, nz);
      const sliceSize = nx * ny; // 单层切片的体素数
      for (let i = 0; i < sliceSize; i++) {
        // 固定 Z=z，按 Y 方向外循环、X 方向内循环逐体素取出
        pixels.push(data[z * sliceSize + i]);
      }
      break;
    }

    case 'coronal': {
      // 冠状位切片：宽度 = X 方向，高度 = Z 方向（左右为 X，上下为 Z）
      width = nx;
      height = nz;
      // 从顶层(z=nz-1)到底层(z=0)遍历，保证图像上下方向正确
      // 冠状位沿 Y 轴切片：先把索引夹到 [0, ny-1]
      const y = clampIndex(sliceIndex, ny);
      for (let z = nz - 1; z >= 0; z--) {
        for (let x = 0; x < nx; x++) {
          // 固定 Y=y：idx = z*(nx*ny) + y*nx + x
          const idx = z * nx * ny + y * nx + x;
          pixels.push(data[idx]);
        }
      }
      break;
    }

    case 'sagittal': {
      // 矢状位切片：宽度 = Y 方向，高度 = Z 方向（左右为 Y，上下为 Z）
      width = ny;
      height = nz;
      // 矢状位沿 X 轴切片：先把索引夹到 [0, nx-1]
      const x = clampIndex(sliceIndex, nx);
      for (let z = nz - 1; z >= 0; z--) {
        // Y 方向也做翻转（y 从 ny-1 到 0），让图像左右符合医学影像惯例
        for (let y = ny - 1; y >= 0; y--) {
          // 固定 X=x：idx = z*(nx*ny) + y*nx + x
          const idx = z * nx * ny + y * nx + x;
          pixels.push(data[idx]);
        }
      }
      break;
    }
  }

  // 返回像素数组与切片宽高，调用方据此创建 ImageData
  return { pixels, width, height };
}

/**
 * 取体数据维度；未加载或维度缺失时回退到默认维度
 *
 * 原先 `mriData?.header.dims || [0, 240, 240, 155]` 散落在 6 个文件里，
 * 收口后兜底值只有一处（config/appConfig 的 DEFAULT_VOLUME_DIMS）。
 */
export function getVolumeDims(mriData?: NiftiData | null): number[] {
  const d = mriData?.header?.dims;
  return d && d.length >= 4 ? d : [...DEFAULT_VOLUME_DIMS];
}

/**
 * 三平面最大切片索引（各方向层数减 1，因为索引从 0 起）
 * 轴位沿 Z、冠状位沿 Y、矢状位沿 X，与 getSliceData 的切法一一对应。
 */
export function getMaxSlices(dims: number[]): {
  axial: number;
  coronal: number;
  sagittal: number;
} {
  return {
    axial: (dims[3] || DEFAULT_VOLUME_DIMS[3]) - 1,
    coronal: (dims[2] || DEFAULT_VOLUME_DIMS[2]) - 1,
    sagittal: (dims[1] || DEFAULT_VOLUME_DIMS[1]) - 1,
  };
}
