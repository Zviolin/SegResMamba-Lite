/**
 * ============================================================
 *  utils/niftiWriter —— NIfTI 文件导出（segData → .nii.gz）
 * ============================================================
 * 用途：把内存中的分割掩码（segData）打包成标准 NIfTI-1 文件并 gzip 压缩，
 * 供用户下载分享。对方拿到文件后，在本 App（或任何 NIfTI 工具）中
 * "加载标签文件"即可看到完全相同的分割结果。
 *
 * 组成：
 *  - buildNiftiHeader  : NIfTI-1 头（348 字节，纯函数，可单测）
 *  - gzipBytes         : gzip 压缩（用浏览器内置 CompressionStream，无需依赖）
 *  - downloadBlob      : 触发浏览器下载（DOM）
 *  - exportSegDataToNiftiGz : 组合入口（segData → 下载 .nii.gz）
 */
import type { NiftiData } from '../types';

/** NIfTI-1 头大小（字节） */
const HEADER_SIZE = 348;
/** vox_offset：header + 4 字节对齐 padding */
const VOX_OFFSET = 352;

/** 写 int16 到 DataView */
function writeInt16(view: DataView, offset: number, value: number): void {
  view.setInt16(offset, value, true);
}

/** 写 float32 到 DataView */
function writeFloat32(view: DataView, offset: number, value: number): void {
  view.setFloat32(offset, value, true);
}

/**
 * 构建 NIfTI-1 二进制头（348 字节，little-endian，mri 解析兼容）
 *
 * @param dims   [nx, ny, nz]（体素尺寸）
 * @param pixDims [sx, sy, sz]（体素间距，mm）
 * @param datatype 数据类型编码（2=uint8）
 * @param bitpix   位深（8）
 */
export function buildNiftiHeader(
  dims: [number, number, number],
  pixDims: [number, number, number],
  datatype = 2,
  bitpix = 8
): ArrayBuffer {
  const buffer = new ArrayBuffer(HEADER_SIZE);
  const view = new DataView(buffer);

  // sizeof_hdr = 348
  view.setInt32(0, HEADER_SIZE, true);
  // dim[0] = 3 维，dim[1..3] = 尺寸
  writeInt16(view, 40, 3);
  writeInt16(view, 42, dims[0]);
  writeInt16(view, 44, dims[1]);
  writeInt16(view, 46, dims[2]);
  // datatype / bitpix
  writeInt16(view, 70, datatype);
  writeInt16(view, 72, bitpix);
  // pixdim[0] = 1（无缩放），pixdim[1..3] = 间距
  writeFloat32(view, 76, 1);
  writeFloat32(view, 80, pixDims[0]);
  writeFloat32(view, 84, pixDims[1]);
  writeFloat32(view, 88, pixDims[2]);
  // vox_offset = 352
  writeFloat32(view, 108, VOX_OFFSET);
  // xyzt_units = 2（mm）+ 8（秒）= 10
  view.setUint8(123, 10);
  // magic = "n+1\0"
  view.setUint8(344, 0x6e); // n
  view.setUint8(345, 0x2b); // +
  view.setUint8(346, 0x31); // 1
  view.setUint8(347, 0x00); // \0

  return buffer;
}

/**
 * gzip 压缩字节流（浏览器内置 CompressionStream，无需 pako 依赖）
 * Node 18+ / 现代浏览器 / 安卓 WebView 均支持。
 */
export async function gzipBytes(data: Uint8Array | ArrayBuffer | ArrayBufferView): Promise<Uint8Array> {
  if (typeof CompressionStream === 'undefined') {
    throw new Error('当前环境不支持 CompressionStream，无法导出 gzip 文件');
  }
  // Blob 构造支持 ArrayBuffer / ArrayBufferView / Uint8Array，
  // 这里统一打包成 Blob 再走流式压缩，避免对调用方做强类型限制。
  const stream = new Blob([data as BlobPart]).stream().pipeThrough(new CompressionStream('gzip'));
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

/** 触发浏览器下载（DOM 操作，仅浏览器环境调用） */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * 构建完整的 .nii.gz 压缩包（Blob）：header + VOX_OFFSET 对齐 + 掩码数据 → gzip
 * 供：①导出下载 ②存入 IndexedDB 历史记录（稍后可再次导出）
 */
export async function buildNiftiGzBlob(
  segData: NiftiData
): Promise<Blob> {
  const dims = segData.header.dims;
  const nx = dims[1] || 0;
  const ny = dims[2] || 0;
  const nz = dims[3] || 0;
  if (!nx || !ny || !nz || !segData.typedArray) {
    throw new Error('分割数据不完整，无法导出');
  }
  const pixDims = segData.header.pixDims;
  const header = buildNiftiHeader(
    [nx, ny, nz],
    [pixDims[1] || 1, pixDims[2] || 1, pixDims[3] || 1]
  );

  // 数据 layout：segData.typedArray 为 x + y*nx + z*nx*ny（与 niftiParser 解析结果一致，可直接拼接）
  const full = new Uint8Array(VOX_OFFSET + segData.typedArray.length);
  full.set(new Uint8Array(header), 0);
  full.set(segData.typedArray, VOX_OFFSET);

  const gz = await gzipBytes(full);
  return new Blob([gz], { type: 'application/gzip' });
}

/**
 * 组合入口：把 segData（分割掩码）导出为 .nii.gz 并触发下载
 * @param segData 分割结果（含 typedArray + header）
 * @param baseName 文件名前缀（默认 segmentation_result）
 */
export async function exportSegDataToNiftiGz(
  segData: NiftiData,
  baseName = 'segmentation_result'
): Promise<void> {
  const blob = await buildNiftiGzBlob(segData);
  downloadBlob(blob, `${baseName}.nii.gz`);
}

/**
 * 从历史记录压缩包直接导出下载（无需解压重建，直接转发 Blob）
 * @param gzBlob IndexedDB 中保存的 .nii.gz 压缩包
 * @param baseName 文件名前缀
 */
export async function exportNiftiGzBlob(gzBlob: Blob, baseName: string): Promise<void> {
  downloadBlob(gzBlob, `${baseName}.nii.gz`);
}

/**
 * ============================================================
 *  exportFullReportTs —— 「完整报告 .ts」导出
 * ============================================================
 * 用途：把一次推理的「完整上下文 + 体数据」打包到一个 TypeScript 模块文件
 * 里（方案 A：text + base64）。该文件：
 *  - 文本部分可直接阅读/编辑（metadata、统计、视图参数）
 *  - base64 部分解码后即 3D 体数据（uint8 label；可选 float32 MRI）
 *  - 后续任何前端项目 `import` 或 `fetch` 即可还原全部上下文
 *
 * 与 .nii.gz 的关系：保留 .nii.gz 给标准 NIfTI 工具用；.ts 是「全包」版本，
 * 含分析报告 + 视图设置 + 模型信息 + 体数据，便于「一键回放 / 离线归档」。
 *
 * 文件结构（伪代码）：
 *   // header 注释
 *   export const metadata = { id, createdAt, model, modality, inference, window, visibility, statistics };
 *   export const segShape = [nx, ny, nz];
 *   export const segBase64 = "...";
 *   export const mriShape?: [nx, ny, nz];
 *   export const mriBase64?: "...";
 */
export interface FullReportOptions {
  /** 是否把 MRI 原始灰度体数据也打包进去（默认 false，避免 .ts 过大） */
  includeMri?: boolean;
  /** 基础文件名（默认 segmentation_report） */
  baseName?: string;
}

export interface FullReportInputs {
  segData: NiftiData;
  mriData?: NiftiData | null;
  /** 任意要写入 metadata 的 KV 字段（已合并默认值后再传入） */
  metadata: Record<string, unknown>;
}

/** Uint8Array → base64 字符串（浏览器原生 btoa，分片避免栈溢出） */
function uint8ToBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(
      null,
      Array.from(bytes.subarray(i, i + chunk))
    );
  }
  return btoa(binary);
}

/** 把任意数组格式化为可读的 metadata 文本字段名 */
function describeNifti(d: NiftiData | undefined | null): string {
  if (!d) return 'null';
  const dims = d.header?.dims || [];
  return `[${dims[1] || 0}, ${dims[2] || 0}, ${dims[3] || 0}]`;
}

/**
 * 导出「完整报告 .ts」：组装 metadata + gzip 后的 base64 体数据，写成 .ts 文本并下载。
 */
export async function exportFullReportTs(
  inputs: FullReportInputs,
  options: FullReportOptions = {}
): Promise<void> {
  const { segData, mriData, metadata } = inputs;
  const { includeMri = false, baseName = 'segmentation_report' } = options;

  if (!segData?.typedArray || !segData.header?.dims) {
    throw new Error('缺少分割数据，无法导出完整报告');
  }

  // ---- 1) 分割体数据：直接 gz 后转 base64 ----
  // segData.typedArray 可能是 Uint8Array 也可能是其它数值视图，
  // 这里按其底层字节重新打包成 Uint8Array，确保下游 gzip / base64 收到正确类型。
  const segBytes = new Uint8Array(
    segData.typedArray.buffer,
    segData.typedArray.byteOffset,
    segData.typedArray.byteLength
  );
  const segGz = await gzipBytes(segBytes);
  const segB64 = uint8ToBase64(segGz);
  const [nx, ny, nz] = [segData.header.dims[1] || 0, segData.header.dims[2] || 0, segData.header.dims[3] || 0];

  // ---- 2) 可选 MRI：float32 typedArray → gz → base64 ----
  let mriSection = '';
  if (includeMri && mriData?.typedArray && mriData.header?.dims) {
    // typedArray 已是 float32 等同语义，按原始字节编码
    const mriBytes = new Uint8Array(
      mriData.typedArray.buffer,
      mriData.typedArray.byteOffset,
      mriData.typedArray.byteLength
    );
    const mriGz = await gzipBytes(mriBytes);
    const mriB64 = uint8ToBase64(mriGz);
    const mriShape = [mriData.header.dims[1] || 0, mriData.header.dims[2] || 0, mriData.header.dims[3] || 0];
    mriSection =
      `\n// 可选：原始 MRI 灰度（float32）体数据\n` +
      `export const mriShape = [${mriShape[0]}, ${mriShape[1]}, ${mriShape[2]}];\n` +
      `export const mriDtype = 'float32';\n` +
      `export const mriBase64 = "${mriB64}";\n`;
  }

  // ---- 3) metadata JSON（紧凑） ----
  const metaJson = JSON.stringify(metadata, null, 2);
  const createdAt = new Date().toISOString();

  // ---- 4) 拼装 .ts 文本 ----
  const tsContent =
`/* eslint-disable */
/**
 * ${baseName}.ts —— 脑肿瘤分割完整报告
 * 由 脑肿瘤可视化系统 于 ${createdAt} 生成
 * 体数据维度：${describeNifti(segData)}${includeMri && mriData ? `；MRI 维度：${describeNifti(mriData)}` : ''}
 *
 * 用法：
 *   import report from './${baseName}';
 *   const segData = decodeFullReport(report).seg;   // Uint8Array（gz 解压后）
 *   const stats   = report.metadata.statistics;     // 直接可读
 */
export const metadata = ${metaJson};
export const segShape = [${nx}, ${ny}, ${nz}];
export const segDtype = 'uint8';
export const segBase64 = "${segB64}";${mriSection}
export default { metadata, segShape, segDtype, segBase64${includeMri ? ', mriShape, mriDtype, mriBase64' : ''} };
`;

  downloadBlob(new Blob([tsContent], { type: 'text/typescript;charset=utf-8' }), `${baseName}.ts`);
}
