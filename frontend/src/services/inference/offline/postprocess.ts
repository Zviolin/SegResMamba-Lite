/**
 * ============================================================
 *  inference/offline/postprocess —— 离线后处理（纯函数）
 * ============================================================
 * 从 www/index.html 逐行移植，对应源码位置：
 *  - softmax4                ← www: softmax4 (2808)
 *  - processMultiClassOutput ← www: processMultiClassOutput (2821)
 *  - countClasses            ← 派生统计（体积计算前置）
 *
 * 全部为纯函数，可独立单元测试。
 */
import { TARGET_SIZE, type PrepareTransform } from './preprocess';

/** 4 类 softmax（跨通道计算，www softmax4 逐行移植） */
export function softmax4(input: Float32Array, offset: number, classSize: number): number[] {
  const maxVal = Math.max(
    input[offset], input[offset + classSize],
    input[offset + 2 * classSize], input[offset + 3 * classSize]
  );
  const e0 = Math.exp(input[offset] - maxVal);
  const e1 = Math.exp(input[offset + classSize] - maxVal);
  const e2 = Math.exp(input[offset + 2 * classSize] - maxVal);
  const e3 = Math.exp(input[offset + 3 * classSize] - maxVal);
  const sum = e0 + e1 + e2 + e3;
  return [e0 / sum, e1 / sum, e2 / sum, e3 / sum];
}

/**
 * 多分类输出后处理：softmax → argmax → labels（Uint8Array 64³）
 * 标签：0=背景 / 1=坏死 / 2=非增强 / 3=增强（www processMultiClassOutput）
 */
export function processMultiClassOutput(output: Float32Array): Uint8Array {
  const classSize = TARGET_SIZE * TARGET_SIZE * TARGET_SIZE;
  const result = new Uint8Array(classSize);
  for (let v = 0; v < classSize; v++) {
    const p = softmax4(output, v, classSize);
    let maxIdx = 0;
    if (p[1] > p[maxIdx]) maxIdx = 1;
    if (p[2] > p[maxIdx]) maxIdx = 2;
    if (p[3] > p[maxIdx]) maxIdx = 3;
    result[v] = maxIdx;
  }
  return result;
}

/** 统计 64³ 掩码各类别体素数 */
export function countClasses(labels: Uint8Array): [number, number, number, number] {
  const counts: [number, number, number, number] = [0, 0, 0, 0];
  for (let i = 0; i < labels.length; i++) counts[labels[i]]++;
  return counts;
}

/**
 * 将 64³ 掩码（[D,H,W] layout）逆变换映射回原始分辨率。
 *
 * 关键：64³ 与原始 MRI 不是简单缩放，中间经历 ①2mm 重采样 ②CropForeground ③pad 到 64³，
 * 必须按变换参数逆映射（逆重采样→逆裁剪→逆填充），否则位置错乱。
 * （历史 bug：曾用纯比例缩放导致掩码散落在脑外）
 */
export function upsampleMaskToOriginal(
  labels: Uint8Array,
  nx: number,
  ny: number,
  nz: number,
  transform: PrepareTransform
): Uint8Array {
  const seg = new Uint8Array(nx * ny * nz);
  const s = TARGET_SIZE;
  const { resampledShape, cropBBox, cropShape, padStarts } = transform;

  // 无变换参数（异常路径）时回退为纯比例缩放，保证不崩溃
  if (!resampledShape || !cropBBox || !cropShape) {
    for (let z = 0; z < nz; z++) {
      const d = Math.min(s - 1, Math.floor((z * s) / nz));
      for (let y = 0; y < ny; y++) {
        const h = Math.min(s - 1, Math.floor((y * s) / ny));
        for (let x = 0; x < nx; x++) {
          const w = Math.min(s - 1, Math.floor((x * s) / nx));
          seg[x + y * nx + z * nx * ny] = labels[d * s * s + h * s + w];
        }
      }
    }
    return seg;
  }

  const [resD, resH, resW] = resampledShape;
  const [minD, minH, minW] = cropBBox;
  const [cropD, cropH, cropW] = cropShape;
  const [startD, startH, startW] = padStarts;

  for (let z = 0; z < nz; z++) {
    // ① 逆重采样：原始 z → 2mm 坐标（与 resizeVolume 映射方向一致）
    const resZ = Math.round((z * (resD - 1)) / Math.max(1, nz - 1));
    // ② 逆裁剪：减去 bbox 起点
    const cD = resZ - minD;
    if (cD < 0 || cD >= cropD) continue;
    for (let y = 0; y < ny; y++) {
      const resY = Math.round((y * (resH - 1)) / Math.max(1, ny - 1));
      const cH = resY - minH;
      if (cH < 0 || cH >= cropH) continue;
      for (let x = 0; x < nx; x++) {
        const resX = Math.round((x * (resW - 1)) / Math.max(1, nx - 1));
        const cW = resX - minW;
        if (cW < 0 || cW >= cropW) continue;
        // ③ 逆填充：加上 pad 偏移，得到 64³ 索引
        const d = cD + startD;
        const h = cH + startH;
        const w = cW + startW;
        if (d < 0 || d >= s || h < 0 || h >= s || w < 0 || w >= s) continue;
        seg[x + y * nx + z * nx * ny] = labels[d * s * s + h * s + w];
      }
    }
  }
  return seg;
}
