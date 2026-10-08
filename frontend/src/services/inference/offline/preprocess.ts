/**
 * ============================================================
 *  inference/offline/preprocess —— 离线预处理（纯函数）
 * ============================================================
 * 从 www/index.html 逐行移植（保证数值一致），对应源码位置：
 *  - resizeVolume            ← www: resizeVolume (index.html:2671)
 *  - resampleToSpacing       ← www: resampleToSpacing (2476)
 *  - cropForeground          ← www: cropForeground (2489) / Python CropForegroundd
 *  - padToSize               ← www: padToSize (2524) / Python SpatialPadd
 *  - normalizeVolume         ← www: normalizeVolume (2627) / 训练 normalize_intensity
 *  - prepareInput            ← www: prepareInput (2715)
 *
 * 全部为纯函数：不依赖 React/UI/ORT，可独立单元测试。
 */

/** 体数据容器：data 为 Float32Array，shape 为 [D,H,W]（z,y,x 顺序，与 www 一致） */
export interface VolumeData {
  data: Float32Array;
  shape: [number, number, number];
  spacing: [number, number, number];
}

export const TARGET_SPACING = 2.0;
export const TARGET_SIZE = 64;
export const CHANNEL_COUNT = 4;

/** 通道顺序（与 BraTS 训练一致） */
export const MODALITY_ORDER = ['t1n', 't1c', 't2w', 't2f'] as const;
export type OfflineModality = (typeof MODALITY_ORDER)[number];

/** 将在线版模态标识映射为离线通道（T1ce/FLAIR/T1/T2 → t1c/t2f/t1n/t2w） */
export function mapModalityToChannel(modality: string): OfflineModality {
  const map: Record<string, OfflineModality> = {
    T1: 't1n',
    T1CE: 't1c',
    T2: 't2w',
    FLAIR: 't2f',
  };
  return map[modality.toUpperCase()] ?? 't1c';
}

/** 三线性插值重采样（www resizeVolume 逐行移植） */
export function resizeVolume(
  data: Float32Array,
  oldShape: [number, number, number],
  newShape: [number, number, number]
): Float32Array {
  const [oldD, oldH, oldW] = oldShape;
  const [newD, newH, newW] = newShape;
  const result = new Float32Array(newD * newH * newW);

  for (let d = 0; d < newD; d++) {
    for (let h = 0; h < newH; h++) {
      for (let w = 0; w < newW; w++) {
        const oldDf = (d / (newD - 1)) * (oldD - 1);
        const oldHf = (h / (newH - 1)) * (oldH - 1);
        const oldWf = (w / (newW - 1)) * (oldW - 1);

        const d0 = Math.floor(oldDf), d1 = Math.min(d0 + 1, oldD - 1);
        const h0 = Math.floor(oldHf), h1 = Math.min(h0 + 1, oldH - 1);
        const w0 = Math.floor(oldWf), w1 = Math.min(w0 + 1, oldW - 1);

        const dd = oldDf - d0, dh = oldHf - h0, dw = oldWf - w0;

        const v000 = data[d0 * oldH * oldW + h0 * oldW + w0];
        const v001 = data[d0 * oldH * oldW + h0 * oldW + w1];
        const v010 = data[d0 * oldH * oldW + h1 * oldW + w0];
        const v011 = data[d0 * oldH * oldW + h1 * oldW + w1];
        const v100 = data[d1 * oldH * oldW + h0 * oldW + w0];
        const v101 = data[d1 * oldH * oldW + h0 * oldW + w1];
        const v110 = data[d1 * oldH * oldW + h1 * oldW + w0];
        const v111 = data[d1 * oldH * oldW + h1 * oldW + w1];

        const v00 = v000 * (1 - dw) + v001 * dw;
        const v01 = v010 * (1 - dw) + v011 * dw;
        const v10 = v100 * (1 - dw) + v101 * dw;
        const v11 = v110 * (1 - dw) + v111 * dw;

        const v0 = v00 * (1 - dh) + v01 * dh;
        const v1 = v10 * (1 - dh) + v11 * dh;

        result[d * newH * newW + h * newW + w] = v0 * (1 - dd) + v1 * dd;
      }
    }
  }
  return result;
}

/** 按目标间距重采样（www resampleToSpacing） */
export function resampleToSpacing(
  data: Float32Array,
  oldShape: [number, number, number],
  oldSpacing: [number, number, number],
  targetSpacing: number
): Float32Array {
  const [oldD, oldH, oldW] = oldShape;
  const [sD, sH, sW] = oldSpacing;
  const targetD = Math.max(1, Math.round((oldD * sD) / targetSpacing));
  const targetH = Math.max(1, Math.round((oldH * sH) / targetSpacing));
  const targetW = Math.max(1, Math.round((oldW * sW) / targetSpacing));
  return resizeVolume(data, oldShape, [targetD, targetH, targetW]);
}

/** 非零前景包围盒裁剪（www cropForeground） */
export function cropForeground(data: Float32Array, shape: [number, number, number]) {
  const [D, H, W] = shape;
  let minD = D, maxD = -1, minH = H, maxH = -1, minW = W, maxW = -1;
  for (let d = 0; d < D; d++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        if (data[d * H * W + h * W + w] > 0) {
          if (d < minD) minD = d;
          if (d > maxD) maxD = d;
          if (h < minH) minH = h;
          if (h > maxH) maxH = h;
          if (w < minW) minW = w;
          if (w > maxW) maxW = w;
        }
      }
    }
  }
  if (maxD < 0) {
    return { data: new Float32Array(64 * 64 * 64), shape: [64, 64, 64] as [number, number, number], bbox: [0, 0, 0, 63, 63, 63] as [number, number, number, number, number, number] };
  }
  const newD = maxD - minD + 1;
  const newH = maxH - minH + 1;
  const newW = maxW - minW + 1;
  const result = new Float32Array(newD * newH * newW);
  for (let d = 0; d < newD; d++) {
    for (let h = 0; h < newH; h++) {
      for (let w = 0; w < newW; w++) {
        result[d * newH * newW + h * newW + w] = data[(d + minD) * H * W + (h + minH) * W + (w + minW)];
      }
    }
  }
  return { data: result, shape: [newD, newH, newW] as [number, number, number], bbox: [minD, minH, minW, maxD, maxH, maxW] as [number, number, number, number, number, number] };
}

/** 中心填充/裁剪到目标尺寸（www padToSize） */
export function padToSize(
  data: Float32Array,
  shape: [number, number, number],
  targetSize: [number, number, number]
): Float32Array {
  const [D, H, W] = shape;
  const [tD, tH, tW] = targetSize;
  const result = new Float32Array(tD * tH * tW);
  result.fill(0);
  const startD = Math.max(0, Math.floor((tD - D) / 2));
  const startH = Math.max(0, Math.floor((tH - H) / 2));
  const startW = Math.max(0, Math.floor((tW - W) / 2));
  const srcStartD = Math.max(0, Math.floor((D - tD) / 2));
  const srcStartH = Math.max(0, Math.floor((H - tH) / 2));
  const srcStartW = Math.max(0, Math.floor((W - tW) / 2));
  const copyD = Math.min(D - srcStartD, tD - startD);
  const copyH = Math.min(H - srcStartH, tH - startH);
  const copyW = Math.min(W - srcStartW, tW - startW);
  if (copyD <= 0 || copyH <= 0 || copyW <= 0) return result;
  for (let d = 0; d < copyD; d++) {
    for (let h = 0; h < copyH; h++) {
      for (let w = 0; w < copyW; w++) {
        result[(d + startD) * tH * tW + (h + startH) * tW + (w + startW)] =
          data[(d + srcStartD) * H * W + (h + srcStartH) * W + (w + srcStartW)];
      }
    }
  }
  return result;
}

/** 1-99 百分位归一化，背景零值保持 0（www normalizeVolume / 训练 normalize_intensity） */
export function normalizeVolume(data: Float32Array): Float32Array {
  const n = data.length;
  if (n === 0) return new Float32Array(0);

  const mask = new Uint8Array(n);
  let nonZeroCount = 0;
  for (let i = 0; i < n; i++) {
    if (data[i] > 0) { mask[i] = 1; nonZeroCount++; }
  }
  if (nonZeroCount === 0) return new Float32Array(n);

  const nz = new Float32Array(nonZeroCount);
  let k = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i]) nz[k++] = data[i];
  }
  const arr = Array.from(nz).sort((a, b) => a - b);
  const p01 = arr[Math.max(0, Math.min(nonZeroCount - 1, Math.floor(nonZeroCount * 0.01)))];
  const p99 = arr[Math.max(0, Math.min(nonZeroCount - 1, Math.ceil(nonZeroCount * 0.99) - 1))];

  const range = p99 - p01;
  const result = new Float32Array(n);
  if (range < 1e-8) return result;

  for (let i = 0; i < n; i++) {
    if (!mask[i]) {
      result[i] = 0;
    } else {
      const v = data[i];
      const clipped = v < p01 ? p01 : (v > p99 ? p99 : v);
      result[i] = (clipped - p01) / range;
    }
  }
  return result;
}

/** 预处理变换参数：用于把 64³ 结果逆映射回原始坐标 */
export interface PrepareTransform {
  /** 2.0mm 重采样后的尺寸 [D,H,W]（无数据时为 null） */
  resampledShape: [number, number, number] | null;
  /** CropForeground 的包围盒 [minD,minH,minW,maxD,maxH,maxW]（重采样坐标） */
  cropBBox: [number, number, number, number, number, number] | null;
  /** 裁剪后尺寸 [D,H,W]（重采样坐标） */
  cropShape: [number, number, number] | null;
  /** padToSize 的起始偏移 [startD,startH,startW] */
  padStarts: [number, number, number];
}

/**
 * 组装模型输入张量 [4,64,64,64]（C,D,H,W）
 * @param volumes 各通道体数据（缺失通道可缺省）
 * @returns 输入张量 + 变换参数（供结果逆映射回原始分辨率）
 */
export function prepareInput(
  volumes: Partial<Record<OfflineModality, VolumeData>>
): { tensor: Float32Array; transform: PrepareTransform } {
  const input = new Float32Array(1 * CHANNEL_COUNT * TARGET_SIZE * TARGET_SIZE * TARGET_SIZE);

  // 找到参考模态（按顺序取第一个存在者）
  let refModality: OfflineModality | null = null;
  for (const mod of MODALITY_ORDER) {
    if (volumes[mod]) { refModality = mod; break; }
  }

  let resampledShape: [number, number, number] | null = null;
  let cropBBox: [number, number, number, number, number, number] | null = null;
  let cropShape: [number, number, number] | null = null;
  let firstResampled: Float32Array | null = null;

  if (refModality) {
    const vol = volumes[refModality]!;
    const spacing = vol.spacing;
    resampledShape = [
      Math.max(1, Math.round((vol.shape[0] * spacing[0]) / TARGET_SPACING)),
      Math.max(1, Math.round((vol.shape[1] * spacing[1]) / TARGET_SPACING)),
      Math.max(1, Math.round((vol.shape[2] * spacing[2]) / TARGET_SPACING)),
    ];
    firstResampled = resampleToSpacing(vol.data, vol.shape, spacing, TARGET_SPACING);
    const cropped = cropForeground(firstResampled, resampledShape);
    cropBBox = cropped.bbox;
    cropShape = cropped.shape;
  }

  let anyChannelFilled = false;
  for (let channel = 0; channel < CHANNEL_COUNT; channel++) {
    const modality = MODALITY_ORDER[channel];
    const modalData = volumes[modality];
    if (!modalData) continue;

    let volume: Float32Array;
    let currentShape: [number, number, number];
    if (modalData === volumes[refModality!] && firstResampled) {
      volume = firstResampled;
      currentShape = resampledShape!;
    } else {
      volume = modalData.data;
      currentShape = modalData.shape;
      const spacing = modalData.spacing;
      if (resampledShape) {
        volume = resampleToSpacing(volume, currentShape, spacing, TARGET_SPACING);
        currentShape = resampledShape;
      }
    }

    // 前景裁剪（复用参考模态 bbox）
    if (cropBBox) {
      const [minD, minH, minW, maxD, maxH, maxW] = cropBBox;
      const nD = maxD - minD + 1, nH = maxH - minH + 1, nW = maxW - minW + 1;
      const croppedVol = new Float32Array(nD * nH * nW);
      for (let d = 0; d < nD; d++) {
        for (let h = 0; h < nH; h++) {
          for (let w = 0; w < nW; w++) {
            croppedVol[d * nH * nW + h * nW + w] =
              volume[(d + minD) * currentShape[1] * currentShape[2] + (h + minH) * currentShape[2] + (w + minW)];
          }
        }
      }
      volume = croppedVol;
      currentShape = [nD, nH, nW];
    }

    volume = padToSize(volume, currentShape, [TARGET_SIZE, TARGET_SIZE, TARGET_SIZE]);
    const normalized = normalizeVolume(volume);

    // C,D,H,W layout
    const cOff = channel * TARGET_SIZE * TARGET_SIZE * TARGET_SIZE;
    for (let i = 0; i < TARGET_SIZE * TARGET_SIZE * TARGET_SIZE; i++) {
      input[cOff + i] = normalized[i];
    }
    anyChannelFilled = true;
  }

  if (!anyChannelFilled) {
    console.warn('[offline] prepareInput: 没有任何有效模态数据');
  }

  // padToSize 的起始偏移（各通道一致，基于裁剪后尺寸计算）
  let padStarts: [number, number, number] = [0, 0, 0];
  if (cropShape) {
    const [cd, ch, cw] = cropShape;
    padStarts = [
      Math.max(0, Math.floor((TARGET_SIZE - cd) / 2)),
      Math.max(0, Math.floor((TARGET_SIZE - ch) / 2)),
      Math.max(0, Math.floor((TARGET_SIZE - cw) / 2)),
    ];
  }

  return {
    tensor: input,
    transform: { resampledShape, cropBBox, cropShape, padStarts },
  };
}
