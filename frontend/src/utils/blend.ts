/**
 * ============================================================
 *  utils/blend —— 切片像素 → RGBA 的唯一渲染实现
 * ============================================================
 * 背景：原先「三平面视图（useViewSync）」与「网格导出（GridExport）」
 * 各自写了一份逐像素混色循环，且 GridExport 那份把标签颜色硬编码成了
 * 与 config/labels.ts 完全不同的值 —— 导致导出的 PNG/PDF 与屏幕上看到的
 * 分割颜色不一致。此处收敛为唯一实现，两边共用。
 *
 * 颜色来源：全部取自 config/labels.ts 的 LABEL_COLORS（唯一取色来源）。
 */
import { applyWindowLevel } from './niftiParser';
import { LABEL_COLORS } from '../config/labels';
import type { ViewSettings } from '../types';

/** 可索引的数值序列（getSliceData 返回的 number[] 或 TypedArray 均可） */
type PixelSource = ArrayLike<number>;

/**
 * 把一层切片像素写入 ImageData（原地修改，不分配新对象）
 *
 * @param pixels       灰度源像素（MRI 体素值）
 * @param segPixels    分割标签像素；传 null 则不叠加
 * @param imageData    目标像素缓冲（长度需 ≥ pixels.length * 4）
 * @param viewSettings 窗宽窗位 / 透明度 / 各区域显隐
 *
 * 逐像素处理：
 *  1. applyWindowLevel 把原始体素映射到 0~255 灰度（R=G=B）；
 *  2. 命中标签时按 LABEL_COLORS 的颜色与透明度混合：
 *     新色 = 原灰度 × (1-opacity) + 标签色 × opacity；
 *  3. Alpha 固定 255（不透明）。
 */
export function renderSlicePixels(
  pixels: PixelSource,
  segPixels: PixelSource | null,
  imageData: ImageData,
  viewSettings: ViewSettings
): void {
  const data = imageData.data;
  // 叠加混合权重提前算好，循环内不再做除法
  const opacity = viewSettings.opacity / 100;
  const invOpacity = 1 - opacity;
  const { windowWidth, windowLevel } = viewSettings;

  for (let i = 0; i < pixels.length; i++) {
    const idx = i * 4; // RGBA 每像素 4 字节
    let r: number, g: number, b: number;

    // ① 窗宽窗位映射为灰度
    const gray = applyWindowLevel(pixels[i], windowWidth, windowLevel);
    r = g = b = gray;

    // ② 分割标签叠加（仅当该平面存在标签像素时执行）
    if (segPixels) {
      const segVal = Math.round(segPixels[i]);

      if (segVal === LABEL_COLORS.NCR.value && viewSettings.showNecrosis) {
        // 标签 1：坏死核心（NCR）
        const c = LABEL_COLORS.NCR.rgb;
        r = r * invOpacity + c[0] * opacity;
        g = g * invOpacity + c[1] * opacity;
        b = b * invOpacity + c[2] * opacity;
      } else if (segVal === LABEL_COLORS.ED.value && viewSettings.showEdema) {
        // 标签 2：水肿区（ED）
        const c = LABEL_COLORS.ED.rgb;
        r = r * invOpacity + c[0] * opacity;
        g = g * invOpacity + c[1] * opacity;
        b = b * invOpacity + c[2] * opacity;
      } else if (
        (LABEL_COLORS.ET.values as readonly number[]).includes(segVal) &&
        viewSettings.showEnhancing
      ) {
        // 标签 3 或 4：增强肿瘤（ET）
        const c = LABEL_COLORS.ET.rgb;
        r = r * invOpacity + c[0] * opacity;
        g = g * invOpacity + c[1] * opacity;
        b = b * invOpacity + c[2] * opacity;
      }
    }

    // ③ 写入 RGBA；data 是 Uint8ClampedArray，赋值自动钳位 0~255
    data[idx] = r;
    data[idx + 1] = g;
    data[idx + 2] = b;
    data[idx + 3] = 255;
  }
}
