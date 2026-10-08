/**
 * ============================================================
 *  useViewSync —— 三平面视图同步与切片图像生成 Hook
 * ============================================================
 * 职责：
 *  1. 计算三个平面的最大切片索引（maxSlices），用于滚动范围与进度显示；
 *  2. 提供 getCurrentSliceImage：根据当前三平面切片位置，生成某张切片的
 *     ImageData（含窗宽窗位灰度映射 + 分割标签颜色叠加），供 Canvas 渲染；
 *  3. 透传 updateViewSettings / setSlicePosition，实现窗宽窗位与切片联动。
 *
 * 三平面联动原理：本 Hook 统一从 store 读取同一个 slicePosition，
 * 任何视图调用 setSlicePosition 更新该位置后，所有视图都会随之重新渲染，
 * 从而实现轴向 / 冠状 / 矢状三个视图的十字线同步。
 */
import { useCallback } from 'react';
import { useViewerStore } from '../store/viewerStore';
import { useVolumeDims } from './useVolumeDims';
import { getSliceData, sameShape } from '../utils/niftiParser';
import { renderSlicePixels } from '../utils/blend';
import type { PlaneType } from '../types';

/** 模块级共享离屏 Canvas：仅用于 createImageData，避免每次切片渲染都新建 canvas（性能优化）。 */
let sharedCanvas: HTMLCanvasElement | null = null;
/** 取共享 Canvas 的 2D 上下文；尺寸变化时才重设 width/height（重设即清空，但本处只借它创建 ImageData）。 */
function getSharedCtx(width: number, height: number): CanvasRenderingContext2D {
  if (!sharedCanvas) sharedCanvas = document.createElement('canvas');
  if (sharedCanvas.width !== width) sharedCanvas.width = width;
  if (sharedCanvas.height !== height) sharedCanvas.height = height;
  return sharedCanvas.getContext('2d')!;
}

/**
 * 视图同步 Hook
 * @returns 各平面最大切片索引、切片 ImageData 生成函数、以及透传的 action
 */
export function useViewSync() {
  // 逐个字段订阅：原先直接 `useViewerStore()` 会订阅整个 store，
  // 导致推理进度每跳一次，所有 PlaneViewer / MobileTriView 都被带着重渲染。
  const mriData = useViewerStore((s) => s.mriData);
  const segData = useViewerStore((s) => s.segData);
  const viewSettings = useViewerStore((s) => s.viewSettings);
  const slicePosition = useViewerStore((s) => s.slicePosition);
  // actions 引用稳定，取一次即可
  const { updateViewSettings, setSlicePosition } = useViewerStore((s) => s.actions);

  /**
   * 体数据维度 + 三平面最大切片索引（缓存）
   * 统一由 useVolumeDims 提供，不在本 Hook 里重复实现兜底值与减一计算。
   */
  const { dims, maxSlices } = useVolumeDims();

  /**
   * 生成指定平面的当前切片 ImageData
   *
   * @param plane 平面类型（axial / coronal / sagittal）
   * @returns { imageData, width, height }
   *   - imageData: 可直接 putImageData 到 Canvas 的像素数据；无数据时返回 null
   *   - width / height: 切片像素宽高
   *
   * 渲染逻辑：
   *  1. 根据 store 中的 slicePosition[plane] 取出当前切片索引；
   *  2. 调用 getSliceData 从 MRI 体数据抽取该切片的原始像素；
   *  3. 若存在分割标签，则同平面抽取对应的标签像素；
   *  4. 逐像素处理（实现在 utils/blend.ts，导出网格图共用同一份，保证配色一致）：
   *     a. 先用窗宽窗位算法把原始灰度映射到 0~255（此时 R=G=B=灰度，即灰度图）；
   *     b. 若该像素命中分割标签，则按标签值用 LABEL_COLORS 中的颜色与透明度混合覆盖：
   *        混合方式：新颜色 = 原色*(1-opacity) + 标签色*opacity
   *     c. Alpha 通道固定为 255（不透明）。
   *
   * 注意：依赖数组包含 slicePosition 与 viewSettings，
   * 因此切片位置或窗宽窗位变化时都会重新生成 ImageData。
   */
  const getCurrentSliceImage = useCallback((
    plane: PlaneType
  ): { imageData: ImageData | null, width: number, height: number } => {
    // 未加载 MRI 数据时返回空 ImageData（调用方据此显示空白）
    if (!mriData) return { imageData: null, width: 512, height: 512 };

    // 取出当前平面切片索引，并抽取 MRI 切片像素
    const sliceIdx = slicePosition[plane];
    const { pixels, width, height } = getSliceData(
      mriData.typedArray,
      dims,
      sliceIdx,
      plane
    );

    // 用共享离屏 Canvas 生成 ImageData（RGBA，每像素 4 字节），避免每次调用都 createElement('canvas')
    const ctx = getSharedCtx(width, height);
    const imageData = ctx.createImageData(width, height);

    // 若存在分割标签，则抽取同一位置同一切片的标签像素；否则为 null（不叠加）。
    // 这里再兜一次形状校验：入口处（useNiftiLoader）已拦截不一致文件，
    // 但历史回放等旁路仍可能写入异构数据 —— 宁可不叠加，也不要把标签画错位置。
    const segPixels = (segData && sameShape(segData.header.dims, dims))
      ? getSliceData(
          segData.typedArray,
          segData.header.dims,
          sliceIdx,
          plane
        ).pixels
      : null;

    // 逐像素生成颜色（灰度映射 + 标签叠加；与网格导出共用同一实现，保证配色一致）
    renderSlicePixels(pixels, segPixels, imageData, viewSettings);

    return { imageData, width, height };
  }, [mriData, segData, dims, slicePosition, viewSettings]);

  // 对外暴露：各平面最大索引、切片 ImageData 生成函数、以及透传的 action
  return {
    maxSlices,
    getCurrentSliceImage,
    updateViewSettings,
    setSlicePosition
  };
}
