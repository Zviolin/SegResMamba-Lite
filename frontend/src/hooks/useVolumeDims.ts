/**
 * ============================================================
 *  useVolumeDims —— 体数据维度与三平面最大切片索引的唯一取法
 * ============================================================
 * 背景：原先「维度兜底值」和「各平面最大切片索引」在各视图组件里各算一遍：
 *   - `mriData?.header.dims || [0, 240, 240, 155]` 出现在 6 个文件；
 *   - `axial = dims[3]-1 / coronal = dims[2]-1 / sagittal = dims[1]-1` 出现在 4 个文件。
 * 同一个概念散落多处，改一处漏一处就会出现「滑块上限和视图层数对不上」这类问题。
 *
 * 这里统一收敛：只订阅 mriData，其余交给 niftiParser 的纯函数。
 */
import { useMemo } from 'react';
import { useViewerStore } from '../store/viewerStore';
import { getVolumeDims, getMaxSlices } from '../utils/niftiParser';

/**
 * 当前体数据的维度与各平面最大切片索引
 *
 * @returns dims      维度数组 [维度数, nx, ny, nz]（未加载时为兜底值）
 * @returns maxSlices 三平面各自的最大切片索引（层数 - 1）
 */
export function useVolumeDims() {
  const mriData = useViewerStore((s) => s.mriData);

  // 仅当 mriData 变化时重算，避免每次渲染都产生新数组导致下游 useMemo 失效
  const dims = useMemo(() => getVolumeDims(mriData), [mriData]);
  const maxSlices = useMemo(() => getMaxSlices(dims), [dims]);

  return { dims, maxSlices };
}
