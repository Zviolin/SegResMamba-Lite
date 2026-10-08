/**
 * ============================================================
 *  useNiftiLoader —— MRI / 分割标签文件加载 Hook
 * ============================================================
 * 职责：封装"用户选择文件 -> 解析 NIfTI -> 写入全局 Zustand store"的完整流程。
 * 对外暴露 loadMriFile（加载原始 MRI）与 loadSegFile（加载分割标签）两个方法，
 * 供 FileUploader 与 LabelFileZone 组件调用。
 *
 * 状态管理：所有解析结果都写入 useViewerStore，因此组件间共享同一份数据；
 * 本 Hook 自身不持有任何 state。
 */
import { useCallback } from 'react';
import { useViewerStore } from '../store/viewerStore';
import { parseNiftiFile, sameShape, getVolumeDims } from '../utils/niftiParser';

/**
 * 加载文件的自定义 Hook
 * @returns { loadMriFile, loadSegFile } 两个异步加载方法
 */
export function useNiftiLoader() {
  // 只订阅 actions（引用稳定），避免推理进度等无关状态变化触发本 Hook 重渲染
  const actions = useViewerStore((s) => s.actions);

  /**
   * 加载原始 MRI 文件
   *
   * @param file 用户选择的 .nii / .nii.gz 文件
   *
   * 流程：
   *  1. 先把文件信息标记为 loading，并打开全局加载遮罩、清空错误；
   *  2. 调用 parseNiftiFile 异步解析 NIfTI；
   *  3. 解析成功后把 NiftiData 写入 store，并把文件状态置为 ready；
   *  4. 把三个平面的初始切片位置设置为体数据各方向的中间层，便于用户起始观察；
   *  5. 出错时记录错误信息并置为 error 状态；最后无论成败都关闭加载遮罩。
   */
  const loadMriFile = useCallback(async (file: File) => {
    try {
      // 标记文件为"加载中"
      actions.setMriFileInfo({
        name: file.name,
        size: file.size,
        status: 'loading'
      });
      actions.setLoading(true);
      actions.setError(null);

      // 解析 NIfTI 文件
      const data = await parseNiftiFile(file);

      // 写入全局 store，供各视图组件读取
      actions.setMriData(data);
      actions.setMriFileInfo({ status: 'ready' });

      // 初始切片位置设为各方向中间层（各平面沿哪个轴切必须与切片索引语义一致）：
      //   - 轴位 axial    沿 Z 轴 → 取 dims[3]（nz）的一半
      //   - 冠状位 coronal 沿 Y 轴 → 取 dims[2]（ny）的一半（此前误用 nz/2，切片不在中间层）
      //   - 矢状位 sagittal 沿 X 轴 → 取 dims[1]（nx）的一半
      // 维度取法与视图侧统一（缺失时的兜底值集中在 getVolumeDims 里）
      const [, nx, ny, nz] = getVolumeDims(data);
      actions.setSlicePosition({
        axial: Math.floor(nz / 2),
        coronal: Math.floor(ny / 2),
        sagittal: Math.floor(nx / 2)
      });

    } catch (error) {
      // 提取可读的错误信息并写入 store
      const message = error instanceof Error ? error.message : 'MRI 文件加载失败';
      actions.setError(message);
      actions.setMriFileInfo({ status: 'error' });
    } finally {
      // 无论成功失败，都关闭全局加载状态
      actions.setLoading(false);
    }
  }, [actions]);

  /**
   * 加载分割标签文件
   *
   * @param file 用户选择的分割标签 .nii / .nii.gz 文件
   *
 * 流程与 loadMriFile 基本一致，区别在于：
 *   - 结果写入 segData / segFileInfo；
 *   - 不重置切片位置（保留用户当前的浏览位置）；
 *   - 多一道形状校验：与已加载的 MRI 尺寸不符时直接拒绝（见函数内说明）。
 */
  const loadSegFile = useCallback(async (file: File) => {
    try {
      // 标记标签文件为"加载中"
      actions.setSegFileInfo({
        name: file.name,
        size: file.size,
        status: 'loading'
      });
      actions.setLoading(true);
      actions.setError(null);

      // 解析 NIfTI 分割文件
      const data = await parseNiftiFile(file);

      // 形状一致性校验：MRI 已加载时必须与之同尺寸。
      // 否则 overlay 会把标签画到错误的解剖位置 —— 不报错，但结果是错的，
      // 医学场景下这种「看起来合理的错误」比直接崩溃危险得多。
      const mri = useViewerStore.getState().mriData;
      if (mri && !sameShape(mri.header.dims, data.header.dims)) {
        throw new Error(
          `标签尺寸 ${data.header.dims.slice(1, 4).join('×')} 与当前 MRI ` +
          `${mri.header.dims.slice(1, 4).join('×')} 不一致，已拒绝加载`
        );
      }

      // 写入全局 store
      actions.setSegData(data);
      actions.setSegFileInfo({ status: 'ready' });

    } catch (error) {
      const message = error instanceof Error ? error.message : '分割文件加载失败';
      actions.setError(message);
      actions.setSegFileInfo({ status: 'error' });
    } finally {
      actions.setLoading(false);
    }
  }, [actions]);

  // 暴露给上层组件的两个加载方法
  return { loadMriFile, loadSegFile };
}
