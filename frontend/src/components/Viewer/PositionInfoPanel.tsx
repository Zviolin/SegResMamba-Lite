/**
 * ============================================================
 *  PositionInfoPanel —— 位置信息面板组件
 * ============================================================
 * 职责：在 MPR 右下角面板中显示当前三平面十字线位置对应的三维坐标
 * （X / Y / Z，含各方向最大索引），以及当前所选 MRI 模态。
 *
 * 坐标对应关系（与 MPRViewer 中标注一致）：
 *   X 坐标 = 矢状位切片索引（slicePosition.sagittal）
 *   Y 坐标 = 冠状位切片索引（slicePosition.coronal）
 *   Z 坐标 = 轴位切片索引（slicePosition.axial）
 *
 * 当尚未加载 MRI 数据时，显示"等待加载数据..."占位提示。
 */
import { Box, Typography } from '@mui/material';
import { useViewerStore } from '../../store/viewerStore';
import { useVolumeDims } from '../../hooks/useVolumeDims';

/**
 * 位置信息面板组件
 */
export default function PositionInfoPanel() {
  // 读取三平面切片位置、MRI 数据与推理状态（用于显示模态）
  const slicePosition = useViewerStore((s) => s.slicePosition);
  const mriData = useViewerStore((s) => s.mriData);
  const inference = useViewerStore((s) => s.inference);

  // 读取体数据各方向尺寸；未加载时回退为默认值（统一取自 useVolumeDims）
  const { dims } = useVolumeDims();
  const nx = dims[1] || 0; // X 方向宽度
  const ny = dims[2] || 0; // Y 方向高度
  const nz = dims[3] || 0; // Z 方向层数

  // 未加载数据：显示占位提示
  if (!mriData) {
    return (
      <Box sx={{
        flex: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '10px'
      }}>
        <Typography variant="body2" color="text.secondary" textAlign="center">
          等待加载数据...
        </Typography>
      </Box>
    );
  }

  return (
    <Box sx={{
      flex: 1,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '10px',
      bgcolor: '#000'
    }}>
      <Box sx={{ textAlign: 'center' }}>
        {/* X 坐标：矢状位切片索引 / X 方向最大索引（nx-1） */}
        <Typography
          variant="h6"
          sx={{
            color: '#00adb5',
            fontWeight: 600,
            fontFamily: 'monospace'
          }}
        >
          X: {slicePosition.sagittal} / {nx - 1}
        </Typography>
        {/* Y 坐标：冠状位切片索引 / Y 方向最大索引（ny-1） */}
        <Typography
          variant="h6"
          sx={{
            color: '#00adb5',
            fontWeight: 600,
            fontFamily: 'monospace'
          }}
        >
          Y: {slicePosition.coronal} / {ny - 1}
        </Typography>
        {/* Z 坐标：轴位切片索引 / Z 方向最大索引（nz-1） */}
        <Typography
          variant="h6"
          sx={{
            color: '#00adb5',
            fontWeight: 600,
            fontFamily: 'monospace'
          }}
        >
          Z: {slicePosition.axial} / {nz - 1}
        </Typography>

        {/* 当前所选模态 */}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ mt: 2, display: 'block' }}
        >
          模态: {inference.selectedModality}
        </Typography>
        {/* 操作提示 */}
        <Typography
          variant="caption"
          color="text.secondary"
        >
          点击视图定位
        </Typography>
      </Box>
    </Box>
  );
}
