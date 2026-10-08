/**
 * ============================================================
 *  MPRViewer —— MPR（多平面重建）主视图组件
 * ============================================================
 * 职责：
 *  - 桌面（≥900px）：CSS Grid 2×2 四宫格
 *      左上=轴位(Z) / 右上=冠状(Y) / 左下=矢状(X) / 右下=位置信息；
 *      三平面共享同一 slicePosition，滚动任一平面同步更新其他十字线。
 *  - 移动（<900px）：仿参考版"1 大 + 2 小"三平面布局（MobileTriView），
 *      点击小图可切换主视图平面，亮度/对比度等调节直接在页面卡片上操作。
 *
 * 每个平面面板包含：顶部标题栏 + 中间 Canvas（PlaneViewer） + 底部切片索引。
 */
import { Box, Typography } from '@mui/material';
import PlaneViewer from './PlaneViewer';
import PositionInfoPanel from './PositionInfoPanel';
import GridExport from './GridExport';
import MobileTriView from './MobileTriView';
import { useIsMobile } from '../../hooks/useIsMobile';
import { useViewerStore } from '../../store/viewerStore';
import type { PlaneType } from '../../types';

/** 平面显示名（桌面标题用） */
const PLANE_META: Record<PlaneType, { title: string; axis: 'X' | 'Y' | 'Z' }> = {
  axial: { title: '轴位 Axial', axis: 'Z' },
  coronal: { title: '冠状位 Coronal', axis: 'Y' },
  sagittal: { title: '矢状位 Sagittal', axis: 'X' },
};

/**
 * 单个平面面板（桌面四宫格使用）
 */
function PlanePanel({ plane }: { plane: PlaneType }) {
  const slicePosition = useViewerStore((s) => s.slicePosition);
  const { title, axis } = PLANE_META[plane];
  return (
    <Box sx={{
      bgcolor: '#1e1e1e',
      borderRadius: 1,
      border: '1px solid #333',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      minHeight: 120,
    }}>
      <Typography
        variant="subtitle2"
        sx={{
          color: '#00adb5',
          fontWeight: 600,
          textAlign: 'center',
          padding: '8px 12px',
          bgcolor: 'rgba(15, 23, 42, 0.8)',
          borderBottom: '1px solid #333',
        }}
      >
        {title} ({axis})
      </Typography>
      <Box sx={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: '#000', overflow: 'hidden', minHeight: 0 }}>
        <PlaneViewer plane={plane} title="" />
      </Box>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{
          textAlign: 'center',
          padding: '5px 10px',
          bgcolor: 'rgba(15, 23, 42, 0.8)',
          borderTop: '1px solid #333',
          fontFamily: 'monospace',
        }}
      >
        {axis} = {slicePosition[plane]}
      </Typography>
    </Box>
  );
}

/**
 * MPR 主视图组件（响应式：桌面 2×2 四宫格 / 移动一大两小三视图）
 */
export default function MPRViewer() {
  const isMobile = useIsMobile();

  return (
    <Box sx={{
      width: '100%',
      height: isMobile ? 'auto' : 'calc(100vh - 96px)',
      overflowY: isMobile ? 'visible' : 'auto',
      display: 'flex',
      flexDirection: 'column',
      minHeight: 0,
    }}>
      {isMobile ? (
        /* ───────── 移动端：1 大 + 2 小三视图 ───────── */
        <MobileTriView />
      ) : (
        /* ───────── 桌面端：2×2 四宫格 ───────── */
        <>
          <Box sx={{
            display: 'grid',
            gridTemplateColumns: 'repeat(2, 1fr)',
            gridTemplateRows: 'repeat(2, 1fr)',
            flex: 1,
            minHeight: 240,
            gap: 10,
            width: '100%',
          }}>
            <PlanePanel plane="axial" />
            <PlanePanel plane="coronal" />
            <PlanePanel plane="sagittal" />
            {/* 右下：位置信息面板 */}
            <Box sx={{
              bgcolor: '#1e1e1e', borderRadius: 1, border: '1px solid #333',
              display: 'flex', flexDirection: 'column', overflow: 'hidden', minHeight: 120,
            }}>
              <Typography
                variant="subtitle2"
                sx={{
                  color: '#00adb5', fontWeight: 600, textAlign: 'center',
                  padding: '8px 12px', bgcolor: 'rgba(15, 23, 42, 0.8)',
                  borderBottom: '1px solid #333',
                }}
              >
                位置信息
              </Typography>
              <PositionInfoPanel />
            </Box>
          </Box>
          <GridExport />
        </>
      )}
    </Box>
  );
}

