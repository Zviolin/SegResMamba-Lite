/**
 * ============================================================
 *  MobileTriView —— 手机端"一大两小"三平面视图
 * ============================================================
 * 模仿参考版（brainseg-app 结果页）的手机排版：
 *  - 上方 1 个大视图（当前激活平面，为主视图）；
 *  - 下方并排 2 个小视图（另外两个平面），点击即可切换为上方大视图；
 *  - 大视图支持 ◀ / ▶ 按钮、鼠标滚轮、手指上下/左右滑动来翻切片；
 *  - 三个平面的切片索引互相独立（与桌面 MPR 一致，共用同一 store）。
 *
 * 渲染逻辑与桌面 PlaneViewer 相同：useViewSync.getCurrentSliceImage
 * 生成 ImageData 后 putImageData 到 canvas，切片/窗宽窗位变化自动重绘。
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Box, Typography, IconButton, Slider } from '@mui/material';
import { ChevronLeft as ChevronLeftIcon, ChevronRight as ChevronRightIcon } from '@mui/icons-material';
import type { PlaneType } from '../../types';
import { useViewSync } from '../../hooks/useViewSync';
import { useVolumeDims } from '../../hooks/useVolumeDims';
import { useViewerStore } from '../../store/viewerStore';


/** 三个平面（用于推导"当前激活平面之外的另外两个"） */
const PLANES: PlaneType[] = ['axial', 'coronal', 'sagittal'];

/** 平面显示名（角标/小图标签共用） */
const PLANE_META: Record<PlaneType, { title: string; short: string }> = {
  axial: { title: '横断面 Axial', short: '横断面' },
  coronal: { title: '冠状位 Coronal', short: '冠状位' },
  sagittal: { title: '矢状位 Sagittal', short: '矢状位' },
};

/**
 * 单个平面 Canvas：画布自动适配容器（保持比例、居中显示）
 */
function PlaneCanvas({ plane }: { plane: PlaneType }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // 渲染计数器：防止过期渲染覆盖最新切片
  const renderRef = useRef(0);
  const { getCurrentSliceImage } = useViewSync();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const currentRenderId = ++renderRef.current;
    const { imageData, width, height } = getCurrentSliceImage(plane);
    if (imageData && currentRenderId === renderRef.current) {
      canvas.width = width;
      canvas.height = height;
      ctx.putImageData(imageData, 0, 0);
    }
  }, [getCurrentSliceImage, plane]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        display: 'block',
        maxWidth: '100%',
        maxHeight: '100%',
        width: 'auto',
        height: 'auto',
        imageRendering: 'pixelated'
      }}
    />
  );
}

/**
 * 手机端一大两小三视图组件
 */
export default function MobileTriView() {
  // 当前激活（大图展示）的平面
  const [active, setActive] = useState<PlaneType>('axial');
  // MRI 数据与当前三平面切片位置（订阅变化驱动 canvas 重绘）
  const mriData = useViewerStore((s) => s.mriData);
  const slicePosition = useViewerStore((s) => s.slicePosition);
  const { setSlicePosition } = useViewerStore((s) => s.actions);

  // 三平面最大切片索引统一取自 useVolumeDims（轴向沿 Z、冠状沿 Y、矢状沿 X）
  const { maxSlices } = useVolumeDims();
  const maxSlice = maxSlices[active];

  // 激活平面当前切片（0 起），展示时 +1
  const cur = slicePosition[active];
  const max = maxSlice;

  /** 步进切片：dir = 1 下一层 / -1 上一层（越界自动钳制） */
  const stepSlice = useCallback((dir: 1 | -1) => {
    const next = Math.max(0, Math.min(max, cur + dir));
    setSlicePosition({ [active]: next });
  }, [active, cur, max, setSlicePosition]);

  // 大视图容器 ref：挂滚轮翻片
  const bigWrapRef = useRef<HTMLDivElement | null>(null);
  // 触摸起点（用于滑动翻片）
  const touchRef = useRef<{ x: number; y: number } | null>(null);

  // 滚轮翻片（仅在数据就绪时生效）
  useEffect(() => {
    const el = bigWrapRef.current;
    if (!el || !mriData) return;
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      stepSlice(e.deltaY < 0 ? 1 : -1);
    };
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [mriData, stepSlice]);

  /** 未加载数据：显示黑色占位（避免空 canvas 全黑无提示） */
  if (!mriData) {
    return (
      <Box
        sx={{
          width: '100%',
          height: 220,
          borderRadius: '12px',
          bgcolor: '#000',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          px: 3,
        }}
      >
        <Typography variant="body2" color="text.secondary" textAlign="center">
          未加载 MRI 数据
          <br />
          请通过「上传 / 推理」加载文件
        </Typography>
      </Box>
    );
  }

  // 非激活的两个平面（保持轴向/冠状/矢状的展示顺序）
  const others = PLANES.filter((p) => p !== active);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      {/* ===== 大视图（激活平面，可滚轮/滑动/按钮翻片） ===== */}
      <Box
        ref={bigWrapRef}
        onTouchStart={(e) => { touchRef.current = { x: e.touches[0].clientX, y: e.touches[0].clientY }; }}
        onTouchEnd={(e) => {
          const start = touchRef.current;
          touchRef.current = null;
          if (!start) return;
          const dx = e.changedTouches[0].clientX - start.x;
          const dy = e.changedTouches[0].clientY - start.y;
          // 取主导方向：水平滑动或垂直滑动都翻片
          if (Math.abs(dx) < 24 && Math.abs(dy) < 24) return;
          const dir: 1 | -1 = Math.abs(dx) > Math.abs(dy)
            ? (dx > 0 ? 1 : -1)
            : (dy < 0 ? 1 : -1);
          stepSlice(dir);
        }}
        sx={{
          position: 'relative',
          width: '100%',
          // 压缩大视图高度，给下方量化分析留空间
          height: 'min(28vh, 260px)',
          minHeight: 160,
          bgcolor: '#000',
          borderRadius: '12px',
          border: '1px solid var(--border-strong)',
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          // 手指在图像上滑动用于翻片，不触发页面滚动
          touchAction: 'none',
          userSelect: 'none',
          WebkitUserSelect: 'none',
          cursor: 'pointer',
        }}
      >
        <PlaneCanvas plane={active} />
        {/* 平面角标（左上） */}
        <Box
          sx={{
            position: 'absolute',
            top: 8,
            left: 8,
            px: 1,
            py: 0.25,
            borderRadius: '999px',
            bgcolor: 'rgba(15, 23, 42, 0.65)',
            color: 'var(--accent)',
            fontSize: '0.7rem',
            fontWeight: 600,
            pointerEvents: 'none',
          }}
        >
          {PLANE_META[active].title}
        </Box>
        {/* 交互提示（右下） */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 8,
            right: 8,
            px: 1,
            py: 0.25,
            borderRadius: '999px',
            bgcolor: 'rgba(15, 23, 42, 0.5)',
            color: 'rgba(255,255,255,0.72)',
            fontSize: '0.62rem',
            pointerEvents: 'none',
          }}
        >
          滑动 / 滚轮翻片
        </Box>
      </Box>

      {/* ===== 两个小视图（点击切换为大图） ===== */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        {others.map((p) => (
          <Box
            key={p}
            onClick={() => setActive(p)}
            sx={{
              flex: 1,
              minWidth: 0,
              position: 'relative',
              height: 64,
              bgcolor: '#000',
              borderRadius: '10px',
              border: '1px solid var(--border)',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
              '&:active': { opacity: 0.75 },
            }}
          >
            <PlaneCanvas plane={p} />
            <Box
              sx={{
                position: 'absolute',
                bottom: 4,
                left: '50%',
                transform: 'translateX(-50%)',
                px: 0.75,
                py: 0.1,
                borderRadius: '999px',
                bgcolor: 'rgba(15, 23, 42, 0.6)',
                color: 'var(--accent)',
                fontSize: '0.6rem',
                fontWeight: 600,
                whiteSpace: 'nowrap',
                pointerEvents: 'none',
              }}
            >
              {PLANE_META[p].short}
            </Box>
          </Box>
        ))}
      </Box>


      {/* ===== 切片导航条 ===== */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
          py: 0.25,
        }}
      >
        <IconButton size="small" onClick={() => stepSlice(-1)} disabled={cur <= 0}>
          <ChevronLeftIcon fontSize="small" />
        </IconButton>
        {/* 切片滑块：移动端主要翻片方式（手指无滚轮，拖动条比滑动更精细） */}
        <Slider
          size="small"
          value={cur}
          min={0}
          max={max}
          step={1}
          onChange={(_e, v) => setSlicePosition({ [active]: v as number })}
          disabled={!mriData}
          sx={{
            flex: 1,
            mx: 0.5,
            color: 'var(--accent)',
            // 加大触摸命中区，单指即可拖动
            '& .MuiSlider-thumb': { width: 16, height: 16 },
            '& .MuiSlider-rail': { opacity: 0.35 },
          }}
        />
        <IconButton size="small" onClick={() => stepSlice(1)} disabled={cur >= max}>
          <ChevronRightIcon fontSize="small" />
        </IconButton>
      </Box>
      {/* 切片位置 + 提示（小字辅助） */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', px: 0.5 }}>
        <Typography
          variant="caption"
          sx={{ fontFamily: 'monospace', color: 'var(--text-primary)', fontSize: '0.72rem' }}
        >
          切片 {cur + 1} / {max + 1}
        </Typography>
        <Typography variant="caption" sx={{ color: 'var(--text-disabled)', fontSize: '0.62rem' }}>
          拖动滑块 / 点小图切换
        </Typography>
      </Box>
    </Box>
  );
}
