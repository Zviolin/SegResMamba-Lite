/**
 * ============================================================
 *  PlaneViewer —— 单平面 Canvas 渲染组件
 * ============================================================
 * 职责：在单个 <canvas> 上渲染一个指定平面（轴向/冠状/矢状）的切片图像，
 * 并支持用鼠标滚轮切换该平面的切片索引（实现切片浏览）。
 *
 * 渲染流程：
 *  1. 通过 useViewSync 的 getCurrentSliceImage 生成当前切片的 ImageData；
 *  2. 在 useEffect 中把 ImageData 写入 canvas（putImageData）；
 *  3. 监听滚轮事件，更新 store 中的 slicePosition[plane]，从而触发重新渲染。
 *
 * 十字线：组件外部包了一层相对定位容器，内含两条绝对定位的半透明十字线，
 * 仅在 hover 时显示（纯视觉辅助，不参与交互）。
 */
import { useRef, useEffect, type MouseEvent } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import { useViewSync } from '../../hooks/useViewSync';
import { useViewerStore } from '../../store/viewerStore';
import { getVolumeDims } from '../../utils/niftiParser';

/** 把数值限制在 [min, max] 区间内 */
function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * PlaneViewer 组件 props
 */
interface PlaneViewerProps {
  /** 要渲染的平面类型：axial=轴位、coronal=冠状位、sagittal=矢状位 */
  plane: 'axial' | 'coronal' | 'sagittal';
  /** 标题文字（可选，显示在画布上方） */
  title: string;
}

/**
 * 单平面 Canvas 渲染组件
 *
 * @param props.plane 平面类型
 * @param props.title 标题文字
 */
export default function PlaneViewer({ plane, title }: PlaneViewerProps) {
  // canvas DOM 引用，用于在渲染时获取 2D 上下文
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // wrapBox 引用：用于手动注册 wheel 事件（passive: false 才能 preventDefault）
  const wrapBoxRef = useRef<HTMLDivElement>(null);
  // 从 useViewSync 获取切片 ImageData 生成函数与各平面最大索引
  const { getCurrentSliceImage, maxSlices } = useViewSync();
  // 订阅本平面的当前切片索引：用于底部「切片 x / y」文字。
  // 原先这里用 useViewerStore.getState() 读快照，组件本身不订阅，
  // 只有父组件碰巧整体订阅时才会更新（换到别处复用就不会刷新）。
  const currentSlice = useViewerStore((s) => s.slicePosition[plane]);
  // 渲染计数器：用于防止异步/过期渲染覆盖最新结果
  const renderRef = useRef<number>(0);

  /**
   * 渲染副作用：每当切片图像或平面变化时，把最新 ImageData 画到 canvas
   *
   * 使用 renderRef 做"最新一次渲染才生效"的守卫：
   * 若组件在渲染过程中因依赖变化被重新触发，旧渲染结果会被丢弃，
   * 避免快速滚动切片时出现画面闪烁或错乱。
   */
  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 自增渲染序号，标记本次渲染
    const currentRenderId = ++renderRef.current;

    // 生成当前平面、当前切片位置的 ImageData
    const { imageData, width, height } = getCurrentSliceImage(plane);

    // 仅当本次仍是最新渲染时写入 canvas（防止过期渲染覆盖）
    if (imageData && currentRenderId === renderRef.current) {
      canvas.width = width;
      canvas.height = height;
      ctx.putImageData(imageData, 0, 0);
    }
  }, [getCurrentSliceImage, plane]);

  /**
   * 滚轮事件处理：切换切片 / 调窗
   *
   * 逻辑：
   *  1. 阻止默认滚动行为（避免页面滚动干扰）；
   *  2. 根据滚轮方向决定步长：向下滚（deltaY>0）减 1，向上滚（deltaY<0）加 1；
   *  3. 修饰键分流（与 Legend 里的快捷操作提示对齐，此前提示了但没实现）：
   *       Shift+滚轮  → 调亮度（窗位 windowLevel）
   *       Ctrl / ⌘+滚轮 → 调对比度（窗宽 windowWidth）
   *       无修饰键      → 切换当前平面的切片
   *  4. 切片：读取当前平面的切片索引，加步长后 clamp 到 [0, max]，
   *     通过 store 的 setSlicePosition 更新，触发全部视图联动。
   *
   * React 17+ 默认把 wheel/touch 事件注册为 passive listener，导致 preventDefault
   * 无效并产生警告。这里改用 useEffect 手动 addEventListener 并显式传 passive: false。
   */
  useEffect(() => {
    const el = wrapBoxRef.current;
    if (!el) return;
    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const { setSlicePosition, updateViewSettings } = useViewerStore.getState().actions;
      const delta = e.deltaY < 0 ? 1 : -1;

      // Shift + 滚轮：调亮度（窗位 0~2000）
      if (e.shiftKey) {
        const cur = useViewerStore.getState().viewSettings.windowLevel;
        updateViewSettings({ windowLevel: clamp(cur + delta * 20, 0, 2000) });
        return;
      }
      // Ctrl / ⌘ + 滚轮：调对比度（窗宽 1~4000）
      if (e.ctrlKey || e.metaKey) {
        const cur = useViewerStore.getState().viewSettings.windowWidth;
        updateViewSettings({ windowWidth: clamp(cur + delta * 50, 1, 4000) });
        return;
      }

      // 无修饰键：切换切片
      const current = useViewerStore.getState().slicePosition[plane];
      const max = maxSlices[plane];
      const newValue = clamp(current + delta, 0, max);
      setSlicePosition({ [plane]: newValue });
    };
    el.addEventListener('wheel', handleWheel, { passive: false });
    return () => el.removeEventListener('wheel', handleWheel);
  }, [plane, maxSlices]);

  /**
   * 点击定位：把点击处的像素坐标换算成体素坐标，同步另外两个平面的切片。
   *
   * 位置信息面板底部写着「点击视图定位」，但此前没有任何点击处理，属于
   * 文案承诺了却没实现的功能。这里补上真正的十字线联动。
   *
   * 坐标换算两步：
   *  1. 屏幕坐标 → 位图坐标：canvas 用 object-fit: contain 等比缩放居中显示，
   *     需要先按实际缩放比去掉居中留白，再除以缩放比得到位图内像素坐标；
   *  2. 位图坐标 → 体素坐标：各平面的行/列含义不同，且 coronal / sagittal
   *     为符合医学显示习惯做了翻转（见 utils/niftiParser.getSliceData），
   *     这里必须按同样的翻转规则反推，否则定位会上下/左右颠倒。
   */
  const handleClick = (e: MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    const { mriData } = useViewerStore.getState();
    if (!canvas || !mriData) return;

    // ① 屏幕坐标 → canvas 位图坐标（去掉 contain 的居中留白）
    const rect = canvas.getBoundingClientRect();
    const bw = canvas.width;
    const bh = canvas.height;
    if (!bw || !bh) return;
    const scale = Math.min(rect.width / bw, rect.height / bh);
    if (scale <= 0) return;
    const offsetX = (rect.width - bw * scale) / 2;
    const offsetY = (rect.height - bh * scale) / 2;
    const u = (e.clientX - rect.left - offsetX) / scale; // 位图列（0 ~ bw）
    const v = (e.clientY - rect.top - offsetY) / scale;  // 位图行（0 ~ bh）
    if (u < 0 || u >= bw || v < 0 || v >= bh) return;    // 点在留白区，忽略

    // 位图宽高本身就等于该平面两个方向的体素数，故只需 z 方向的层数做翻转反推
    const dims = getVolumeDims(mriData);
    const ny = dims[2];
    const nz = dims[3];

    // ② 位图坐标 → 体素坐标（按 getSliceData 的翻转规则反推）
    let x = 0, y = 0, z = 0;
    if (plane === 'axial') {
      // 图像宽=X(nx)、高=Y(ny)，无翻转
      x = Math.floor(u);
      y = Math.floor(v);
      z = useViewerStore.getState().slicePosition.axial;
    } else if (plane === 'coronal') {
      // 图像宽=X(nx)、高=Z(nz)，Z 自顶向底为 nz-1 → 0
      x = Math.floor(u);
      z = nz - 1 - Math.floor(v);
      y = useViewerStore.getState().slicePosition.coronal;
    } else {
      // 图像宽=Y(ny)、高=Z(nz)，Y 与 Z 均翻转
      y = ny - 1 - Math.floor(u);
      z = nz - 1 - Math.floor(v);
      x = useViewerStore.getState().slicePosition.sagittal;
    }

    // ③ 把另外两个平面拉到该体素所在的切片（本平面切片不动）
    const { setSlicePosition } = useViewerStore.getState().actions;
    if (plane !== 'axial') setSlicePosition({ axial: clamp(z, 0, maxSlices.axial) });
    if (plane !== 'coronal') setSlicePosition({ coronal: clamp(y, 0, maxSlices.coronal) });
    if (plane !== 'sagittal') setSlicePosition({ sagittal: clamp(x, 0, maxSlices.sagittal) });
  };

  return (
    <Paper
      elevation={3}
      sx={{
        p: 1,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        width: '100%',
        height: '100%',
        minHeight: 0,
        bgcolor: '#000'
      }}
    >
      {/* 平面标题 */}
      <Typography
        variant="subtitle2"
        sx={{
          color: '#00adb5',
          mb: 1,
          alignSelf: 'flex-start',
          px: 1
        }}
      >
        {title}
      </Typography>

      {/* 画布容器：相对定位，hover 时显示十字线；ref 用于 wheel 事件 */}
      <Box
        ref={wrapBoxRef}
        sx={{
          position: 'relative',
          flex: 1,
          minHeight: 0,
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          '&:hover': {
            '& .crosshair-h': { opacity: 1 },
            '& .crosshair-v': { opacity: 1 }
          }
        }}
      >
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          style={{
            width: '100%',
            height: '100%',
            maxHeight: '100%',
            objectFit: 'contain',            
            imageRendering: 'pixelated', // 像素风渲染，保持医学影像的锐利观感
            cursor: 'crosshair'
          }}
        />

        {/* 水平十字线（绝对定位，默认隐藏，hover 显示） */}
        <Box
          className="crosshair-h"
          sx={{
            position: 'absolute',
            top: '50%',
            left: 0,
            right: 0,
            height: '1px',
            bgcolor: 'rgba(0, 173, 181, 0.5)',
            opacity: 0,
            transition: 'opacity 0.2s',
            pointerEvents: 'none'
          }}
        />

        {/* 垂直十字线（绝对定位，默认隐藏，hover 显示） */}
        <Box
          className="crosshair-v"
          sx={{
            position: 'absolute',
            left: '50%',
            top: 0,
            bottom: 0,
            width: '1px',
            bgcolor: 'rgba(0, 173, 181, 0.5)',
            opacity: 0,
            transition: 'opacity 0.2s',
            pointerEvents: 'none'
          }}
        />
      </Box>

      {/* 底部操作提示：滚轮切换切片 + 当前切片索引 / 最大索引 */}
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
        滚轮切换切片 | 切片 {currentSlice} / {maxSlices[plane]}
      </Typography>
    </Paper>
  );
}

