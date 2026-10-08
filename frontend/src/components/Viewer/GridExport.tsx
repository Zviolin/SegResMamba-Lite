/**
 * ============================================================
 *  GridExport —— 网格视图导出组件
 * ============================================================
 * 职责：把指定平面的多张切片以"网格"形式拼接到一张 Canvas 上，并支持导出为
 * PNG / PDF 图片，便于把分割结果直接存证或用于报告。
 *
 * 核心能力：
 *  1. 平面选择（轴位 / 冠状 / 矢状）；
 *  2. 布局模式：
 *       - N×N 网格：固定 2x2 ~ 6x6 的均匀切片；
 *       - 全部肿瘤切片：自动扫描所有含肿瘤的切片并排列成接近方形的网格；
 *  3. "仅含肿瘤"开关：只从肿瘤切片中均匀取样显示；
 *  4. 分割叠加：把分割标签以半透明颜色叠加到 MRI 灰度图（与主视图一致）；
 *  5. 导出：离屏渲染整张网格 Canvas 后，通过 canvas.toBlob 导出 PNG，
 *     或借助 jspdf 动态导入生成 PDF。
 *
 * 性能优化：含肿瘤切片列表使用 useRef 做 Map 缓存，同一平面只扫描一次。
 */
import { useState, useRef, useCallback, useMemo, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  ToggleButtonGroup,
  ToggleButton,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  GridView as GridIcon,
  Image as ImageIcon,
  PictureAsPdf as PdfIcon
} from '@mui/icons-material';
import { useViewerStore } from '../../store/viewerStore';
import { useVolumeDims } from '../../hooks/useVolumeDims';
import { getSliceData, sameShape } from '../../utils/niftiParser';
import { renderSlicePixels } from '../../utils/blend';
import type { PlaneType } from '../../types';
import type { MouseEvent } from 'react';
import type { SelectChangeEvent } from '@mui/material/Select';

/** 网格边长可选项（N×N 网格模式） */
type GridSize = 2 | 3 | 4 | 5 | 6;

/** 每个网格单元中图像的显示边长（像素） */
const CELL_SIZE = 180;
/** 每个网格单元顶部标签（切片编号）的高度（像素） */
const LABEL_HEIGHT = 22;
/** 网格单元之间的间距（像素） */
const GAP = 4;

/**
 * 网格视图导出组件
 */
export default function GridExport() {
  // 读取 MRI / 分割数据与视图设置（窗宽窗位、透明度等）
  const mriData = useViewerStore((s) => s.mriData);
  const segData = useViewerStore((s) => s.segData);
  const viewSettings = useViewerStore((s) => s.viewSettings);
  // 屏幕预览 Canvas 引用
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // ---- 导出配置状态 ----
  const [plane, setPlane] = useState<PlaneType>('axial');            // 当前平面
  const [gridSize, setGridSize] = useState<GridSize>(3);             // 固定网格边长
  const [includeSeg, setIncludeSeg] = useState(true);                // 是否叠加分割
  const [tumorOnly, setTumorOnly] = useState(true);                  // 是否仅含肿瘤切片
  const [autoArrange, setAutoArrange] = useState(false);             // 是否自动排列全部肿瘤切片
  const [exporting, setExporting] = useState<'none' | 'png' | 'pdf'>('none'); // 当前导出状态
  const [error, setError] = useState<string | null>(null);           // 导出错误信息

  /**
   * 体数据维度与三平面最大切片索引（缓存）：统一取自 useVolumeDims
   */
  const { dims, maxSlices } = useVolumeDims();
  const maxSlice = maxSlices[plane];

  // 扫描所有含肿瘤的切片（仅首次推理后扫描一次，后续命中缓存）
  // 缓存键 = "平面-维度"，用 useRef 保存 Map 以免每次渲染都重新扫描
  const tumorSliceCache = useRef<Map<string, number[]>>(new Map());

  /**
   * 所有含肿瘤的切片索引列表（缓存）
   * 逐一切片调用 getSliceData 并检查是否存在任意非零标签体素。
   */
  const allTumorSlices = useMemo(() => {
    if (!segData) return [];
    // 缓存键：平面 + 分割数据维度（数据未变则复用上次扫描结果）
    const cacheKey = `${plane}-${segData.header.dims?.join(',')}`;
    let cached = tumorSliceCache.current.get(cacheKey);
    if (!cached) {
      cached = [];
      for (let i = 0; i <= maxSlice; i++) {
        const { pixels } = getSliceData(
          segData.typedArray,
          segData.header.dims || dims,
          i,
          plane
        );
        // 若该切片存在任一标签值 > 0 的体素，则认为其含肿瘤
        if (pixels.some(p => Math.round(p) > 0)) {
          cached.push(i);
        }
      }
      tumorSliceCache.current.set(cacheKey, cached);
    }
    return cached;
  }, [segData, dims, plane, maxSlice]);

  // 根据模式决定网格列数/行数
  /**
   * 网格列数（缓存）
   * 自动排列模式：按切片数量取平方根向上取整，尽量接近方形；
   * 固定模式：直接使用 gridSize。
   */
  const gridCols = useMemo(() => {
    if (autoArrange) {
      // 自动排列：根据切片数量计算列数，尽量接近方形
      const n = allTumorSlices.length || gridSize * gridSize;
      return Math.max(1, Math.ceil(Math.sqrt(n)));
    }
    return gridSize;
  }, [autoArrange, allTumorSlices, gridSize]);

  /**
   * 网格行数（缓存）
   * 自动排列：总切片数除以列数向上取整；固定模式：直接使用 gridSize。
   */
  const gridRows = useMemo(() => {
    if (autoArrange) {
      const n = allTumorSlices.length || gridSize * gridSize;
      return Math.ceil(n / gridCols);
    }
    return gridSize;
  }, [autoArrange, allTumorSlices, gridCols, gridSize]);

  // 决定显示哪些切片
  /**
   * 决定网格中实际显示哪些切片索引（缓存）
   *
   * 逻辑分支：
   *  1. autoArrange + 有分割：直接返回所有肿瘤切片；
   *  2. autoArrange + 无肿瘤切片：降级为均匀分布（去掉首尾 5% 边缘）；
   *  3. tumorOnly + 有分割：从肿瘤切片中均匀取样 n 张（不足则重复最后一张）；
   *  4. tumorOnly + 无肿瘤切片：降级为均匀分布；
   *  5. 默认（均匀分布）：在 [margin, maxSlice-margin] 内均匀取 n 张。
   * 其中 n = gridSize * gridSize（固定模式的目标张数）。
   */
  const sliceIndices = useMemo(() => {
    const n = gridSize * gridSize;
    if (maxSlice <= 0) return [];

    // 自动排列：显示所有肿瘤切片
    if (autoArrange && segData) {
      if (allTumorSlices.length === 0) {
        // 没有肿瘤切片，降级为均匀分布
        const margin = Math.max(1, Math.floor(maxSlice * 0.05));
        const start = margin;
        const end = maxSlice - margin;
        if (end <= start) return Array.from({ length: n }, () => 0);
        const step = (end - start) / (n - 1);
        return Array.from({ length: n }, (_, i) => Math.round(start + i * step));
      }
      return allTumorSlices;
    }

    // 「仅含肿瘤」模式：从肿瘤切片中均匀取样
    if (tumorOnly && segData) {
      if (allTumorSlices.length === 0) {
        const margin = Math.max(1, Math.floor(maxSlice * 0.05));
        const start = margin;
        const end = maxSlice - margin;
        if (end <= start) return Array.from({ length: n }, () => 0);
        const step = (end - start) / (n - 1);
        return Array.from({ length: n }, (_, i) => Math.round(start + i * step));
      }
      // 肿瘤切片不足 n 张时，用最后一张补齐剩余位置
      if (allTumorSlices.length <= n) {
        return Array.from({ length: n }, (_, i) =>
          allTumorSlices[Math.min(i, allTumorSlices.length - 1)]
        );
      }
      // 肿瘤切片多于 n 张时，等间距取样
      const step = (allTumorSlices.length - 1) / (n - 1);
      return Array.from({ length: n }, (_, i) =>
        allTumorSlices[Math.round(i * step)]
      );
    }

    // 均匀分布模式（默认）：去掉首尾 5% 边缘后等间距取 n 张
    const margin = Math.max(1, Math.floor(maxSlice * 0.05));
    const start = margin;
    const end = maxSlice - margin;
    if (end <= start) return Array.from({ length: n }, () => 0);
    const step = (end - start) / (n - 1);
    return Array.from({ length: n }, (_, i) =>
      Math.round(start + i * step)
    );
  // 依赖里不需要 plane：平面差异已由 maxSlice 体现（不同平面切片数不同），
  // 且真正取切片像素的 renderGrid 单独依赖了 plane，切平面时会正常重绘。
  }, [gridSize, maxSlice, tumorOnly, autoArrange, allTumorSlices, segData]);

  // 渲染网格到 canvas
  /**
   * 把切片网格渲染到指定的 canvas（屏幕预览或离屏导出共用）
   *
   * @param targetCanvas 目标 canvas 元素（会被设置宽高并绘制）
   *
   * 绘制步骤：
   *  1. 根据列数/行数计算整张画布尺寸，设置背景色；
   *  2. 逐网格单元：
   *       a. 绘制切片编号标签（#n）；
   *       b. 用 getSliceData 取该切片 MRI 像素（与主视图相同索引算法）；
   *       c. 若叠加分割，取同一位置的分割标签像素；
   *       d. 逐像素：窗宽窗位映射成灰度，再按标签颜色与透明度混合；
   *       e. putImageData 到临时切片 canvas，再 drawImage 缩放绘制到目标；
   *       f. 绘制单元边框；
   *  3. 最后绘制整张画布的外边框。
   */
  const renderGrid = useCallback((targetCanvas: HTMLCanvasElement) => {
    if (!mriData) return;

    // 计算整张网格画布的宽高（含标签区与间距）
    const cols = gridCols;
    const rows = gridRows;
    const totalW = cols * (CELL_SIZE + GAP) + GAP;
    const totalH = rows * (CELL_SIZE + LABEL_HEIGHT + GAP) + GAP;

    targetCanvas.width = totalW;
    targetCanvas.height = totalH;
    const ctx = targetCanvas.getContext('2d')!;

    // 背景
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, totalW, totalH);

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // 逐网格单元绘制
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        // 网格单元在 sliceIndices 中的线性序号
        const idx = row * cols + col;
        const sliceIdx = sliceIndices[idx];
        if (sliceIdx === undefined) continue; // 超出实际切片数的单元跳过
        // 单元左上角坐标
        const x = GAP + col * (CELL_SIZE + GAP);
        const y = GAP + row * (CELL_SIZE + LABEL_HEIGHT + GAP);

        // 切片标签（青色编号）
        ctx.fillStyle = '#00adb5';
        ctx.font = '11px monospace';
        ctx.fillText(`#${sliceIdx}`, x + CELL_SIZE / 2, y + LABEL_HEIGHT / 2);

        // 获取该切片的 MRI 像素
        const { pixels, width, height } = getSliceData(
          mriData.typedArray,
          dims,
          sliceIdx,
          plane
        );

        // 若开启分割叠加且有分割数据，则取同一切片的标签像素。
        // 与主视图同样的形状兜底：形状不符时不叠加，避免导出图里的标签画错位置。
        const segPixels = (segData && includeSeg && sameShape(segData.header.dims, dims))
          ? getSliceData(
              segData.typedArray,
              segData.header.dims,
              sliceIdx,
              plane
            ).pixels
          : null;

        // 创建该切片的临时画布与 ImageData
        const sliceCanvas = document.createElement('canvas');
        sliceCanvas.width = width;
        sliceCanvas.height = height;
        const sliceCtx = sliceCanvas.getContext('2d')!;
        const imageData = sliceCtx.createImageData(width, height);

        // 逐像素生成颜色：与主视图共用 renderSlicePixels（颜色统一取自 LABEL_COLORS，
        // 否则导出的 PNG/PDF 与屏幕观感不一致）
        renderSlicePixels(pixels, segPixels, imageData, viewSettings);

        // 把该切片像素写入临时画布
        sliceCtx.putImageData(imageData, 0, 0);

        // 绘制到目标 canvas，缩放至 CELL_SIZE（drawImage 自带缩放）
        const drawW = CELL_SIZE;
        const drawH = CELL_SIZE;
        const sx = 0, sy = 0, sw = width, sh = height;
        ctx.drawImage(sliceCanvas, sx, sy, sw, sh, x, y + LABEL_HEIGHT, drawW, drawH);

        // 单元边框
        ctx.strokeStyle = '#444';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y + LABEL_HEIGHT, drawW, drawH);
      }
    }

    // 整体边框
    ctx.strokeStyle = '#555';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, totalW - 1, totalH - 1);
  }, [mriData, segData, dims, plane, gridCols, gridRows, sliceIndices, includeSeg, viewSettings]);

  /**
   * 渲染到离屏 canvas（用于导出），不干扰屏幕预览
   */
  const renderToOffscreen = useCallback(() => {
    const offscreen = document.createElement('canvas');
    renderGrid(offscreen);
    return offscreen;
  }, [renderGrid]);

  /**
   * 渲染到屏幕预览 canvas
   *
   * 这里原先是在 useState 初始器里直接画 canvas，并在 render 阶段改 ref + setTimeout
   * 调度重绘 —— 属于渲染期副作用（StrictMode 下双调用），且用于比对的参数签名漏了
   * showNecrosis / showEdema / showEnhancing，导致勾选区域显隐后网格图不刷新。
   *
   * 现在改由 renderGrid 自身的依赖驱动：它依赖 mriData / segData / plane /
   * gridCols / gridRows / sliceIndices / includeSeg / viewSettings（对象引用变化即重绘，
   * 窗宽窗位、透明度、三个显隐开关全部覆盖）。
   */
  useEffect(() => {
    if (!mriData || !canvasRef.current) return;
    renderGrid(canvasRef.current);
  }, [renderGrid, mriData]);

/**
 * 等待浏览器完成一帧渲染
 *
 * 用途：导出前先让 exporting 状态刷到界面（按钮进入 loading），再做耗时的离屏渲染。
 * 原先这里写死 sleep(100) —— 慢设备上 100ms 可能还没渲染完，快设备又白等一整。
 * 等一帧既保证状态已上屏，又不引入多余的固定延迟。
 */
const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  const handleExportPng = useCallback(async () => {
    if (!mriData) return;
    setExporting('png');
    setError(null);
    try {
      // 等一帧，确保 exporting 状态已渲染（按钮显示 loading）
      await nextFrame();
      const canvas = renderToOffscreen();
      // toBlob 把 canvas 编码为 PNG Blob
      canvas.toBlob((blob) => {
        if (!blob) { setError('生成图片失败'); setExporting('none'); return; }
        // 创建临时 <a> 触发浏览器下载
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = autoArrange
          ? `brain-tumor-grid-${plane}-all-tumor-slices.png`
          : `brain-tumor-grid-${plane}-${gridSize}x${gridSize}.png`;
        a.click();
        // 释放对象 URL 与导出状态
        URL.revokeObjectURL(url);
        setExporting('none');
      }, 'image/png');
    } catch (e) {
      setError(`导出失败: ${e}`);
      setExporting('none');
    }
  }, [mriData, renderToOffscreen, plane, gridSize, autoArrange]);

  // 导出 PDF
  /**
   * 导出为 PDF 文件
   * 动态导入 jspdf（按需加载），把离屏 canvas 转成 PNG 数据后写入 PDF 并下载。
   */
  const handleExportPdf = useCallback(async () => {
    if (!mriData) return;
    setExporting('pdf');
    setError(null);
    try {
      await nextFrame();
      const canvas = renderToOffscreen();

      // 按需动态加载 jsPDF，减小首屏体积
      const { jsPDF } = await import('jspdf');
      const imgData = canvas.toDataURL('image/png');

      // 根据画布宽高决定页面方向与尺寸（px 单位，四周留白）
      const pdf = new jsPDF({
        orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
        unit: 'px',
        format: [canvas.width + 40, canvas.height + 60]
      });

      // 标题
      const planeNames = { axial: '轴位 Axial', coronal: '冠状位 Coronal', sagittal: '矢状位 Sagittal' };
      const gridLabel = autoArrange
        ? `${sliceIndices.length} 张肿瘤切片`
        : `${gridSize}x${gridSize} 网格`;
      pdf.setFontSize(12);
      pdf.text(
        `脑肿瘤分割 - ${planeNames[plane]} - ${gridLabel}`,
        20, 20
      );

      // 把整张网格图写入 PDF 并保存
      pdf.addImage(imgData, 'PNG', 20, 30, canvas.width, canvas.height);
      pdf.save(autoArrange
        ? `brain-tumor-grid-${plane}-all-tumor-slices.pdf`
        : `brain-tumor-grid-${plane}-${gridSize}x${gridSize}.pdf`);

      setExporting('none');
    } catch (e) {
      setError(`PDF 导出失败: ${e}`);
      setExporting('none');
    }
  }, [mriData, renderToOffscreen, plane, gridSize, autoArrange, sliceIndices]);

  /** 平面切换（ToggleButtonGroup 回调，空值忽略） */
  const handlePlaneChange = (_: MouseEvent<HTMLElement>, val: PlaneType | null) => {
    if (val) setPlane(val);
  };

  /** 网格大小切换（Select 回调，字符串转数字） */
  const handleGridChange = (e: SelectChangeEvent<unknown>) => {
    setGridSize(Number(e.target.value) as GridSize);
  };

  return (
    <Paper elevation={3} sx={{ p: 2, mt: 2, bgcolor: '#1e1e1e' }}>
      {/* 标题 */}
      <Typography variant="h6" gutterBottom sx={{ color: '#00adb5', display: 'flex', alignItems: 'center', gap: 1 }}>
        <GridIcon /> 网格视图导出
      </Typography>

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, alignItems: 'center', mb: 2 }}>
        {/* 平面选择 */}
        <ToggleButtonGroup
          value={plane}
          exclusive
          onChange={handlePlaneChange}
          size="small"
          sx={{
            '& .MuiToggleButton-root': {
              color: '#aaa', borderColor: '#444',
              '&.Mui-selected': { color: '#00adb5', bgcolor: 'rgba(0,173,181,0.15)' }
            }
          }}
        >
          <ToggleButton value="axial">轴位</ToggleButton>
          <ToggleButton value="coronal">冠状位</ToggleButton>
          <ToggleButton value="sagittal">矢状位</ToggleButton>
        </ToggleButtonGroup>

        {/* 布局模式 */}
        <ToggleButtonGroup
          value={autoArrange ? 'auto' : 'fixed'}
          exclusive
          size="small"
          onChange={(_, val) => val && setAutoArrange(val === 'auto')}
          sx={{
            '& .MuiToggleButton-root': {
              color: '#aaa', borderColor: '#444',
              '&.Mui-selected': { color: '#00adb5', bgcolor: 'rgba(0,173,181,0.15)' }
            }
          }}
        >
          <ToggleButton value="fixed">N×N 网格</ToggleButton>
          <ToggleButton value="auto">全部肿瘤切片</ToggleButton>
        </ToggleButtonGroup>

        {/* 网格大小选择（固定网格模式） */}
        <FormControl size="small" sx={{ minWidth: 100 }} disabled={autoArrange}>
          <InputLabel sx={{ color: '#aaa' }}>网格大小</InputLabel>
          <Select
            value={gridSize}
            onChange={handleGridChange}
            label="网格大小"
            sx={{ color: '#e0e0e0', '& .MuiOutlinedInput-notchedOutline': { borderColor: '#444' } }}
          >
            {[2, 3, 4, 5, 6].map(n => (
              <MenuItem key={n} value={n}>{n} x {n}（{n * n} 切片）</MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* 显示分割（叠加开关） */}
        <Button
          variant={includeSeg ? 'contained' : 'outlined'}
          size="small"
          onClick={() => setIncludeSeg(!includeSeg)}
          sx={{
            color: includeSeg ? '#fff' : '#aaa',
            bgcolor: includeSeg ? '#00adb5' : 'transparent',
            '&:hover': { bgcolor: includeSeg ? '#008f9a' : 'rgba(0,173,181,0.1)' }
          }}
        >
          分割叠加
        </Button>

        {/* 仅含肿瘤模式（有分割数据时才显示） */}
        {segData && (
          <Button
            variant={tumorOnly ? 'contained' : 'outlined'}
            size="small"
            onClick={() => setTumorOnly(!tumorOnly)}
            sx={{
              color: tumorOnly ? '#fff' : '#aaa',
              bgcolor: tumorOnly ? '#e65100' : 'transparent',
              borderColor: '#e65100',
              '&:hover': { bgcolor: tumorOnly ? '#bf360c' : 'rgba(230,81,0,0.1)' }
            }}
          >
            {tumorOnly ? '仅含肿瘤' : '全部切片'}
          </Button>
        )}

        {/* 导出按钮 */}
        <Button
          variant="contained"
          size="small"
          startIcon={exporting === 'png' ? <CircularProgress size={16} color="inherit" /> : <ImageIcon />}
          onClick={handleExportPng}
          disabled={!mriData || exporting !== 'none'}
          sx={{ bgcolor: '#2e7d32', '&:hover': { bgcolor: '#1b5e20' } }}
        >
          PNG
        </Button>
        <Button
          variant="contained"
          size="small"
          startIcon={exporting === 'pdf' ? <CircularProgress size={16} color="inherit" /> : <PdfIcon />}
          onClick={handleExportPdf}
          disabled={!mriData || exporting !== 'none'}
          sx={{ bgcolor: '#c62828', '&:hover': { bgcolor: '#b71c1c' } }}
        >
          PDF
        </Button>
      </Box>

      {/* 导出错误提示 */}
      {error && <Alert severity="error" sx={{ mb: 1, fontSize: '0.85rem' }}>{error}</Alert>}

      {/* 信息提示：根据当前模式显示切片统计说明 */}
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
        {autoArrange && segData
          ? `全部肿瘤切片 | 共 ${allTumorSlices.length} 张含肿瘤切片（排列为 ${gridCols} 列 × ${gridRows} 行）`
          : tumorOnly && segData
            ? `仅含肿瘤切片 | 显示 ${sliceIndices.length} 张（共 ${allTumorSlices.length} 张含肿瘤）`
            : `切片范围: 0 ~ ${maxSlice} | 显示 ${sliceIndices.length} 张均匀分布切片`
        }
      </Typography>

      {/* 网格 canvas 预览区 */}
      <Box sx={{
        display: 'flex',
        justifyContent: 'center',
        bgcolor: '#111',
        borderRadius: 1,
        p: 1,
        minHeight: 120,
        alignItems: 'center'
      }}>
        {mriData ? (
          <canvas
            ref={canvasRef}
            style={{ maxWidth: '100%', height: 'auto', imageRendering: 'pixelated' }}
          />
        ) : (
          <Typography variant="body2" color="text.secondary">
            请先加载 MRI 文件
          </Typography>
        )}
      </Box>
    </Paper>
  );
}
