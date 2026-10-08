/**
 * ============================================================
 *  ControlPanel —— 显示控制面板组件
 * ============================================================
 * 职责：提供所有"显示参数"的调节控件：
 *  1. 三平面切片位置滑块（轴位/冠状/矢状）—— 与 MPR 视图联动；
 *  2. 窗宽（对比度）与窗位（亮度）滑块 —— 控制灰度映射；
 *  3. 分割叠加透明度滑块；
 *  4. 各肿瘤区域（坏死核心/水肿区/增强肿瘤）的显隐勾选框。
 *
 * 所有变更都通过 store 的 updateViewSettings / setSlicePosition 写入全局，
 * 从而驱动所有视图与导出组件同步更新。
 */
import { useCallback } from 'react';
import {
  Box,
  Typography,
  Slider,
  Paper,
  FormControlLabel,
  Checkbox,
  FormGroup
} from '@mui/material';
import { Tune as TuneIcon } from '@mui/icons-material';
import { useViewerStore } from '../../store/viewerStore';
import { useVolumeDims } from '../../hooks/useVolumeDims';

/**
 * 显示控制面板组件
 */
export default function ControlPanel() {
  // 读取视图设置、切片位置与 MRI 数据（用于计算滑块范围）
  const viewSettings = useViewerStore((s) => s.viewSettings);
  const slicePosition = useViewerStore((s) => s.slicePosition);
  const mriData = useViewerStore((s) => s.mriData);
  const { updateViewSettings, setSlicePosition } = useViewerStore((s) => s.actions);

  // 体数据维度与各平面最大切片索引（未加载时回退默认值），用于限制滑块最大值。
  // 统一取自 useVolumeDims，不在本组件里重复兜底值与减一计算。
  const { maxSlices } = useVolumeDims();

  // ---- 窗宽窗位与透明度滑块回调 ----
  /** 窗宽滑块：更新 viewSettings.windowWidth */
  const handleWindowWidthChange = useCallback((_: Event, value: number | number[]) => {
    updateViewSettings({ windowWidth: value as number });
  }, [updateViewSettings]);

  /** 窗位滑块：更新 viewSettings.windowLevel */
  const handleWindowLevelChange = useCallback((_: Event, value: number | number[]) => {
    updateViewSettings({ windowLevel: value as number });
  }, [updateViewSettings]);

  /** 透明度滑块：更新 viewSettings.opacity */
  const handleOpacityChange = useCallback((_: Event, value: number | number[]) => {
    updateViewSettings({ opacity: value as number });
  }, [updateViewSettings]);

  // ---- 三平面切片位置滑块回调 ----
  /** 轴位切片滑块：更新 slicePosition.axial */
  const handleAxialSliceChange = useCallback((_: Event, value: number | number[]) => {
    setSlicePosition({ axial: value as number });
  }, [setSlicePosition]);

  /** 冠状位切片滑块：更新 slicePosition.coronal */
  const handleCoronalSliceChange = useCallback((_: Event, value: number | number[]) => {
    setSlicePosition({ coronal: value as number });
  }, [setSlicePosition]);

  /** 矢状位切片滑块：更新 slicePosition.sagittal */
  const handleSagittalSliceChange = useCallback((_: Event, value: number | number[]) => {
    setSlicePosition({ sagittal: value as number });
  }, [setSlicePosition]);

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      {/* 标题 */}
      <Typography variant="h6" gutterBottom sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 1 }}>
        <TuneIcon /> 显示控制
      </Typography>

      {/* 轴位切片滑块 */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          轴状位切片: {slicePosition.axial} / {maxSlices.axial}
        </Typography>
        <Slider
          value={slicePosition.axial}
          onChange={handleAxialSliceChange}
          min={0}
          max={maxSlices.axial}
          disabled={!mriData}
          sx={{ mt: 1 }}
        />
      </Box>

      {/* 冠状位切片滑块 */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          冠状位切片: {slicePosition.coronal} / {maxSlices.coronal}
        </Typography>
        <Slider
          value={slicePosition.coronal}
          onChange={handleCoronalSliceChange}
          min={0}
          max={maxSlices.coronal}
          disabled={!mriData}
          sx={{ mt: 1 }}
        />
      </Box>

      {/* 矢状位切片滑块 */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          矢状位切片: {slicePosition.sagittal} / {maxSlices.sagittal}
        </Typography>
        <Slider
          value={slicePosition.sagittal}
          onChange={handleSagittalSliceChange}
          min={0}
          max={maxSlices.sagittal}
          disabled={!mriData}
          sx={{ mt: 1 }}
        />
      </Box>

      <Box sx={{ mt: 3 }} />

      {/* 窗宽窗位分区标题 */}
      <Typography variant="subtitle2" sx={{ mb: 1 }}>窗宽窗位调节</Typography>

      {/* 窗宽滑块（对比度） */}
      <Box sx={{ mt: 1 }}>
        <Typography variant="body2" color="text.secondary">
          窗宽 (对比度): {viewSettings.windowWidth}
        </Typography>
        <Slider
          value={viewSettings.windowWidth}
          onChange={handleWindowWidthChange}
          min={1}
          max={4000}
          step={10}
          sx={{ mt: 1 }}
        />
      </Box>

      {/* 窗位滑块（亮度） */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          窗位 (亮度): {viewSettings.windowLevel}
        </Typography>
        <Slider
          value={viewSettings.windowLevel}
          onChange={handleWindowLevelChange}
          min={0}
          max={2000}
          step={10}
          sx={{ mt: 1 }}
        />
      </Box>

      {/* 分割叠加透明度滑块 */}
      <Box sx={{ mt: 2 }}>
        <Typography variant="body2" color="text.secondary">
          分割叠加透明度: {viewSettings.opacity}%
        </Typography>
        <Slider
          value={viewSettings.opacity}
          onChange={handleOpacityChange}
          min={0}
          max={100}
          sx={{ mt: 1 }}
        />
      </Box>

      {/* 各肿瘤区域显隐勾选 */}
      <FormGroup sx={{ mt: 3 }}>
        <Typography variant="subtitle2" sx={{ mb: 1 }}>显示区域</Typography>

        {/* 坏死核心（NCR，标签值 1，蓝色） */}
        <FormControlLabel
          control={
            <Checkbox
              checked={viewSettings.showNecrosis}
              onChange={(e) => updateViewSettings({ showNecrosis: e.target.checked })}
              size="small"
              sx={{ color: '#64B5F6', '&.Mui-checked': { color: '#64B5F6' } }}
            />
          }
          label={<Typography variant="body2" sx={{ color: '#64B5F6' }}>坏死核心</Typography>}
        />

        {/* 水肿区（ED，标签值 2，绿色） */}
        <FormControlLabel
          control={
            <Checkbox
              checked={viewSettings.showEdema}
              onChange={(e) => updateViewSettings({ showEdema: e.target.checked })}
              size="small"
              sx={{ color: '#81C784', '&.Mui-checked': { color: '#81C784' } }}
            />
          }
          label={<Typography variant="body2" sx={{ color: '#81C784' }}>水肿区</Typography>}
        />

        {/* 增强肿瘤（ET，标签值 3/4，红色） */}
        <FormControlLabel
          control={
            <Checkbox
              checked={viewSettings.showEnhancing}
              onChange={(e) => updateViewSettings({ showEnhancing: e.target.checked })}
              size="small"
              sx={{ color: '#E57373', '&.Mui-checked': { color: '#E57373' } }}
            />
          }
          label={<Typography variant="body2" sx={{ color: '#E57373' }}>增强肿瘤</Typography>}
        />
      </FormGroup>
    </Paper>
  );
}
