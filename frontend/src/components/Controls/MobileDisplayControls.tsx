/**
 * ============================================================
 *  MobileDisplayControls —— 右栏/移动端“显示调节”内容组件
 * ============================================================
 * 紧凑排版（v2026-09-27-D4）：
 *  - 亮度（窗位 Window Level）/ 对比度（窗宽 Window Width）/ 分割叠加透明度 三个滑块；
 *  - 坏死核心 / 水肿区 / 增强肿瘤 三个区域显隐勾选；
 *  - 压掉 MUI Slider 根节点自带的上下 padding（之前滑块间隔过大的原因）；
 *  - 底部勾选项用固定 gap 分隔，避免图标与文字重叠。
 * 未加载 MRI 数据时全部禁用。
 */
import { Box, Typography, Slider, FormControlLabel, Checkbox } from '@mui/material';
import { useViewerStore } from '../../store/viewerStore';
import { LABEL_COLORS } from '../../config/labels';

/** 滑块组配置（key 对应 ViewSettings 数值字段） */
const SLIDER_ITEMS = [
  { key: 'windowLevel', label: '亮度 · 窗位', min: 0, max: 2000, step: 10, suffix: '' },
  { key: 'windowWidth', label: '对比度 · 窗宽', min: 1, max: 4000, step: 10, suffix: '' },
  { key: 'opacity', label: '分割叠加透明度', min: 0, max: 100, step: 1, suffix: '%' },
] as const;

/** 区域显隐条目（颜色与图例保持一致） */
const REGION_ITEMS = [
  { key: LABEL_COLORS.NCR.key, label: LABEL_COLORS.NCR.label, color: LABEL_COLORS.NCR.hex },
  { key: LABEL_COLORS.ED.key, label: LABEL_COLORS.ED.label, color: LABEL_COLORS.ED.hex },
  { key: LABEL_COLORS.ET.key, label: LABEL_COLORS.ET.label, color: LABEL_COLORS.ET.hex },
] as const;

/**
 * 显示调节内容组件（标题/折叠由外层 Accordion 控制）
 */
export default function MobileDisplayControls() {
  const viewSettings = useViewerStore((s) => s.viewSettings);
  const mriData = useViewerStore((s) => s.mriData);
  const { updateViewSettings } = useViewerStore((s) => s.actions);
  const disabled = !mriData;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', px: 1, py: 0.5 }}>
      {SLIDER_ITEMS.map((item, idx) => {
        const value = viewSettings[item.key];
        return (
          <Box key={item.key} sx={{ pt: idx === 0 ? 0 : 0.75 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', lineHeight: 1.2 }}>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.68rem' }}>
                {item.label}
              </Typography>
              <Typography
                variant="caption"
                sx={{ fontFamily: 'monospace', color: 'var(--accent)', fontSize: '0.68rem' }}
              >
                {value}{item.suffix}
              </Typography>
            </Box>
            <Slider
              size="small"
              value={value}
              min={item.min}
              max={item.max}
              step={item.step}
              disabled={disabled}
              onChange={(_e, v) =>
                updateViewSettings({ [item.key]: v as number } as Partial<typeof viewSettings>)
              }
              sx={{
                my: 0,
                /* 压掉 Slider 根节点自带的上下 padding（间隔过大的元凶）。
                   关键：用 &.MuiSlider-root 提高优先级 —— MUI 基础样式
                   `.MuiSlider-root { padding: '13px 0' }`（size=small 为 20px）
                   与 sx 生成的类同为 0-1-0 优先级，谁生效取决于 emotion 注入顺序，
                   会出现"刷新时紧凑、稳定后又被撑大"的现象；提升到 0-2-0 后
                   无论顺序如何都恒定生效。 */
                '&.MuiSlider-root': {
                  paddingTop: '0px',
                  paddingBottom: '0px',
                  marginTop: '0px',
                  marginBottom: '0px',
                },
                color: 'var(--accent)',
                '& .MuiSlider-thumb': {
                  width: 12, height: 12,
                  '&::after': { width: 20, height: 20 },
                },
                '& .MuiSlider-track': { height: 3 },
                '& .MuiSlider-rail': { height: 3 },
              }}
            />
          </Box>
        );
      })}

      {/* 区域显隐：一行勾选，固定 gap 分隔，杜绝图标/文字重叠 */}
      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          columnGap: 1.25,
          rowGap: 0.25,
          mt: 0.75,
          pt: 0.75,
          borderTop: '1px solid var(--border)',
        }}
      >
        {REGION_ITEMS.map((item) => (
          <FormControlLabel
            key={item.key}
            control={
              <Checkbox
                size="small"
                checked={viewSettings[item.key]}
                onChange={(e) =>
                  updateViewSettings({ [item.key]: e.target.checked } as Partial<typeof viewSettings>)
                }
                disabled={disabled}
                sx={{
                  /* 同样提高优先级，避免 MUI 基础 padding:9px / 默认色
                     依赖注入顺序反超（否则勾选框间距/颜色会"时而失效"）。
                     数值全部用带单位的字符串，避免序列化歧义。 */
                  '&.MuiCheckbox-root': { padding: '2px', color: item.color },
                  '&.MuiCheckbox-root.Mui-checked': { color: item.color },
                  '&.MuiCheckbox-root.Mui-disabled': { color: 'var(--text-disabled)' },
                }}
              />
            }
            label={
              <Typography
                variant="caption"
                sx={{
                  color: disabled ? 'var(--text-disabled)' : item.color,
                  fontSize: '0.68rem',
                  whiteSpace: 'nowrap',
                }}
              >
                {item.label}
              </Typography>
            }
            sx={{
              mr: 0,
              ml: 0,
              '&.MuiFormControlLabel-root': { marginLeft: 0, marginRight: 0 },
              '& .MuiFormControlLabel-label': { fontSize: '0.68rem' },
            }}
          />
        ))}
      </Box>
    </Box>
  );
}
