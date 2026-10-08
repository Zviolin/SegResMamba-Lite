/**
 * ============================================================
 *  Legend —— 分割标签图例组件
 * ============================================================
 * 职责：静态展示分割标签值与颜色的对应关系，以及常用快捷操作提示。
 * 纯展示组件，不持有状态、不发起请求。
 *
 * 颜色约定（与视图叠加 / 导出组件中的颜色保持一致）：
 *   - 增强肿瘤（ET）     标签值 3 或 4 -> 红色 #E57373
 *   - 水肿区（ED）       标签值 2     -> 绿色 #81C784
 *   - 坏死核心（NCR）    标签值 1     -> 蓝色 #64B5F6
 */
import { Box, Paper, Typography } from '@mui/material';
import { LABEL_COLORS } from '../../config/labels';

/**
 * 分割标签图例组件
 */
export default function Legend() {
  return (
    <Paper elevation={3} sx={{ p: 2, mt: 2 }}>
      {/* 图例标题 */}
      <Typography variant="subtitle2" gutterBottom>
        📊 分割标签图例
      </Typography>

      {/* 图例条目列表 */}
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
        {/* 增强肿瘤（ET）：红色方块 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 20,
              borderRadius: 1,
              bgcolor: LABEL_COLORS.ET.hex,
              flexShrink: 0
            }}
          />
          <Typography variant="body2">增强肿瘤 (ET) - 标签值 3 或 4</Typography>
        </Box>

        {/* 水肿区（ED）：绿色方块 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 20,
              borderRadius: 1,
              bgcolor: LABEL_COLORS.ED.hex,
              flexShrink: 0
            }}
          />
          <Typography variant="body2">水肿区 (ED) - 标签值 2</Typography>
        </Box>

        {/* 坏死核心（NCR）：蓝色方块 */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 20,
              borderRadius: 1,
              bgcolor: LABEL_COLORS.NCR.hex,
              flexShrink: 0
            }}
          />
          <Typography variant="body2">坏死核心 (NCR) - 标签值 1</Typography>
        </Box>
      </Box>

      {/* 快捷操作提示 */}
      <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid var(--border)' }}>
        <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.8, display: 'block' }}>
          ⌨️ 快捷操作：滚轮切换切片 | Shift+滚轮调节亮度 | Ctrl+滚轮调节对比度
        </Typography>
      </Box>
    </Paper>
  );
}
