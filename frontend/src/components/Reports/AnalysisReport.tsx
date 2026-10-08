/**
 * ============================================================
 *  AnalysisReport —— 量化分析报告组件
 * ============================================================
 * 职责：在右侧面板展示肿瘤分割的量化分析报告：
 *  1. 有 MRI + 分割数据时，自动调用 calculateTumorStatistics 计算统计并写入 store；
 *  2. 若存在云端推理结果，额外展示"云端推理结果"卡片（模型、耗时、WT 体积）；
 *  3. 展示总体统计（总体积、受影响切片数、总体素数、切片范围）；
 *  4. 以表格 + 进度条展示各区域（NCR/ED/ET）的体素数、体积与占比。
 *
 * 数据来源：统计结果直接从全局 store 读取（statistics），
 * 由本组件的 useEffect 在 mriData / segData 就绪时计算一次。
 */
import { useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Chip
} from '@mui/material';
import { Assessment as AssessmentIcon, Cloud as CloudIcon } from '@mui/icons-material';
import { useViewerStore } from '../../store/viewerStore';
import { calculateTumorStatistics, formatVolume } from '../../utils/statistics';
import { getVolumeDims } from '../../utils/niftiParser';
import type { NiftiData } from '../../types';

/**
 * 量化分析报告组件
 */
export default function AnalysisReport() {
  // 读取 MRI / 分割数据与推理状态
  const mriData = useViewerStore((s) => s.mriData);
  const segData = useViewerStore((s) => s.segData);
  const inference = useViewerStore((s) => s.inference);
  const { setStatistics } = useViewerStore((s) => s.actions);
  // 用选择器单独订阅 statistics，避免整个 store 变化都触发重渲染
  const statistics = useViewerStore((state) => state.statistics);

  /**
   * 副作用：当 MRI 与分割数据都就绪时，自动计算肿瘤统计并写入 store。
   *
   * 去重：用 ref 记住上次是「哪一对 (segData, mriData)」算的，只有数据真的换了才重算。
   * 此前推理成功后 useInference 会算一次、本组件挂载/重渲染再算一次（900 万体素遍历两遍），
   * 而且本组件在折叠面板里反复开合会反复触发。
   */
  const lastComputedRef = useRef<{ seg: NiftiData | null; mri: NiftiData | null } | null>(null);
  useEffect(() => {
    if (!mriData || !segData) return;
    const last = lastComputedRef.current;
    if (last && last.seg === segData && last.mri === mriData) return; // 同一份数据已算过
    lastComputedRef.current = { seg: segData, mri: mriData };
    try {
      const stats = calculateTumorStatistics(
        segData.typedArray,
        mriData.header
      );
      setStatistics(stats);
    } catch (error) {
      console.error('Statistics calculation error:', error);
    }
  }, [mriData, segData, setStatistics]);

  // 尚无统计或 MRI 数据：显示"请上传数据"占位
  if (!statistics || !mriData) {
    return (
      <Paper elevation={3} sx={{ p: 1, mt: 1 }}>
        <Typography
          variant="subtitle2"
          gutterBottom
          sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 0.5, fontSize: '0.78rem' }}
        >
          <AssessmentIcon sx={{ fontSize: 16 }} /> 量化分析报告
        </Typography>
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography color="text.secondary" sx={{ fontSize: '0.7rem' }}>
            请上传 MRI 和分割标签文件以生成分析报告
          </Typography>
        </Box>
      </Paper>
    );
  }

  // 占比计算：总体素数为 0（分割结果全背景）时直接给 0，避免出现 NaN% 与 NaN 进度条
  const percentOf = (voxels: number): number =>
    statistics.totalVoxels > 0 ? (voxels / statistics.totalVoxels) * 100 : 0;

  // 表格数据行：各区域的标签、体素数、体积与颜色
  const dataRows = [
    {
      label: '坏死核心 (NCR)',
      voxels: statistics.necrosisVoxels,
      volume: statistics.necrosisVolume,
      color: '#64B5F6'
    },
    {
      label: '水肿区 (ED)',
      voxels: statistics.edemaVoxels,
      volume: statistics.edemaVolume,
      color: '#81C784'
    },
    {
      label: '增强肿瘤 (ET)',
      voxels: statistics.enhancingVoxels,
      volume: statistics.enhancingVolume,
      color: '#E57373'
    }
  ];

  return (
    <Paper elevation={3} sx={{ p: 1, mt: 1 }}>
      {/* 标题 */}
      <Typography
        variant="subtitle2"
        gutterBottom
        sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 0.5, fontSize: '0.78rem' }}
      >
        <AssessmentIcon sx={{ fontSize: 16 }} /> 量化分析报告
      </Typography>

      {/* 云端推理结果卡片（有推理结果时显示） */}
      {inference.result && (
        <Box sx={{ mb: 1, p: 1, bgcolor: 'var(--bg-inset)', borderRadius: 1, borderLeft: '3px solid var(--accent)' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.25 }}>
            <CloudIcon sx={{ color: 'var(--accent)', fontSize: 14 }} />
            <Typography variant="caption" sx={{ color: 'var(--accent)', fontWeight: 600, fontSize: '0.7rem' }}>
              云端推理结果
            </Typography>
            {/* 模型名 Chip */}
            <Chip
              label={inference.result.modelUsed}
              size="small"
              sx={{ ml: 'auto', bgcolor: 'var(--accent)', color: '#000', fontWeight: 'bold', height: 18, fontSize: '0.6rem' }}
            />
          </Box>
          <Box sx={{ display: 'flex', gap: 2 }}>
            {/* 推理耗时 */}
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
              推理耗时: <span style={{ color: 'var(--accent)' }}>{(inference.result.executionTimeMs / 1000).toFixed(2)} 秒</span>
            </Typography>
            {/* WT（全肿瘤）体积 */}
            {inference.result.volumes && Object.keys(inference.result.volumes).length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                WT: <span style={{ color: 'var(--accent)' }}>{(inference.result.volumes['WT'] || 0).toFixed(2)} cm³</span>
              </Typography>
            )}
          </Box>
        </Box>
      )}

      <Box sx={{ mt: 1 }}>
        {/* 总体统计标题 */}
        <Typography variant="caption" sx={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.7rem', display: 'block', mb: 0.5 }}>
          总体统计
        </Typography>

        {/* 四宫格总体指标卡片 */}
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, mb: 1.5 }}>
          {/* 肿瘤总体积 */}
          <Box sx={{ p: 1, bgcolor: 'var(--bg-inset)', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.62rem', display: 'block' }}>
              肿瘤总体积
            </Typography>
            <Typography variant="subtitle1" sx={{ color: 'var(--accent)', mt: 0.25, lineHeight: 1.2, fontSize: '0.85rem', fontWeight: 600 }}>
              {formatVolume(statistics.totalVolume)}
            </Typography>
          </Box>

          {/* 受影响切片数 / 总层数 */}
          <Box sx={{ p: 1, bgcolor: 'var(--bg-inset)', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.62rem', display: 'block' }}>
              受影响切片数
            </Typography>
            <Typography variant="subtitle1" sx={{ color: 'var(--accent)', mt: 0.25, lineHeight: 1.2, fontSize: '0.85rem', fontWeight: 600 }}>
              {statistics.tumorSlices} / {getVolumeDims(mriData)[3]}
            </Typography>
          </Box>

          {/* 总体素数量 */}
          <Box sx={{ p: 1, bgcolor: 'var(--bg-inset)', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.62rem', display: 'block' }}>
              总体素数量
            </Typography>
            <Typography variant="subtitle1" sx={{ color: 'var(--accent)', mt: 0.25, lineHeight: 1.2, fontSize: '0.85rem', fontWeight: 600 }}>
              {statistics.totalVoxels.toLocaleString()}
            </Typography>
          </Box>

          {/* 切片范围 */}
          <Box sx={{ p: 1, bgcolor: 'var(--bg-inset)', borderRadius: 1 }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.62rem', display: 'block' }}>
              切片范围
            </Typography>
            <Typography variant="subtitle1" sx={{ color: 'var(--accent)', mt: 0.25, lineHeight: 1.2, fontSize: '0.85rem', fontWeight: 600 }}>
              {statistics.sliceRange[0]} - {statistics.sliceRange[1]}
            </Typography>
          </Box>
        </Box>

        {/* 区域分布明细标题 */}
        <Typography variant="caption" sx={{ color: 'var(--text-primary)', fontWeight: 600, fontSize: '0.7rem', display: 'block', mb: 0.5 }}>
          区域分布明细
        </Typography>

        {/* 明细表格 */}
        <TableContainer>
          <Table size="small" padding="none">
            <TableHead>
              <TableRow>
                <TableCell sx={{ fontSize: '0.65rem', py: 0.5, color: 'var(--text-secondary)' }}>区域</TableCell>
                <TableCell align="right" sx={{ fontSize: '0.65rem', py: 0.5, color: 'var(--text-secondary)' }}>体素数</TableCell>
                <TableCell align="right" sx={{ fontSize: '0.65rem', py: 0.5, color: 'var(--text-secondary)' }}>体积</TableCell>
                <TableCell sx={{ fontSize: '0.65rem', py: 0.5, color: 'var(--text-secondary)' }}>占比</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {dataRows.map((row) => (
                <TableRow key={row.label}>
                  {/* 区域类型（带色块标识） */}
                  <TableCell sx={{ py: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Box
                        sx={{
                          width: 8,
                          height: 8,
                          borderRadius: '2px',
                          bgcolor: row.color,
                          flexShrink: 0
                        }}
                      />
                      <Typography sx={{ fontSize: '0.65rem', lineHeight: 1.2 }}>
                        {row.label}
                      </Typography>
                    </Box>
                  </TableCell>
                  {/* 体素数 */}
                  <TableCell align="right" sx={{ py: 0.5, fontSize: '0.65rem' }}>
                    {row.voxels.toLocaleString()}
                  </TableCell>
                  {/* 体积 */}
                  <TableCell align="right" sx={{ py: 0.5, fontSize: '0.65rem' }}>
                    {formatVolume(row.volume)}
                  </TableCell>
                  {/* 占比进度条 + 百分比 */}
                  <TableCell sx={{ py: 0.5 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <LinearProgress
                        variant="determinate"
                        value={percentOf(row.voxels)}
                        sx={{
                          flex: 1,
                          height: 4,
                          borderRadius: 1,
                          '& .MuiLinearProgress-bar': {
                            bgcolor: row.color
                          }
                        }}
                      />
                      <Typography sx={{ minWidth: 36, fontSize: '0.62rem', textAlign: 'right' }}>
                        {percentOf(row.voxels).toFixed(1)}%
                      </Typography>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Paper>
  );
}
