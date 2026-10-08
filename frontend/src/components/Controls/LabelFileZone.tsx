/**
 * ============================================================
 *  LabelFileZone —— 分割标签文件区组件
 * ============================================================
 * 职责：管理"分割标签文件"的展示与操作：
 *  1. 上传已有分割标签（.nii / .nii.gz），通过 loadSegFile 解析并写入 store；
 *  2. 展示当前标签文件信息（文件名、大小、状态）；
 *  3. 判断当前标签是否来自模型推理（文件名以 inference_mask 开头且推理已完成）；
 *  4. 提供"清除标签"按钮，一键清空分割数据、文件信息与肿瘤统计。
 */
import React, { useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Chip,
  Tooltip,
  IconButton
} from '@mui/material';
import {
  Label as LabelIcon,
  Description as FileIcon,
  UploadFile as UploadFileIcon,
  SmartToy as InferenceIcon,
  Clear as ClearIcon
} from '@mui/icons-material';
import { useNiftiLoader } from '../../hooks/useNiftiData';
import { useViewerStore } from '../../store/viewerStore';
import type { ChipProps } from '@mui/material/Chip';

/**
 * 标签文件区组件
 */
export default function LabelFileZone() {
  // 获取加载分割标签的方法
  const { loadSegFile } = useNiftiLoader();
  // 读取分割文件信息、推理状态与分割数据本身
  const segFileInfo = useViewerStore((s) => s.segFileInfo);
  const inference = useViewerStore((s) => s.inference);
  const segData = useViewerStore((s) => s.segData);
  // 取出清除操作需要的 action
  const { setSegData, setSegFileInfo, setStatistics } = useViewerStore((s) => s.actions);

  /**
   * 标签文件选择回调：取第一个文件并交给 loadSegFile 解析
   */
  const handleSegUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) loadSegFile(file);
  }, [loadSegFile]);

  // 判断当前标签文件是否来自模型推理
  /**
   * 判断当前标签文件是否来自模型推理
   * 条件：推理已完成，且文件名以 'inference_mask' 开头（推理产物命名约定）。
   */
  const isFromInference = useCallback(() => {
    return inference.status === 'completed' &&
      segFileInfo.name?.startsWith('inference_mask');
  }, [inference.status, segFileInfo.name]);

  /**
   * 清除当前标签：清空分割数据、文件信息与肿瘤统计
   */
  const handleClear = useCallback(() => {
    setSegData(null);
    setSegFileInfo({ name: '', size: 0, status: 'idle' });
    setStatistics(null);
  }, [setSegData, setSegFileInfo, setStatistics]);

  /**
   * 文件状态 -> Chip 颜色
   */
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'success';
      case 'loading': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  /**
   * 文件状态 -> 中文标签
   */
  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'ready': return '已加载';
      case 'loading': return '正在读取...';
      case 'error': return '加载失败';
      default: return '未加载';
    }
  };

  /**
   * 格式化文件大小
   */
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  // 是否存在标签：分割数据非空 或 文件状态为 ready
  const hasLabel = segData !== null || segFileInfo.status === 'ready';

  return (
    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
      {/* 标题 */}
      <Typography variant="h6" gutterBottom sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 1 }}>
        <LabelIcon /> 标签文件区
      </Typography>

      {/* 上传已有标签文件 */}
      <Box sx={{ mb: 1 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          上传已有分割标签 (.nii / .nii.gz)
        </Typography>
        {/* 隐藏文件选择框 + label 触发 */}
        <input
          accept=".nii,.nii.gz"
          style={{ display: 'none' }}
          id="seg-file-upload"
          type="file"
          onChange={handleSegUpload}
        />
        <label htmlFor="seg-file-upload" style={{ width: '100%' }}>
          <Button
            variant="outlined"
            component="span"
            fullWidth
            size="small"
            startIcon={<UploadFileIcon />}
            sx={{ justifyContent: 'flex-start', color: 'var(--text-muted)', borderColor: 'var(--border-strong)' }}
          >
            选择标签文件
          </Button>
        </label>
      </Box>

      {/* 当前标签文件状态 */}
      <Box sx={{
        p: 1,
        bgcolor: 'var(--bg-app)',
        borderRadius: 1,
        border: '1px solid var(--border)'
      }}>
        {/* 标题行 + 清除按钮 */}
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
          <Typography variant="caption" color="text.secondary">
            当前标签文件
          </Typography>
          {hasLabel && (
            <Tooltip title="清除标签">
              <IconButton size="small" onClick={handleClear} sx={{ color: 'var(--text-secondary)', p: 0.3 }}>
                <ClearIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          )}
        </Box>

        {/* 文件名与来源图标（推理产物为紫色，人工上传为青色） */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap' }}>
          <FileIcon sx={{ color: isFromInference() ? '#7c4dff' : 'var(--accent)', fontSize: 18 }} />
          <Typography
            variant="body2"
            sx={{
              color: 'var(--text-primary)',
              flex: 1,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap'
            }}
          >
            {segFileInfo.name || '未加载标签文件'}
          </Typography>
        </Box>

        {/* 状态 Chip + 文件大小 + 推理来源标记 */}
        <Box sx={{ mt: 1, display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip
            label={getStatusLabel(segFileInfo.status)}
            color={getStatusColor(segFileInfo.status) as ChipProps['color']}
            size="small"
            variant="outlined"
          />
          {segFileInfo.size > 0 && (
            <Typography variant="caption" color="text.secondary">
              {formatFileSize(segFileInfo.size)}
            </Typography>
          )}
          {/* 推理来源标签（紫色） */}
          {isFromInference() && (
            <Chip
              icon={<InferenceIcon sx={{ fontSize: 14 }} />}
              label="模型推理"
              size="small"
              sx={{ color: '#7c4dff', borderColor: '#7c4dff' }}
              variant="outlined"
            />
          )}
        </Box>
      </Box>
    </Paper>
  );
}
