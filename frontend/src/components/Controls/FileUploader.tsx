/**
 * ============================================================
 *  FileUploader —— 原始 MRI 文件上传组件
 * ============================================================
 * 职责：提供隐藏的 <input type="file"> + 自定义按钮，让用户选择 .nii / .nii.gz
 * 原始 MRI 文件；选择后调用 useNiftiLoader 的 loadMriFile 完成解析并写入全局 store。
 *
 * 展示内容：
 *   - 上传按钮（显示已选文件名或提示文字）；
 *   - 状态 Chip（等待上传 / 正在读取 / 已加载 / 加载失败）；
 *   - 文件大小（自动格式化）。
 */
import React, { useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Paper,
  Chip
} from '@mui/material';
import {
  CloudUpload as UploadIcon,
  Description as FileIcon
} from '@mui/icons-material';
import { useNiftiLoader } from '../../hooks/useNiftiData';
import { useViewerStore } from '../../store/viewerStore';
import type { ChipProps } from '@mui/material/Chip';

/**
 * MRI 文件上传组件
 */
export default function FileUploader() {
  // 获取加载 MRI 的方法
  const { loadMriFile } = useNiftiLoader();
  // 读取当前 MRI 文件信息（文件名、大小、状态）
  const mriFileInfo = useViewerStore((s) => s.mriFileInfo);

  /**
   * 文件选择回调：取第一个文件并交给 loadMriFile 解析
   *
   * 同时把 File 对象写入全局 viewerStore.mriFile，
   * 让 InferencePanel 不再依赖 document.getElementById 跨组件读文件。
   */
  const handleMriUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      loadMriFile(file);
      // 把 File 引用写到全局 store，仅内存存（不会持久化），供运行推理时取用
      useViewerStore.getState().actions.setMriFile(file);
    }
  }, [loadMriFile]);

  /**
   * 把文件状态映射为 Chip 的颜色
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
   * 把文件状态映射为中文标签
   */
  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'ready': return '✓ 已加载';
      case 'loading': return '正在读取...';
      case 'error': return '加载失败';
      default: return '等待上传...';
    }
  };

  /**
   * 格式化文件大小：B / KB / MB
   */
  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  return (
    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
      {/* 标题 */}
      <Typography variant="h6" gutterBottom sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 1 }}>
        <UploadIcon /> 文件上传
      </Typography>

      <Box>
        {/* 说明文字 */}
        <Typography variant="body2" color="text.secondary" gutterBottom>
          原始 MRI 图像 (.nii / .nii.gz)
        </Typography>
        {/* 隐藏的原生文件选择框，由下方 label 触发 */}
        <input
          accept=".nii,.nii.gz"
          style={{ display: 'none' }}
          id="mri-file-upload"
          type="file"
          onChange={handleMriUpload}
        />
        {/* 用 label 关联隐藏 input，点击按钮等价于选择文件 */}
        <label htmlFor="mri-file-upload" style={{ width: '100%' }}>
          <Button
            variant="outlined"
            component="span"
            fullWidth
            startIcon={<FileIcon />}
            sx={{ justifyContent: 'flex-start' }}
          >
            {/* 已选文件时显示文件名，否则显示提示 */}
            {mriFileInfo.name || '选择 MRI 文件'}
          </Button>
        </label>
        {/* 状态 Chip + 文件大小 */}
        <Box sx={{ mt: 1, display: 'flex', gap: 1, alignItems: 'center' }}>
          <Chip
            label={getStatusLabel(mriFileInfo.status)}
            color={getStatusColor(mriFileInfo.status) as ChipProps['color']}
            size="small"
          />
          {mriFileInfo.size > 0 && (
            <Typography variant="caption" color="text.secondary">
              {formatFileSize(mriFileInfo.size)}
            </Typography>
          )}
        </Box>
      </Box>
    </Paper>
  );
}
