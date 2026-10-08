/**
 * ============================================================
 *  InferencePanel —— 模型推理控制面板组件
 * ============================================================
 * 职责：提供模型推理的完整操作界面：
 *  1. 模型下拉选择（从后端加载的模型列表）；
 *  2. MRI 模态下拉选择（根据所选模型的支持模态动态过滤）；
 *  3. "运行推理"按钮（条件禁用：需 MRI 就绪、已选模型、且非推理中）；
 *  4. 重置推理按钮；
 *  5. 推理进度条（上传/推理阶段的百分比）；
 *  6. 状态 Chip、错误提示与推理结果摘要。
 */
import { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  LinearProgress,
  Chip,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  Settings as BrainIcon,
  PlayArrow as PlayIcon,
  Refresh as RefreshIcon,
  Close as CloseIcon,
  Download as DownloadIcon
} from '@mui/icons-material';
import type { SelectChangeEvent } from '@mui/material/Select';
import { useInference } from '../../hooks/useInference';
import { useViewerStore } from '../../store/viewerStore';
import { useModeStore } from '../../store/modeStore';
import { exportSegDataToNiftiGz } from '../../utils/niftiWriter';
import { notifyError, errMessage} from '../../utils/notify';
import type { ChipProps } from '@mui/material/Chip';

/** 离线模式 4 模态定义（与 www 对齐） */
const OFFLINE_MODALITIES = [
  { key: 't1n', label: 'T1 原始' },
  { key: 't1c', label: 'T1 增强' },
  { key: 't2w', label: 'T2 加权' },
  { key: 't2f', label: 'FLAIR' },
];

/**
 * 模型推理控制面板组件
 */
export default function InferencePanel() {
  // 从 useInference 获取模型列表、推理状态、运行方法、可用标志
  const { modelList, inference, runModelInference, isReadyForInference } = useInference();
  // 读取 MRI 文件信息（用于显示已加载文件名）
  const mriFileInfo = useViewerStore((s) => s.mriFileInfo);
  // 导出分割结果状态（导出中标记）
  const [exporting, setExporting] = useState(false);
  // 取出模型/模态选择与重置 action
  const { setSelectedModel, setSelectedModality, resetInference } = useViewerStore((s) => s.actions);
  // 当前模式：离线时显示 4 模态文件槽
  const mode = useModeStore((s) => s.mode);
  // 离线多模态文件（键 t1n/t1c/t2w/t2f）
  const [modFiles, setModFiles] = useState<Record<string, File>>({});

  /** 模型下拉选择回调 */
  const handleModelChange = (event: SelectChangeEvent<string>) => {
    setSelectedModel(event.target.value);
  };

  /** 模态下拉选择回调 */
  const handleModalityChange = (event: SelectChangeEvent<string>) => {
    setSelectedModality(event.target.value);
  };

  /**
   * 运行推理按钮回调
   *
   * 文件源：viewerStore.mriFile（M3 重构后），由 FileUploader 在用户选择时写入；
   * 这样 InferencePanel 不再依赖 document.getElementById 跨组件读文件，组件解耦。
   * 离线模式同时携带 4 模态文件（如有），传给 runModelInference 启动完整推理流程。
   *
   * 若用户尚未上传文件，直接把原因写到全局错误提示，让用户知道为何「无反应」。
   */
  const handleRunInference = () => {
    const file = useViewerStore.getState().mriFile;
    if (!file) {
      const msg = '请先在「文件上传」中加载 MRI 文件 (.nii / .nii.gz) 再运行推理';
      // 统一错误上报：改走 notifyError 写入全局错误 store（顶栏 Banner 显示）
      notifyError(msg);
      return;
    }
    runModelInference(file, mode === 'offline' && Object.keys(modFiles).length > 0 ? modFiles : undefined);
  };

  /** 「运行推理」按钮被禁用的具体原因（用于 Tooltip 提示用户） */
  const disableReason = (() => {
    if (inference.status === 'uploading' || inference.status === 'processing') {
      return '推理进行中，请等待完成';
    }
    if (mriFileInfo.status !== 'ready') {
      return '请先在「文件上传」中加载 MRI 文件';
    }
    if (!inference.selectedModel) {
      return '暂无可用模型（请检查后端 / 离线模型配置）';
    }
    return '';
  })();

  /** 多模态文件槽选择回调 */
  const handleModFile = (modKey: string, file: File | undefined) => {
    setModFiles(prev => {
      const next = { ...prev };
      if (file) next[modKey] = file;
      else delete next[modKey];
      return next;
    });
  };

  /** 导出分割结果：内存 segData → .nii.gz 下载（方案B，可分享给他人加载） */
  const handleExport = async () => {
    const segData = useViewerStore.getState().segData;
    if (!segData) {
      alert('当前没有分割结果可导出');
      return;
    }
    setExporting(true);
    try {
      await exportSegDataToNiftiGz(segData, `segmentation_result`);
    } catch (e) {
      alert('导出失败: ' + errMessage(e, e));
    } finally {
      setExporting(false);
    }
  };

  /** 重置推理状态回调 */
  const handleReset = () => {
    resetInference();
  };

  /** 推理状态 -> 中文标签 */
  const getStatusLabel = () => {
    switch (inference.status) {
      case 'idle': return '等待推理';
      case 'uploading': return '上传中...';
      case 'processing': return '推理中...';
      case 'completed': return '✓ 推理完成';
      case 'error': return '✗ 推理失败';
      default: return '未知状态';
    }
  };

  /** 推理状态 -> Chip 颜色 */
  const getStatusColor = () => {
    switch (inference.status) {
      case 'idle': return 'default';
      case 'uploading': return 'warning';
      case 'processing': return 'warning';
      case 'completed': return 'success';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  // 根据当前选中模型查出其支持模态列表
  const selectedModelInfo = modelList.find(m => m.modelType === inference.selectedModel);

  return (
    <Paper elevation={3} sx={{ p: 2, mt: 2 }}>
      {/* 标题 */}
      <Typography
        variant="h6"
        gutterBottom
        sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 1 }}
      >
        <BrainIcon /> 模型推理
      </Typography>

      {/* 模型选择 */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          选择模型
        </Typography>
        <FormControl fullWidth size="small">
          <InputLabel>模型名称</InputLabel>
          <Select
            value={inference.selectedModel}
            onChange={handleModelChange}
            label="模型名称"
            disabled={inference.status === 'uploading' || inference.status === 'processing'}
          >
            {modelList.map((model) => (
              <MenuItem key={model.modelType} value={model.modelType}>
                <Box>
                  {/* 模型名 + 版本/描述 */}
                  <Typography variant="body1">{model.modelName}</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {model.modelVersion} - {model.description}
                  </Typography>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {/* MRI 模态选择 */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          MRI 模态
        </Typography>
        <FormControl fullWidth size="small">
          <InputLabel>模态类型</InputLabel>
          <Select
            value={inference.selectedModality}
            onChange={handleModalityChange}
            label="模态类型"
            disabled={inference.status === 'uploading' || inference.status === 'processing'}
          >
            {/* 优先展示所选模型支持模态；模型信息缺失时回退到默认四模态 */}
            {(selectedModelInfo?.supportedModalities || ['T1ce', 'FLAIR', 'T1', 'T2']).map((modality) => (
              <MenuItem key={modality} value={modality}>
                {modality}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Box>

      {/* 离线模式：BraTS 4 模态文件槽（对齐 www，提升单模态精度） */}
      {mode === 'offline' && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            离线多模态（选填，建议上传 4 个模态提升精度）
          </Typography>
          {OFFLINE_MODALITIES.map(({ key, label }) => (
            <Box key={key} sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
              <Typography variant="caption" sx={{ width: 64, flexShrink: 0, color: 'var(--text-secondary)' }}>
                {label}
              </Typography>
              <input
                accept=".nii,.nii.gz"
                style={{ display: 'none' }}
                id={`offline-mod-${key}`}
                type="file"
                onChange={(e) => handleModFile(key, e.target.files?.[0])}
              />
              <label htmlFor={`offline-mod-${key}`} style={{ flexGrow: 1, minWidth: 0 }}>
                <Button variant="outlined" component="span" fullWidth size="small" sx={{ justifyContent: 'flex-start', textTransform: 'none', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                  {modFiles[key]?.name || `选择 ${label} 文件`}
                </Button>
              </label>
              {modFiles[key] && (
                <Tooltip title="清除">
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => handleModFile(key, undefined)}
                    sx={{ p: 0.5 }}
                  >
                    <CloseIcon fontSize="small" />
                  </IconButton>
                </Tooltip>
              )}
            </Box>
          ))}
        </Box>
      )}

      {/* 运行 / 重置按钮 */}
      <Box sx={{ display: 'flex', gap: 1 }}>
        {/* 按钮被禁用时也用 Tooltip 提示原因，避免「点了没反应」的体验 */}
        <Tooltip title={disableReason || '开始运行模型推理'} arrow>
          <span style={{ flex: 1, display: 'flex' }}>
            <Button
              variant="contained"
              fullWidth
              startIcon={<PlayIcon />}
              onClick={handleRunInference}
              disabled={!isReadyForInference}
              sx={{
                bgcolor: 'var(--accent)',
                '&:hover': { bgcolor: 'var(--accent-hover)' },
                '&:disabled': { bgcolor: 'var(--bg-disabled)', color: 'var(--text-disabled)' }
              }}
            >
              运行推理
            </Button>
          </span>
        </Tooltip>

        {/* 重置按钮（仅图标） */}
        <Button
          variant="outlined"
          startIcon={<RefreshIcon />}
          onClick={handleReset}
          disabled={inference.status === 'uploading' || inference.status === 'processing'}
          sx={{ minWidth: 48 }}
        />
      </Box>

      {/* 上传/推理进度条 */}
      {inference.status === 'uploading' || inference.status === 'processing' ? (
        <Box sx={{ mt: 2 }}>
          <LinearProgress
            value={inference.progress}
            variant="determinate"
            sx={{
              height: 8,
              borderRadius: 1,
              bgcolor: 'var(--border)',
              '& .MuiLinearProgress-bar': { bgcolor: 'var(--accent)' }
            }}
          />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1, textAlign: 'right' }}>
            {inference.progress}%
          </Typography>
        </Box>
      ) : null}

      {/* 状态 Chip + 已加载文件名 */}
      <Box sx={{ mt: 2, display: 'flex', gap: 1, alignItems: 'center' }}>
        <Chip
          label={getStatusLabel()}
          color={getStatusColor() as ChipProps['color']}
          size="small"
        />
        {mriFileInfo.status === 'ready' && (
          <Typography variant="caption" color="text.secondary">
            已加载: {mriFileInfo.name}
          </Typography>
        )}
      </Box>

      {/* 推理错误提示 */}
      {inference.error && (
        <Alert severity="error" sx={{ mt: 2, fontSize: '0.875rem' }}>
          {inference.error}
        </Alert>
      )}

      {/* 推理结果摘要 */}
      {inference.result && (
        <Box sx={{ mt: 2, p: 2, bgcolor: 'var(--bg-inset)', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom sx={{ color: 'var(--accent)' }}>
            推理结果
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            {/* 使用模型 */}
            <Typography variant="body2" color="text.secondary">
              使用模型: {inference.result.modelUsed}
            </Typography>
            {/* 推理耗时 */}
            <Typography variant="body2" color="text.secondary">
              推理耗时: {(inference.result.executionTimeMs / 1000).toFixed(2)} 秒
            </Typography>
            {/* 体积统计 */}
            {inference.result.volumes && Object.keys(inference.result.volumes).length > 0 && (
              <Box sx={{ mt: 1 }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                  体积统计 (cm³):
                </Typography>
                {Object.entries(inference.result.volumes).map(([key, value]) => (
                  <Typography key={key} variant="body2" sx={{ color: 'var(--accent)' }}>
                    {key}: {value.toFixed(2)}
                  </Typography>
                ))}
              </Box>
            )}
            {/* 导出分割结果（方案B：下载 .nii.gz，他人可加载标签文件查看同一分割） */}
            <Button
              variant="outlined"
              size="small"
              startIcon={<DownloadIcon />}
              onClick={handleExport}
              disabled={exporting}
              sx={{ mt: 1, color: 'var(--accent)', borderColor: 'var(--accent)', textTransform: 'none' }}
            >
              {exporting ? '导出中...' : '导出分割结果 (.nii.gz)'}
            </Button>
          </Box>
        </Box>
      )}
    </Paper>
  );
}
