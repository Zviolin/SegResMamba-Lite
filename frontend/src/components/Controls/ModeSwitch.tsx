/**
 * ============================================================
 *  ModeSwitch —— 在线 / 离线模式切换开关
 * ============================================================
 * 挂载于顶部导航栏（AppBar）：
 *  - 开关 ON  = 在线模式（推理走后端服务器，需网络）；
 *  - 开关 OFF = 离线模式（推理在本地设备完成，断网可用）。
 *
 * 切换行为：
 *  - 写 modeStore（自动持久化到 localStorage，刷新保持）；
 *  - 重置推理状态（避免旧模式的推理结果/进度残留）；
 *  - Snackbar 提示当前模式。
 * M1 阶段离线引擎为占位实现，切换后若执行推理会得到"开发中"提示。
 */
import { useState } from 'react';
import {
  Box, Typography, Switch, Tooltip, Snackbar, Alert
} from '@mui/material';
import WifiIcon from '@mui/icons-material/Wifi';
import OfflineBoltIcon from '@mui/icons-material/OfflineBolt';
import { useModeStore } from '../../store/modeStore';
import { useViewerStore } from '../../store/viewerStore';
import { useIsMobile } from '../../hooks/useIsMobile';
import { API_BASE_STORAGE_KEY } from '../../services/api/apiClient';
import ServerSettingsDialog from './ServerSettingsDialog';

/**
 * 模式切换开关组件
 */
export default function ModeSwitch() {
  const { mode, setMode } = useModeStore();
  // 切换模式时重置推理状态，避免旧模式残留
  const resetInference = useViewerStore((s) => s.actions.resetInference);
  // 推理进行中禁用切换：此刻换引擎会让旧流程与新流程并行向同一个 store 写进度，
  // 结果是「进度条乱跳 / 显示上一次的结果」。
  const inferenceStatus = useViewerStore((s) => s.inference.status);
  const busy = inferenceStatus === 'processing' || inferenceStatus === 'uploading';
  // 底部提示条文案（null 表示不显示）
  const [toast, setToast] = useState<string | null>(null);

  const isOnline = mode === 'online';
  // 移动端：紧凑模式（只显示 Switch 滑块，去掉图标和"在线/离线"两个文字）
  const isMobile = useIsMobile();
  // 服务器设置弹窗（切在线但未配置后端地址时引导）
  const [settingsOpen, setSettingsOpen] = useState(false);

  /** 开关切换回调 */
  const handleChange = (checked: boolean) => {
    // 推理进行中拒绝切换（UI 已禁用，这里是双保险）
    if (busy) return;
    const next = checked ? 'online' : 'offline';
    setMode(next);
    resetInference();
    // 切到「在线」且尚未配置后端地址 → 弹出服务器设置引导输入 IP
    if (checked) {
      let hasUrl = false;
      try {
        hasUrl = !!localStorage.getItem(API_BASE_STORAGE_KEY);
      } catch {
        // localStorage 不可用时按未配置处理
      }
      if (!hasUrl) {
        setSettingsOpen(true);
      }
    }
    setToast(
      next === 'online'
        ? '已切换到在线模式（推理走后端服务器）'
        : '已切换到离线模式（推理在本地完成）'
    );
  };

  return (
    <>
      <Tooltip title={
        busy
          ? '推理进行中，暂不可切换模式（完成后自动恢复）'
          : isOnline ? '在线模式：推理由后端服务器完成' : '离线模式：推理在本地设备完成'
      }>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mr: isMobile ? 0 : 2, flexShrink: 0 }}>
          {/* 桌面端：完整图标 + 文字 + 开关 + 图标 + 文字 */}
          {!isMobile && (
            <>
              <WifiIcon fontSize="small" sx={{ color: isOnline ? 'var(--accent)' : 'var(--text-disabled)' }} />
              <Typography variant="body2" sx={{ color: isOnline ? 'var(--accent)' : 'var(--text-secondary)' }}>
                在线
              </Typography>
            </>
          )}

          {/* 切换开关（移动端独占，桌面也保留） */}
          <Switch
            size="small"
            checked={isOnline}
            disabled={busy}
            onChange={(e) => handleChange(e.target.checked)}
            sx={{
              '& .MuiSwitch-switchBase.Mui-checked': { color: 'var(--accent)' },
              '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': { bgcolor: 'var(--accent)' }
            }}
          />

          {/* 移动端：在 Switch 旁显示当前模式文字（颜色高亮），让用户一眼看到状态 */}
          {isMobile && (
            <Typography
              variant="caption"
              sx={{
                fontWeight: 600,
                fontSize: '0.7rem',
                color: isOnline ? 'var(--accent)' : 'var(--text-primary)',
                minWidth: 28,
                textAlign: 'left',
              }}
            >
              {isOnline ? '在线' : '离线'}
            </Typography>
          )}

          {!isMobile && (
            <>
              <Typography variant="body2" sx={{ color: isOnline ? 'var(--text-secondary)' : 'var(--accent)' }}>
                离线
              </Typography>
              <OfflineBoltIcon fontSize="small" sx={{ color: isOnline ? 'var(--text-disabled)' : 'var(--accent)' }} />
            </>
          )}
        </Box>
      </Tooltip>

      {/* 切换提示 */}
      <Snackbar
        open={!!toast}
        autoHideDuration={2500}
        onClose={() => setToast(null)}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert severity="info" variant="filled" onClose={() => setToast(null)} sx={{ fontSize: '0.875rem' }}>
          {toast}
        </Alert>
      </Snackbar>

      {/* 切到在线且未配置后端地址时的引导弹窗（输入后端 IP） */}
      <ServerSettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />
    </>
  );
}
