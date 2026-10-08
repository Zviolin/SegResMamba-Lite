/**
 * ============================================================
 *  ServerSettingsDialog —— 后端服务器地址设置（M5 打包支持）
 * ============================================================
 * 用途：手机 APK 中 localhost 指向手机自身，无法连接开发机后端。
 * 用户可在此输入开发机局域网 IP 或公网地址（如 http://192.168.1.5:8080），
 * 保存后所有在线请求自动切换（localStorage 持久化，刷新生效）。
 *
 * 留空 = 使用默认（开发环境 '/api' 走 Vite 代理）。
 */
import { useState, useEffect } from 'react';
import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  TextField, Button, Typography, Box
} from '@mui/material';
import { API_BASE_STORAGE_KEY, resolveApiBase } from '../../services/api/apiClient';
import { checkBackendReachable } from '../../services/api/ping';

/**
 * 校验并规范化后端地址（必须为 http / https，自动去尾斜杠）
 *
 * 为什么必须校验：helperText 教用户填「192.168.1.5:8080」这种 IP 形式，
 * 一旦漏写 http://，浏览器会把它当成相对路径请求当前站点 —— 表现为 404，
 * 而旧文案只报「获取模型列表失败」，用户根本想不到是地址格式写错了。
 */
type UrlCheckResult =
  | { kind: 'ok'; value: string }
  | { kind: 'error'; msg: string };

function normalizeBase(input: string): UrlCheckResult {
  let u: URL;
  try {
    u = new URL(input);
  } catch {
    return { kind: 'error', msg: '地址格式不正确，请写成 http://192.168.1.5:8080 这样的完整形式（别漏 http://）' };
  }
  if (!['http:', 'https:'].includes(u.protocol)) {
    return { kind: 'error', msg: `不支持 ${u.protocol.replace(':', '')} 协议，仅支持 http / https` };
  }
  return { kind: 'ok', value: `${u.origin}${u.pathname.replace(/\/+$/, '')}` };
}

/**
 * 服务器设置对话框
 * @param open   是否打开
 * @param onClose 关闭回调
 */
export default function ServerSettingsDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [value, setValue] = useState('');
  // 校验 / 探活反馈文案（null 表示无错误）
  const [error, setError] = useState<string | null>(null);
  // 探活进行中（按钮禁用，避免连点）
  const [probing, setProbing] = useState(false);

  // 打开时载入当前已保存的地址
  useEffect(() => {
    if (open) {
      setError(null);
      try {
        setValue(localStorage.getItem(API_BASE_STORAGE_KEY) || '');
      } catch {
        setValue('');
      }
    }
  }, [open]);

  /** 保存：先校验格式与时延探活，通过后写 localStorage（空值=清除，回退默认） */
  const handleSave = async () => {
    const trimmed = value.trim();

    // 留空：清除用户配置，回退到默认地址
    if (!trimmed) {
      try {
        localStorage.removeItem(API_BASE_STORAGE_KEY);
      } catch {
        // 忽略
      }
      onClose();
      return;
    }

    // 格式与协议校验
    const parsed = normalizeBase(trimmed);
    if (parsed.kind === 'error') {
      setError(parsed.msg);
      return;
    }

    // 先落地再探活：checkBackendReachable 内部按 resolveApiBase() 取地址
    try {
      localStorage.setItem(API_BASE_STORAGE_KEY, parsed.value);
    } catch {
      setError('无法写入本地存储（浏览器隐私模式？），配置未保存');
      return;
    }

    setProbing(true);
    setError(null);
    const reachable = await checkBackendReachable(2500);
    setProbing(false);

    if (!reachable) {
      // 地址已保存但连不通：明确列出排查方向，而不是让用户等到点推理才知道失败
      setError('地址已保存，但暂时连不上。请确认：后端已启动、手机与电脑在同一 WiFi、防火墙未拦截端口');
      return;
    }

    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle sx={{ color: 'var(--accent)' }}>服务器设置</DialogTitle>
      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          当前生效地址：<strong>{resolveApiBase() || '(默认)'}</strong>
        </Typography>
        <TextField
          autoFocus
          fullWidth
          size="small"
          label="后端服务器地址"
          placeholder="http://192.168.1.5:8080"
          value={value}
          error={!!error}
          onChange={(e) => {
            setValue(e.target.value);
            if (error) setError(null);
          }}
          helperText={error || '手机 App 中 localhost 指向手机自身，需填写开发机局域网 IP 或公网地址；留空使用默认配置'}
        />
        <Box sx={{ mt: 1.5, p: 1.5, bgcolor: 'var(--bg-inset)', borderRadius: 1 }}>
          <Typography variant="caption" color="text.secondary">
            提示：开发电脑与手机连接同一 WiFi，运行后端后查看电脑局域网 IP（如 192.168.x.x），
            填写「http://IP:8080」。修改后立即生效。
          </Typography>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>取消</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={probing}
          sx={{ bgcolor: 'var(--accent)', '&:hover': { bgcolor: 'var(--accent-hover)' } }}
        >
          {probing ? '正在连接…' : '保存'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}
