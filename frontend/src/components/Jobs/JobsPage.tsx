/**
 * ============================================================
 *  JobsPage —— 异步推理任务管理页面
 * ============================================================
 * 职责：展示后端异步推理任务列表，并支持：
 *  1. 每 5 秒自动轮询刷新任务列表（组件卸载时清除定时器）；
 *  2. 手动"刷新"按钮；
 *  3. 点击"查看详情"弹窗展示单个任务的完整信息；
 *  4. 每个任务的实时进度条。
 *
 * 对应路由：/jobs。
 */
import { useState, useEffect, useCallback } from 'react';
import {
  Box, Typography, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Chip, IconButton, Tooltip, CircularProgress,
  Button, LinearProgress, Dialog, DialogTitle, DialogContent, DialogActions
} from '@mui/material';
import { Refresh as RefreshIcon, Visibility as ViewIcon, Schedule as ScheduleIcon } from '@mui/icons-material';
import { getJobService } from '../../services/jobs/jobService';
import { errMessage } from '../../utils/notify';
import type { JobStatus as JobStatusType } from '../../types';
import { useModeStore } from '../../store/modeStore';
import { useIsMobile } from '../../hooks/useIsMobile';

/**
 * 任务状态 -> 中文标签 / Chip 颜色的映射表
 */
const STATUS_MAP: Record<string, { label: string; color: 'default' | 'primary' | 'success' | 'error' | 'warning' | 'info' }> = {
  QUEUED: { label: '排队中', color: 'default' },
  RUNNING: { label: '运行中', color: 'info' },
  COMPLETED: { label: '已完成', color: 'success' },
  FAILED: { label: '失败', color: 'error' },
  TIMEOUT: { label: '超时', color: 'warning' },
};

/**
 * 格式化耗时：小于 1 秒显示毫秒，否则显示秒
 */
function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * 格式化 ISO 时间为本地中文时间字符串
 */
function formatDate(iso: string): string {
  return new Date(iso).toLocaleString('zh-CN');
}

/**
 * 格式化体积统计映射为可读字符串
 * 例如 { WT: 123.4, TC: 45.6 } -> "WT: 123.4cm3  TC: 45.6cm3"
 */
function formatVolume(volumes: Record<string, number> | null | undefined): string {
  if (!volumes) return '-';
  return Object.entries(volumes)
    .map(([k, v]) => `${k}: ${v.toFixed(1)}cm3`)
    .join('  ');
}

/** 单个任务卡片（移动端，避免表格横向溢出） */
function JobCard({
  job,
  onView,
}: {
  job: JobStatusType;
  onView: (id: string) => void;
}) {
  const st = STATUS_MAP[job.status] || { label: job.status, color: 'default' as const };
  return (
    <Paper sx={{ p: 1.5, mb: 1.5, bgcolor: 'var(--bg-panel)', border: '1px solid var(--border)' }}>
      {/* 任务 ID + 状态 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
        <Typography
          sx={{ flex: 1, fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-primary)', wordBreak: 'break-all' }}
          title={job.jobId}
        >
          {job.jobId}
        </Typography>
        <Chip label={st.label} size="small" color={st.color} variant="outlined" sx={{ flexShrink: 0 }} />
      </Box>
      {/* 进度条 + 时间 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        <LinearProgress
          variant="determinate"
          value={job.progress}
          sx={{ flex: 1, height: 6, borderRadius: 3, bgcolor: 'var(--border)', '& .MuiLinearProgress-bar': { bgcolor: 'var(--accent)' } }}
        />
        <Typography variant="caption" sx={{ minWidth: 32, color: 'var(--text-primary)' }}>{job.progress}%</Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography sx={{ fontSize: '0.75rem', color: 'var(--text-disabled)' }}>{formatDate(job.createdAt)}</Typography>
        <Tooltip title="查看详情">
          <IconButton size="small" onClick={() => onView(job.jobId)} sx={{ color: 'var(--accent)' }}>
            <ViewIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </Paper>
  );
}

/**
 * 异步推理任务列表页面组件
 */
export default function JobsPage() {
  // 当前模式：离线时不调用后端 API
  const mode = useModeStore((s) => s.mode);
  // 屏幕断点：移动端用卡片列表替代表格，避免横向溢出
  const isMobile = useIsMobile();
  // 任务列表 / 加载中 / 错误 / 详情弹窗状态
  const [jobs, setJobs] = useState<JobStatusType[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detailJob, setDetailJob] = useState<JobStatusType | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);

  /**
   * 拉取任务列表
   * 成功时写入列表并清空错误，失败时记录错误，无论成败都关闭加载态。
   * 离线模式下不调用后端 API（M4 阶段会接入本地任务队列）。
   */
  const fetchJobs = useCallback(async () => {
    // 数据访问走服务层，组件不再直接依赖 HTTP 接口
    const service = getJobService(mode);
    if (!service) {
      // 离线暂无本地任务队列：明确告知，而不是静默显示空列表
      setJobs([]);
      setError('离线模式暂不支持任务队列，切到在线模式查看');
      setLoading(false);
      return;
    }
    try {
      const data = await service.list();
      setJobs(data);
      setError(null);
    } catch (e) {
      setError(errMessage(e, '操作失败'));
    } finally {
      setLoading(false);
    }
  }, [mode]);

  /**
   * 挂载时立即拉取一次，并开启 5 秒轮询；
   * 组件卸载时清除定时器，避免内存泄漏。
   */
  useEffect(() => {
    fetchJobs();
    // 离线模式不轮询（M4 接入本地队列后再说）
    if (mode === 'offline') return;
    const timer = setInterval(fetchJobs, 5000);
    return () => clearInterval(timer);
  }, [fetchJobs, mode]);

  /**
   * 查看任务详情：拉取单任务最新状态并打开弹窗
   */
  const handleViewDetail = async (jobId: string) => {
    const service = getJobService();
    if (!service) {
      setError('离线模式暂不支持查看任务详情');
      return;
    }
    try {
      const job = await service.status(jobId);
      setDetailJob(job);
      setDetailOpen(true);
    } catch (e) {
      setError(errMessage(e, '操作失败'));
    }
  };

  return (
    <Box sx={{ p: isMobile ? 1.5 : 3, maxWidth: 1200, mx: 'auto' }}>
      {/* 页头 + 刷新按钮 */}
      <Box sx={{ display: 'flex', alignItems: 'center', mb: isMobile ? 1.5 : 3 }}>
        <Typography variant="h5" sx={{ flexGrow: 1, color: 'var(--accent)', fontSize: isMobile ? '1.2rem' : undefined }}>
          异步推理任务
        </Typography>
        <Button
          variant="outlined"
          size="small"
          startIcon={<RefreshIcon />}
          onClick={fetchJobs}
          sx={{ color: 'var(--accent)', borderColor: 'var(--accent)' }}
        >
          刷新
        </Button>
      </Box>

      {/* 错误提示条 */}
      {error && (
        <Paper sx={{ p: 2, mb: 2, bgcolor: '#2a1a1a', borderLeft: '4px solid #f44336' }}>
          <Typography color="error">{error}</Typography>
        </Paper>
      )}

      {/* 三种展示分支：加载中 / 空列表（离线提示）/ 任务表格 */}
      {loading ? (
        /* 加载中：居中转圈 */
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress sx={{ color: 'var(--accent)' }} />
        </Box>
      ) : mode === 'offline' ? (
        /* 离线模式：任务管理依赖后端队列，本地不可用（M4） */
        <Paper sx={{ p: 6, textAlign: 'center', bgcolor: 'var(--bg-panel)' }}>
          <ScheduleIcon sx={{ fontSize: 48, color: 'var(--border-strong)', mb: 1 }} />
          <Typography sx={{ color: 'var(--text-secondary)' }}>离线模式下任务管理不可用</Typography>
          <Typography variant="body2" sx={{ mt: 1, color: 'var(--text-disabled)' }}>
            异步任务依赖后端队列，请切换到在线模式或使用本地历史记录
          </Typography>
        </Paper>
      ) : jobs.length === 0 ? (
        /* 空列表：提示无任务 */
        <Paper sx={{ p: 6, textAlign: 'center', bgcolor: 'var(--bg-panel)' }}>
          <ScheduleIcon sx={{ fontSize: 48, color: 'var(--border-strong)', mb: 1 }} />
          <Typography sx={{ color: 'var(--text-secondary)' }}>暂无异步推理任务</Typography>
          {/* 如实说明：提交接口 apiClient.submitJob 已实现，但主页没有异步提交入口，
              此前"通过主页面提交推理时可选择异步模式"指向的是不存在的功能。 */}
          <Typography variant="body2" sx={{ mt: 1, color: 'var(--text-disabled)' }}>
            当前版本主页仅支持同步推理，异步提交入口尚未开放
          </Typography>
        </Paper>
      ) : isMobile ? (
        /* 移动端：任务卡片列表（避免表格横向溢出） */
        <Box>
          {jobs.map((job) => (
            <JobCard key={job.jobId} job={job} onView={handleViewDetail} />
          ))}
        </Box>
      ) : (
        /* 任务表格 */
        <TableContainer component={Paper} sx={{ bgcolor: 'var(--bg-panel)' }}>
          <Table size="small">
            <TableHead>
              <TableRow sx={{ '& th': { color: 'var(--accent)', fontWeight: 'bold', borderBottom: '2px solid var(--border)' } }}>
                <TableCell>任务ID</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>进度</TableCell>
                <TableCell>创建时间</TableCell>
                <TableCell align="center">操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {jobs.map((job) => {
                // 从映射表取状态展示；未知状态时直接显示原始状态码
                const st = STATUS_MAP[job.status] || { label: job.status, color: 'default' as const };
                return (
                  <TableRow key={job.jobId} sx={{ '&:hover': { bgcolor: 'var(--bg-hover)' }, '& td': { color: 'var(--text-primary)', borderBottom: '1px solid var(--border)' } }}>
                    {/* 任务 ID（等宽字体） */}
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>
                      {job.jobId}
                    </TableCell>
                    {/* 状态 Chip */}
                    <TableCell>
                      <Chip label={st.label} size="small" color={st.color} variant="outlined" />
                    </TableCell>
                    {/* 进度条 + 百分比 */}
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <LinearProgress
                          variant="determinate"
                          value={job.progress}
                          sx={{ flex: 1, height: 6, borderRadius: 3, bgcolor: 'var(--border)', '& .MuiLinearProgress-bar': { bgcolor: 'var(--accent)' } }}
                        />
                        <Typography variant="caption" sx={{ minWidth: 32, color: 'var(--text-primary)' }}>{job.progress}%</Typography>
                      </Box>
                    </TableCell>
                    {/* 创建时间 */}
                    <TableCell sx={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>{formatDate(job.createdAt)}</TableCell>
                    {/* 查看详情按钮 */}
                    <TableCell align="center">
                      <Tooltip title="查看详情">
                        <IconButton size="small" onClick={() => handleViewDetail(job.jobId)} sx={{ color: 'var(--accent)' }}>
                          <ViewIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* 任务详情弹窗 */}
      <Dialog open={detailOpen} onClose={() => setDetailOpen(false)} maxWidth="sm" fullWidth
        PaperProps={{ sx: { bgcolor: 'var(--bg-panel)', color: 'var(--text-primary)' } }}>
        <DialogTitle sx={{ color: 'var(--accent)' }}>任务详情</DialogTitle>
        <DialogContent>
          {detailJob && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5, pt: 1 }}>
              {/* 基本信息 */}
              <Typography><strong>任务ID:</strong> {detailJob.jobId}</Typography>
              <Typography><strong>状态:</strong> {STATUS_MAP[detailJob.status]?.label || detailJob.status}</Typography>
              <Typography><strong>进度:</strong> {detailJob.progress}%</Typography>
              <Typography><strong>创建时间:</strong> {formatDate(detailJob.createdAt)}</Typography>
              {/* 完成时间（若有） */}
              {detailJob.finishedAt && (
                <Typography><strong>完成时间:</strong> {formatDate(detailJob.finishedAt)}</Typography>
              )}
              {/* 结果信息（若有） */}
              {detailJob.result && (
                <>
                  <Typography><strong>耗时:</strong> {formatTime(detailJob.result.executionTimeMs)}</Typography>
                  <Typography><strong>使用模型:</strong> {detailJob.result.modelUsed}</Typography>
                  <Typography><strong>体积统计:</strong> {formatVolume(detailJob.result.volumes)}</Typography>
                </>
              )}
              {/* 错误信息（若有） */}
              {detailJob.error && (
                <Paper sx={{ p: 1.5, bgcolor: '#2a1a1a', borderLeft: '4px solid #f44336' }}>
                  <Typography color="error" variant="body2">{detailJob.error}</Typography>
                </Paper>
              )}
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailOpen(false)} sx={{ color: 'var(--accent)' }}>关闭</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
