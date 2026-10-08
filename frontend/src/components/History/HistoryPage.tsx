/**
 * ============================================================
 *  HistoryPage —— 推理历史记录页面
 * ============================================================
 * 职责：展示后端保存的推理历史记录列表，并支持删除单条记录。
 *
 * 功能点：
 *  1. 页面加载时拉取历史记录；
 *  2. "刷新"按钮重新拉取；
 *  3. 每条记录展示 ID、文件名、模型、状态、耗时、体积统计、时间；
 *  4. 删除按钮：调用后端删除接口成功后，从本地列表移除该条。
 *
 * 对应路由：/history。
 */
import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, Typography, Table, TableBody, TableCell, TableContainer,
  TableHead, TableRow, Paper, Chip, IconButton, Tooltip, CircularProgress, Button,
  Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions
} from '@mui/material';
import {
  Delete as DeleteIcon, Refresh as RefreshIcon, CheckCircle, Error as ErrorIcon,
  Download as DownloadIcon, FolderOpen as FolderOpenIcon, Storage as StorageIcon,
  Article as ArticleIcon, DriveFileRenameOutline as RenameIcon
} from '@mui/icons-material';
import {
  getHistoryRecords, deleteHistoryRecord, getRecordStorage,
  renameHistoryRecord,
  connectFolderStorage, disconnectFolderStorage, getFolderStorageStatus,
  folderStorageSupported,
} from '../../services/storage/historyGateway';
import { exportNiftiGzBlob, exportFullReportTs } from '../../utils/niftiWriter';
import { useViewerStore } from '../../store/viewerStore';
import { useModeStore } from '../../store/modeStore';
import { parseNiftiFromGzData } from '../../utils/niftiParser';
import { notifyError, errMessage} from '../../utils/notify';
import type { HistoryRecord } from '../../types';
import { useIsMobile } from '../../hooks/useIsMobile';
import type { ChangeEvent, KeyboardEvent} from 'react';

/** 历史记录卡片单行：标签 + 值（移动端卡片化排版） */
function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1, fontSize: '0.8rem', py: 0.25 }}>
      <Typography component="span" sx={{ width: 56, flexShrink: 0, color: 'var(--text-disabled)' }}>{label}</Typography>
      <Typography component="span" sx={{ color: 'var(--text-primary)', flex: 1, wordBreak: 'break-all' }}>{children}</Typography>
    </Box>
  );
}

/** 单条历史记录卡片（移动端）—— 点击整卡进入详情回放页（主页同款三视图排版） */
function HistoryCard({
  record,
  onDelete,
  onRename,
  onExport,
  onExportTs,
  onOpen,
  exporting,
}: {
  record: HistoryRecord;
  onDelete: (record: HistoryRecord) => void;
  onRename: (record: HistoryRecord) => void;
  onExport: (id: number, fileName: string) => void;
  onExportTs: (id: number, fileName: string) => void;
  onOpen: (id: number) => void;
  exporting: boolean;
}) {
  const isOk = record.status === 'COMPLETED';

  return (
    <Paper
      onClick={() => onOpen(record.id)}
      sx={{
        p: 1.5, mb: 1.5, bgcolor: 'var(--bg-panel)', border: '1px solid var(--border)',
        cursor: 'pointer',
        transition: 'border-color .15s',
        '&:hover': { borderColor: 'var(--accent)' },
        '&:active': { opacity: 0.85 },
      }}
    >
      {/* 文件名 + 状态 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
        <Typography
          sx={{
            flex: 1, fontSize: '0.85rem', fontWeight: 500, color: 'var(--text-primary)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
          }}
          title={record.fileName}
        >
          {record.fileName || '-'}
        </Typography>
        <Chip
          icon={isOk ? <CheckCircle /> : <ErrorIcon />}
          label={isOk ? '成功' : '失败'}
          size="small"
          color={isOk ? 'success' : 'error'}
          variant="outlined"
          sx={{ flexShrink: 0 }}
        />
      </Box>
      {/* 模型 + 耗时 */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5, flexWrap: 'wrap' }}>
        <Chip label={record.modelUsed || record.modelType} size="small" sx={{ bgcolor: '#1a3a4a', color: 'var(--accent)', fontSize: '0.7rem' }} />
        <Typography sx={{ fontSize: '0.75rem', color: 'var(--text-disabled)' }}>{formatTime(record.executionTimeMs)}</Typography>
      </Box>
      {/* 体积统计 */}
      <FieldRow label="体积">{formatVolume(record.volumes)}</FieldRow>
      {/* 时间 + 导出 + 删除 */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mt: 0.5 }}>
        <Typography sx={{ fontSize: '0.75rem', color: 'var(--accent)' }}>
          {isOk ? '点此查看三视图回放 →' : ''}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tooltip title="导出分割文件 (.nii.gz)">
            <IconButton
              size="small"
              disabled={exporting}
              onClick={(e) => { e.stopPropagation(); onExport(record.id, record.fileName); }}
              sx={{ color: 'var(--accent)' }}
            >
              <DownloadIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="导出完整报告 (.ts)">
            <IconButton
              size="small"
              disabled={exporting}
              onClick={(e) => { e.stopPropagation(); onExportTs(record.id, record.fileName); }}
              sx={{ color: 'var(--accent)' }}
            >
              <ArticleIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="重命名">
            <IconButton
              size="small"
              onClick={(e) => { e.stopPropagation(); onRename(record); }}
              sx={{ color: 'var(--text-disabled)', '&:hover': { color: 'var(--accent)' } }}
            >
              <RenameIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <IconButton
            size="small"
            onClick={(e) => { e.stopPropagation(); onDelete(record); }}
            sx={{ color: 'var(--text-disabled)', '&:hover': { color: '#f44336' } }}
          >
            <DeleteIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>
    </Paper>
  );
}

/**
 * 格式化耗时：小于 1 秒显示毫秒，否则显示秒
 */
function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * 格式化体积统计映射为可读字符串
 */
function formatVolume(volumes: Record<string, number> | null): string {
  if (!volumes) return '-';
  return Object.entries(volumes)
    .map(([k, v]) => `${k}: ${v.toFixed(1)}cm3`)
    .join('  ');
}

/**
 * 格式化 ISO 时间为本地中文时间字符串
 */
function formatDate(iso: string): string {
  return new Date(iso).toLocaleString('zh-CN');
}

/**
 * 推理历史记录页面组件
 */
export default function HistoryPage() {
  // 历史记录列表 / 加载中（错误统一走 notifyError 上报全局 store，不再本地保存）
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(true);
  // 屏幕断点：移动端用卡片列表替代表格，避免横向溢出
  const isMobile = useIsMobile();
  // 当前模式（用于提示用户「在线模式不支持重命名」）
  const mode = useModeStore((s) => s.mode);
  // 正在导出的记录 id（防重复点击）
  const [exportingId, setExportingId] = useState<number | null>(null);
  // 本地文件夹存储状态（active=已连接；name=目录名）
  const [folder, setFolder] = useState<{ active: boolean; name: string | null }>({ active: false, name: null });
  // 跳转历史详情回放页（/history/:id）
  const navigate = useNavigate();
  const openDetail = (id: number) => navigate(`/history/${id}`);

  /** 刷新文件夹存储状态 */
  const refreshFolder = useCallback(async () => {
    setFolder(await getFolderStorageStatus());
  }, []);

  // 挂载时查询存储状态
  useEffect(() => { refreshFolder(); }, [refreshFolder]);

  /** 点击"存储位置"：连接/断开本地文件夹 */
  const handleStorageClick = async () => {
    // 在线模式的历史一律由后端管理，文件夹开关在这里点了也不生效，直接挡掉
    if (mode === 'online') {
      notifyError('在线模式的历史记录由后端保存，本地文件夹存储仅在离线模式下可用');
      return;
    }
    if (folder.active) {
      // 断开：已写文件保留，之后新记录回到浏览器内置存储
      if (window.confirm('断开本地文件夹存储？已保存的文件不会删除；之后的记录改存浏览器内置存储。')) {
        await disconnectFolderStorage();
        await refreshFolder();
        await fetchHistory();
      }
      return;
    }
    if (!folderStorageSupported()) {
      alert('当前浏览器不支持"选择本地文件夹"（请使用电脑版 Chrome / Edge）。');
      return;
    }
    const res = await connectFolderStorage();
    if (res.active) {
      setFolder({ active: true, name: res.name });
      await fetchHistory();
    }
    // 用户取消选择则静默返回
  };

  /**
   * 导出该条历史记录的分割结果（文件夹模式读真实文件；IndexedDB 模式读压缩包）
   */
  const handleExport = async (id: number, fileName: string) => {
    if (exportingId !== null) return;
    setExportingId(id);
    try {
      const { record, segGz } = await getRecordStorage(id);
      if (!segGz) {
        alert('该记录没有保存分割文件（在线记录或旧记录，无法导出）');
        return;
      }
      const base = (record?.fileName || fileName || 'segmentation').replace(/\.(nii\.gz|nii)$/i, '');
      await exportNiftiGzBlob(segGz, base);
    } catch (e) {
      alert('导出失败: ' + errMessage(e, e));
    } finally {
      setExportingId(null);
    }
  };

  /**
   * 导出「完整报告 .ts」：解析 segGz（必要时 mriGz）→ 打包 metadata + 体数据。
   * 默认不打包 MRI 体数据（体积大），需要时再扩展 UI 选项。
   */
  const handleExportTs = async (id: number, fileName: string) => {
    if (exportingId !== null) return;
    setExportingId(id);
    try {
      const { record, segGz, mriGz } = await getRecordStorage(id);
      if (!segGz) {
        alert('该记录没有保存分割文件（无法导出完整报告）');
        return;
      }
      const segData = await parseNiftiFromGzData(segGz);
      // 可选 MRI：默认不打包以减小文件体积
      let mriData = null;
      if (mriGz) {
        try { mriData = await parseNiftiFromGzData(mriGz); } catch { /* 旧记录无 MRI 不致命 */ }
      }
      const base = (record?.fileName || fileName || 'segmentation_report')
        .replace(/\.(nii\.gz|nii|ts)$/i, '');
      const metadata = {
        id,
        recordId: record?.id ?? id,
        fileName: record?.fileName || fileName || null,
        modelType: record?.modelType || null,
        modelUsed: record?.modelUsed || null,
        modality: record?.volumes ? Object.keys(record.volumes) : [],
        inference: {
          executionTimeMs: record?.executionTimeMs || 0,
          status: record?.status || 'COMPLETED',
          error: record?.error || null,
        },
        createdAt: record?.createdAt || new Date().toISOString(),
        exportedAt: new Date().toISOString(),
        volumes: record?.volumes || {},
        viewSettingsSnapshot: useViewerStore.getState().viewSettings,
        // 顺手把当前用户调节的窗位窗宽、显隐写入，方便他人回放视图
        notes: '导出工具版本 v1；包含 metadata + 分割体数据；MRI 默认未打包',
      };
      await exportFullReportTs(
        { segData, mriData, metadata },
        { includeMri: false, baseName: base }
      );
    } catch (e) {
      alert('导出完整报告失败: ' + errMessage(e, e));
    } finally {
      setExportingId(null);
    }
  };

  /**
   * 拉取历史记录列表
   * 拉取期间显示加载态；成功写列表，失败经 notifyError 上报全局错误（顶栏 Banner 显示）。
   * 离线模式下不调用后端 API（M4 阶段会接入 IndexedDB 本地历史）。
   */
  const fetchHistory = useCallback(async () => {
    setLoading(true);
    try {
      // 模式判断已收敛到 storage/historyGateway：离线读 IndexedDB，在线读后端
      const data = await getHistoryRecords();
      setRecords(data);
    } catch (e) {
      // 统一错误上报：写入全局错误 store（顶栏 Banner 显示）
      notifyError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  // 页面挂载时拉取一次历史记录
  useEffect(() => { fetchHistory(); }, [fetchHistory]);

  /**
   * 删除单条历史记录（模式判断在网关层，组件无感知）
   * 仅当 Dialog 二次确认后才真正删除，避免误操作。
   */
  const [pendingDelete, setPendingDelete] = useState<HistoryRecord | null>(null);
  const [deleting, setDeleting] = useState(false);
  /** 触发删除二次确认 */
  const askDelete = (record: HistoryRecord) => setPendingDelete(record);
  /** 取消二次确认 */
  const cancelDelete = () => setPendingDelete(null);
  /** 二次确认后真正执行删除 */
  const confirmDelete = async () => {
    if (!pendingDelete) return;
    const id = pendingDelete.id;
    setDeleting(true);
    try {
      await deleteHistoryRecord(id);
      setRecords(prev => prev.filter(r => r.id !== id));
      setPendingDelete(null);
    } catch (e) {
      // 统一错误上报：写入全局错误 store（顶栏 Banner 显示）
      notifyError(e);
    } finally {
      setDeleting(false);
    }
  };

  /**
   * 重命名历史记录（保留 ID / 推理结果，仅改 fileName）。
   */
  const [pendingRename, setPendingRename] = useState<HistoryRecord | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [renaming, setRenaming] = useState(false);
  const [renameError, setRenameError] = useState<string | null>(null);
  /** 触发重命名 Dialog */
  const askRename = (record: HistoryRecord) => {
    setPendingRename(record);
    // 编辑框预填当前文件名（去掉常见 .nii/.nii.gz 后缀，方便用户只看主体名）
    const base = (record.fileName || '').replace(/\.(nii\.gz|nii|ts)$/i, '');
    setRenameValue(base);
    if (mode === 'online') {
      setRenameError('当前为在线模式：需要后端实现 PATCH /api/history/:id 才可重命名。');
    } else {
      setRenameError(null);
    }
  };
  /** 取消重命名 */
  const cancelRename = () => {
    if (renaming) return;
    setPendingRename(null);
    setRenameValue('');
    setRenameError(null);
  };
  /** 真正执行重命名 */
  const confirmRename = async () => {
    if (!pendingRename) return;
    const next = renameValue.trim();
    if (!next) {
      setRenameError('文件名不能为空');
      return;
    }
    if (next === pendingRename.fileName) {
      setPendingRename(null);
      setRenameValue('');
      setRenameError(null);
      return;
    }
    setRenaming(true);
    setRenameError(null);
    try {
      const res = await renameHistoryRecord(pendingRename.id, next);
      if (!res.ok) {
        console.warn('[rename] 失败:', res.reason);
        setRenameError(res.reason || '重命名失败');
        return;
      }
      setRecords(prev => prev.map(r => r.id === pendingRename.id ? { ...r, fileName: next } : r));
      setPendingRename(null);
      setRenameValue('');
    } catch (e) {
      console.error('[rename] 异常:', e);
      setRenameError(errMessage(e, '重命名失败'));
    } finally {
      setRenaming(false);
    }
  };

  return (
    <Box sx={{ p: isMobile ? 1.5 : 3, maxWidth: 1200, mx: 'auto' }}>
      {/* 页头 + 存储位置 + 刷新按钮 */}
      <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 1, mb: isMobile ? 1.5 : 3 }}>
        <Typography variant="h5" sx={{ flexGrow: 1, color: 'var(--accent)', fontSize: isMobile ? '1.2rem' : undefined, minWidth: 140 }}>
          推理历史记录
        </Typography>
        {/* 本地文件夹存储开关/状态（在线模式历史由后端管理，置灰并说明原因） */}
        <Tooltip title={
          mode === 'online'
            ? '在线模式的历史记录由后端保存；本地文件夹存储仅在离线模式下可用'
            : folder.active
              ? '数据存于所选文件夹（真实文件）。点击断开'
              : '点击选择本地文件夹存储（数据存为真实文件，可用任意 NIfTI 工具打开）'
        }>
          <Chip
            icon={mode === 'online' || !folder.active ? <StorageIcon fontSize="small" /> : <FolderOpenIcon fontSize="small" />}
            label={
              mode === 'online'
                ? '后端存储（在线）'
                : folder.active
                  ? `文件夹: ${folder.name ?? ''}`
                  : '浏览器内置存储'
            }
            size="small"
            onClick={handleStorageClick}
            clickable={mode !== 'online'}
            disabled={mode === 'online'}
            color={folder.active && mode !== 'online' ? 'primary' : 'default'}
            variant={folder.active && mode !== 'online' ? 'filled' : 'outlined'}
            sx={{
              flexShrink: 0,
              maxWidth: 220,
              '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis' },
            }}
          />
        </Tooltip>
        <Button
          variant="outlined"
          size="small"
          startIcon={<RefreshIcon />}
          onClick={fetchHistory}
          sx={{ color: 'var(--accent)', borderColor: 'var(--accent)' }}
        >
          刷新
        </Button>
      </Box>

      {/* 三种展示分支：加载中 / 空列表 / 记录列表（移动端卡片 / 桌面表格） */}
      {loading ? (
        /* 加载中：居中转圈 */
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
          <CircularProgress sx={{ color: 'var(--accent)' }} />
        </Box>
      ) : records.length === 0 ? (
        /* 空列表：提示无记录 */
        <Paper sx={{ p: 6, textAlign: 'center', bgcolor: 'var(--bg-panel)' }}>
          <Typography sx={{ color: 'var(--text-secondary)' }}>暂无推理历史记录</Typography>
        </Paper>
      ) : isMobile ? (
        /* 移动端：卡片列表（避免表格横向溢出） */
        <Box>
          {records.map((r) => (
            <HistoryCard
              key={r.id}
              record={r}
              onDelete={askDelete}
              onRename={askRename}
              onExport={handleExport}
              onExportTs={handleExportTs}
              onOpen={openDetail}
              exporting={exportingId === r.id}
            />
          ))}
        </Box>
      ) : (
        /* 桌面端：表格 */
        <TableContainer component={Paper} sx={{ bgcolor: 'var(--bg-panel)' }}>
          <Table size="small">
            <TableHead>
              <TableRow sx={{ '& th': { color: 'var(--accent)', fontWeight: 'bold', borderBottom: '2px solid var(--border)' } }}>
                <TableCell>ID</TableCell>
                <TableCell>文件名</TableCell>
                <TableCell>模型</TableCell>
                <TableCell>状态</TableCell>
                <TableCell>耗时</TableCell>
                <TableCell>体积统计</TableCell>
                <TableCell>时间</TableCell>
                <TableCell align="center">导出</TableCell>
                <TableCell align="center">操作</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {records.map((r) => (
                <TableRow
                  key={r.id}
                  hover
                  onClick={() => openDetail(r.id)}
                  sx={{
                    cursor: 'pointer',
                    '&:hover': { bgcolor: 'var(--bg-hover)' },
                    '& td': { color: 'var(--text-primary)', borderBottom: '1px solid var(--border)' },
                  }}
                >
                  {/* ID */}
                  <TableCell>{r.id}</TableCell>
                  {/* 文件名（超长省略） */}
                  <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {r.fileName || '-'}
                  </TableCell>
                  {/* 模型名 Chip */}
                  <TableCell>
                    <Chip label={r.modelUsed || r.modelType} size="small" sx={{ bgcolor: '#1a3a4a', color: 'var(--accent)', fontSize: '0.75rem' }} />
                  </TableCell>
                  {/* 状态 Chip（成功/失败） */}
                  <TableCell>
                    <Chip
                      icon={r.status === 'COMPLETED' ? <CheckCircle /> : <ErrorIcon />}
                      label={r.status === 'COMPLETED' ? '成功' : '失败'}
                      size="small"
                      color={r.status === 'COMPLETED' ? 'success' : 'error'}
                      variant="outlined"
                    />
                  </TableCell>
                  {/* 耗时 */}
                  <TableCell>{formatTime(r.executionTimeMs)}</TableCell>
                  {/* 体积统计（超长省略） */}
                  <TableCell sx={{ fontSize: '0.8rem', maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {formatVolume(r.volumes)}
                  </TableCell>
                  {/* 创建时间 */}
                  <TableCell sx={{ fontSize: '0.8rem', whiteSpace: 'nowrap' }}>{formatDate(r.createdAt)}</TableCell>
                  {/* 导出分割文件 */}
                  <TableCell align="center">
                    <Box sx={{ display: 'inline-flex', gap: 0.25 }}>
                      <Tooltip title="导出分割文件 (.nii.gz)">
                        <IconButton
                          size="small"
                          disabled={exportingId === r.id}
                          onClick={() => handleExport(r.id, r.fileName)}
                          sx={{ color: 'var(--accent)' }}
                        >
                          <DownloadIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="导出完整报告 (.ts)">
                        <IconButton
                          size="small"
                          disabled={exportingId === r.id}
                          onClick={() => handleExportTs(r.id, r.fileName)}
                          sx={{ color: 'var(--accent)' }}
                        >
                          <ArticleIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                  {/* 删除按钮 */}
                  <TableCell align="center">
                    <Box sx={{ display: 'inline-flex', gap: 0.25 }}>
                      <Tooltip title="重命名">
                        <IconButton size="small" onClick={() => askRename(r)} sx={{ color: 'var(--text-disabled)', '&:hover': { color: 'var(--accent)' } }}>
                          <RenameIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="删除">
                        <IconButton size="small" onClick={() => askDelete(r)} sx={{ color: 'var(--text-disabled)', '&:hover': { color: '#f44336' } }}>
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}

      {/* 记录总数 */}
      <Typography variant="body2" sx={{ mt: 2, color: 'var(--text-disabled)' }}>
        共 {records.length} 条记录
      </Typography>

      {/* 删除二次确认 Dialog（避免误操作） */}
      <Dialog
        open={!!pendingDelete}
        onClose={deleting ? undefined : cancelDelete}
        PaperProps={{ sx: { bgcolor: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 2 } }}
      >
        <DialogTitle sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 1, fontSize: '1rem' }}>
          <DeleteIcon fontSize="small" /> 删除历史记录？
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ color: 'var(--text-primary)', fontSize: '0.85rem' }}>
            即将删除以下记录，操作不可撤销：
            <Box
              sx={{
                mt: 1, p: 1, bgcolor: 'var(--bg-inset)', borderRadius: 1,
                fontFamily: 'monospace', fontSize: '0.78rem', color: 'var(--text-secondary)',
                wordBreak: 'break-all',
              }}
            >
              #{pendingDelete?.id} · {pendingDelete?.fileName || '(未命名)'}
              <br />
              模型：{pendingDelete?.modelUsed || pendingDelete?.modelType || '-'}
              <br />
              时间：{pendingDelete ? formatDate(pendingDelete.createdAt) : '-'}
            </Box>
          </DialogContentText>
        </DialogContent>
        <DialogActions sx={{ px: 2, pb: 1.5 }}>
          <Button onClick={cancelDelete} disabled={deleting} sx={{ color: 'var(--text-secondary)' }}>
            取消
          </Button>
          <Button
            onClick={confirmDelete}
            disabled={deleting}
            variant="contained"
            color="error"
            sx={{ textTransform: 'none' }}
          >
            {deleting ? '删除中...' : '确认删除'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* 重命名 Dialog：仅修改文件名，不影响 ID / 推理结果 */}
      <Dialog
        open={!!pendingRename}
        onClose={renaming ? undefined : cancelRename}
        PaperProps={{ sx: { bgcolor: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: 2 } }}
      >
        <DialogTitle sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 1, fontSize: '1rem' }}>
          <RenameIcon fontSize="small" /> 重命名历史记录
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ color: 'var(--text-primary)', fontSize: '0.85rem' }}>
            即将修改文件名为：
          </DialogContentText>
          <Box
            component="input"
            value={renameValue}
            onChange={(e: ChangeEvent<HTMLInputElement>) => setRenameValue(e.target.value)}
            onKeyDown={(e: KeyboardEvent<HTMLInputElement>) => {
              if (e.key === 'Enter' && !renaming) confirmRename();
              if (e.key === 'Escape' && !renaming) cancelRename();
            }}
            autoFocus
            disabled={renaming}
            sx={{
              mt: 1, width: '100%', p: 1, fontSize: '0.9rem',
              bgcolor: 'var(--bg-inset)', border: '1px solid var(--border)',
              borderRadius: 1, color: 'var(--text-primary)',
              outline: 'none', fontFamily: 'monospace',
              '&:focus': { borderColor: 'var(--accent)' },
            }}
          />
          <DialogContentText sx={{ mt: 1, color: 'var(--text-disabled)', fontSize: '0.7rem' }}>
            原名：{pendingRename?.fileName || '(未命名)'}
            <br />
            说明：仅修改文件名（不含扩展名），不影响推理结果与 ID。
          </DialogContentText>
          {renameError && (
            <Box sx={{ mt: 1, p: 0.75, bgcolor: '#2a1a1a', borderLeft: '3px solid #f44336', borderRadius: 0.5, color: '#f44336', fontSize: '0.75rem' }}>
              {renameError}
            </Box>
          )}
        </DialogContent>
        <DialogActions sx={{ px: 2, pb: 1.5 }}>
          <Button onClick={cancelRename} disabled={renaming} sx={{ color: 'var(--text-secondary)' }}>
            取消
          </Button>
          <Button
            onClick={confirmRename}
            disabled={renaming || !renameValue.trim() || mode === 'online'}
            variant="contained"
            sx={{ bgcolor: 'var(--accent)', '&:hover': { bgcolor: 'var(--accent-hover)' }, textTransform: 'none' }}
          >
            {renaming ? '保存中...' : '保存'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
