/**
 * ============================================================
 *  HistoryDetailPage —— 历史记录详情回放页（与主页同款排版）
 * ============================================================
 * 用户诉求：点历史记录后不是跳回主页、也不是只给一个小预览，
 * 而是像主页一样直接展示：
 *   - 三平面视图卡（桌面 2×2 / 移动 1 大 + 2 小，可交互翻片）
 *   - 显示调节卡（亮度/对比度/透明度 + 区域显隐）
 *   - 量化分析报告卡（完整表格）
 * 唯一区别：不含任何上传/推理按钮，顶部只有「返回历史」。
 *
 * 实现：从 IndexedDB 读取该记录保存的 mriDataGz + segDataGz，
 * 解析成 NiftiData 后写入全局 viewerStore（三视图/分析报告组件
 * 都是从 store 取数据，无需改动它们），页面本体只做布局。
 * 旧记录若只有 segDataGz、没有 mriDataGz：以分割图同时充当背景，
 * 保证仍可回放（注：此时亮度滑块的灰度变化不明显，会有提示）。
 */
import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box, Typography, Paper, IconButton, CircularProgress, Chip, Alert
} from '@mui/material';
import {
  ArrowBack as ArrowBackIcon,
  WarningAmber as WarningAmberIcon
} from '@mui/icons-material';
import MPRViewer from '../Viewer/MPRViewer';
import AnalysisReport from '../Reports/AnalysisReport';
import { getRecordStorage } from '../../services/storage/historyGateway';
import { parseNiftiFromGzData } from '../../utils/niftiParser';
import { useViewerStore } from '../../store/viewerStore';
import type { NiftiData } from '../../types';
import { LEGEND_ORDER } from '../../config/labels';
import { errMessage } from '../../utils/notify';

/**
 * 为没有保存原始 MRI 的旧记录合成一份"伪 MRI 灰度"
 *
 * 旧记录只有分割标签（0=背景 / 1=NCR / 2=ED / 3-4=ET），直接当 MRI 用
 * 会让所有像素都映射到接近 0 的灰度，最终黑底彩色遮罩，没有脑组织感。
 *
 * 这里按 BraTS 灰度分布把不同标签映射成由暗到亮的多档灰度，
 * 与默认窗位(窗位400/窗宽800)配合后会呈现：
 *   - 背景 200  → 灰度 64  (深灰 = 脑组织)
 *   - NCR 350   → 灰度 112 (中等)
 *   - ED  500   → 灰度 160 (较亮)
 *   - ET  650   → 灰度 208 (最亮 = 增强区)
 * 与彩色分割叠加后视觉上接近"灰阶 MRI + 分割"的效果。
 */
function synthesizeMriFromSeg(segData: NiftiData): NiftiData {
  const n = segData.typedArray.length;
  const arr = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const label = Math.round(segData.typedArray[i]);
    switch (label) {
      case 1: arr[i] = 350; break;
      case 2: arr[i] = 500; break;
      case 3: case 4: arr[i] = 650; break;
      default: arr[i] = 200;
    }
  }
  return { ...segData, typedArray: arr };
}

/** 每张卡片的圆角/边框（与主页卡片一致） */
const CARD_SX = {
  p: 1.25,
  mb: 1.25,
  bgcolor: 'var(--bg-panel)',
  borderRadius: 3,
  border: '1px solid var(--border)',
};

export default function HistoryDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  // 页面内轻量状态：加载中 / 错误 / 是否只读分割（无 MRI）
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ fileName: string; modelUsed: string; createdAt: string; segOnly: boolean } | null>(null);

  // 全局 store 的写入 action（三视图与报告组件从 store 读数据）
  const actions = useViewerStore((s) => s.actions);

  useEffect(() => {
    let cancelled = false;
    const recordId = Number(id);
    if (!Number.isFinite(recordId)) {
      setError('无效的历史记录');
      setLoading(false);
      return;
    }

    (async () => {
      try {
        setLoading(true);
        setError(null);
        // 统一网关：文件夹模式读真实文件，IndexedDB 模式读浏览器存储
        const { record: full, segGz, mriGz } = await getRecordStorage(recordId);
        if (cancelled) return;
        if (!full) {
          setError('未找到该历史记录（可能已被删除，或来自在线后端）');
          return;
        }

        // 必须要有分割文件才能回放
        if (!segGz) {
          setError('该记录未保存分割文件（在线记录或旧记录无法回放）');
          return;
        }

        const segData = await parseNiftiFromGzData(segGz);
        if (cancelled) return;

        let mriData: NiftiData;
        let segOnly = false;
        if (mriGz) {
          // 正常情况：有 MRI → 灰度底图 + 分割叠加 = 主页同等效果
          mriData = await parseNiftiFromGzData(mriGz);
          if (cancelled) return;
        } else {
          // 旧记录无 MRI：用分割图按 BraTS 灰度分布合成为"伪 MRI 灰度"
          // 这样三视图会有脑组织感（灰底），而不是纯黑+彩色 mask
          mriData = synthesizeMriFromSeg(segData);
          segOnly = true;
        }

        // 写入全局 store（布局组件自动取数）
        const { setMriData, setSegData, setMriFileInfo, setSegFileInfo, setSlicePosition, resetInference } = actions;
        resetInference();
        setMriData(mriData);
        setSegData(segData);
        setMriFileInfo({ name: full.fileName || 'history.mri.nii.gz', size: mriData.image.byteLength, status: 'ready' });
        setSegFileInfo({ name: 'segmentation_result.nii.gz', size: segData.image.byteLength, status: 'ready' });
        // 跳到三个方向的中间切片，方便直接看到病灶
        const d = mriData.header.dims;
        setSlicePosition({
          axial: Math.floor(((d[3] || 1) - 1) / 2),
          coronal: Math.floor(((d[2] || 1) - 1) / 2),
          sagittal: Math.floor(((d[1] || 1) - 1) / 2),
        });
        // 肿瘤统计交给下方 <AnalysisReport /> 计算（它内部对同一份 (segData, mriData)
        // 做了去重，避免这里再算一遍 900 万体素）

        setMeta({
          fileName: full.fileName || '(未命名)',
          modelUsed: full.modelUsed || full.modelType || '',
          createdAt: full.createdAt,
          segOnly,
        });
      } catch (e) {
        if (!cancelled) setError(errMessage(e, '解析失败，无法回放'));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();

    return () => { cancelled = true; };
  }, [id, actions]);

  /** 返回历史列表 */
  const goBack = () => navigate('/history');

  return (
    <Box sx={{ px: 1.5, pt: 0.5, pb: 2, width: '100%', maxWidth: 560, mx: 'auto' }}>
      {/* ── 顶部：返回 + 记录名（不放任何上传按钮） ── */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
        <IconButton size="small" onClick={goBack} sx={{ color: 'var(--text-secondary)' }}>
          <ArrowBackIcon fontSize="small" />
        </IconButton>
        <Box sx={{ flexGrow: 1, minWidth: 0 }}>
          <Typography
            variant="subtitle1"
            sx={{
              fontWeight: 700, lineHeight: 1.3,
              color: 'var(--text-primary)', fontSize: '0.95rem',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}
          >
            {meta?.fileName || '历史分割回放'}
          </Typography>
          {meta && (
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{
                display: 'block',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                fontSize: '0.65rem',
              }}
            >
              {meta.modelUsed} · {new Date(meta.createdAt).toLocaleString('zh-CN')}
            </Typography>
          )}
        </Box>
        {meta?.segOnly && (
          <Chip label="仅分割" size="small" color="warning" variant="outlined" sx={{ flexShrink: 0, height: 20, fontSize: '0.62rem' }} />
        )}
      </Box>

      {/* ── 加载中 / 错误 ── */}
      {loading ? (
        <Paper elevation={0} sx={{ ...CARD_SX, textAlign: 'center', py: 6 }}>
          <CircularProgress size={24} sx={{ color: 'var(--accent)' }} />
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1.5 }}>
            正在读取分割结果并重建三平面视图…
          </Typography>
        </Paper>
      ) : error ? (
        <Paper elevation={0} sx={{ ...CARD_SX, textAlign: 'center', py: 6 }}>
          <Typography color="error" variant="body2">{error}</Typography>
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center', gap: 1 }}>
            <IconButton size="small" onClick={goBack} sx={{ color: 'var(--accent)' }}><ArrowBackIcon fontSize="small" /> 返回历史</IconButton>
          </Box>
        </Paper>
      ) : (
        <>
          {/* 旧记录无原始 MRI 时，顶部醒目提示 + 操作建议 */}
          {meta?.segOnly && (
            <Alert
              severity="warning"
              icon={<WarningAmberIcon fontSize="small" />}
              sx={{ mb: 1, fontSize: '0.7rem', py: 0.5, '& .MuiAlert-message': { width: '100%' } }}
            >
              旧记录未保存原始 MRI，下方三视图已用「伪 MRI 灰度」近似显示——
              如需查看完整 MRI + 分割叠加效果，请用同一文件重新运行推理。
            </Alert>
          )}

          {/* ── 卡1：三平面视图（与主页一致的紧凑排版） ── */}
          <Paper elevation={0} sx={{ p: 1, mb: 1, bgcolor: 'var(--bg-panel)', borderRadius: 3, border: '1px solid var(--border)' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
              <Typography variant="subtitle2" sx={{ color: 'var(--accent)', fontWeight: 700, fontSize: '0.78rem' }}>
                三平面视图
              </Typography>
              <Box sx={{ flexGrow: 1 }} />
              <Box
                sx={{
                  px: 1, py: 0.25, borderRadius: '999px', fontSize: '0.6rem', fontWeight: 600,
                  bgcolor: meta?.segOnly ? 'rgba(255,152,0,.15)' : 'rgba(76,175,80,.14)',
                  color: meta?.segOnly ? '#ff9800' : '#4caf50',
                }}
              >
                {meta?.segOnly ? '仅分割叠加' : 'MRI + 分割叠加'}
              </Box>
            </Box>
            <MPRViewer />
          </Paper>

          {/* 移动端：显示调节已并入三平面视图内部（折叠面板，避免重复显示） */}

          {/* ── 卡2：分割标签图例（与主页一致：无标题紧凑一行居中） ── */}
          <Paper
            elevation={0}
            sx={{
              p: 1, mb: 1, bgcolor: 'var(--bg-panel)',
              borderRadius: 3, border: '1px solid var(--border)',
            }}
          >
            <Box
              sx={{
                display: 'flex',
                flexWrap: 'wrap',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 1.25,
              }}
            >
{LEGEND_ORDER.map((item) => (
                <Box key={item.key} sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Box sx={{ width: 10, height: 10, borderRadius: '2px', bgcolor: item.hex, flexShrink: 0 }} />
                  <Typography sx={{ color: 'var(--text-primary)', fontSize: '0.65rem', lineHeight: 1.2 }}>
                    {item.label} ({item.short})
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>

          {/* ── 卡3：量化分析报告（与主页一致的紧凑字号） ── */}
          <Box sx={{
            /* 移动端压缩量化分析报告字号/间距，不影响桌面排版 */
            '& .MuiPaper-root .MuiTypography-h6': { fontSize: '0.78rem', fontWeight: 700 },
            '& .MuiPaper-root .MuiTypography-h5': { fontSize: '0.85rem' },
            '& .MuiPaper-root .MuiTypography-subtitle1': { fontSize: '0.78rem', fontWeight: 600 },
            '& .MuiPaper-root .MuiTypography-subtitle2': { fontSize: '0.72rem', fontWeight: 700 },
            '& .MuiPaper-root .MuiTypography-body1': { fontSize: '0.7rem' },
            '& .MuiPaper-root .MuiTypography-body2': { fontSize: '0.66rem' },
            '& .MuiPaper-root .MuiTypography-caption': { fontSize: '0.62rem' },
            '& .MuiPaper-root .MuiTableCell-root': { fontSize: '0.65rem', padding: '4px 6px' },
            '& .MuiPaper-root .MuiLinearProgress-root': { height: 4, borderRadius: 2 },
          }}>
            <AnalysisReport />
          </Box>
        </>
      )}
    </Box>
  );
}