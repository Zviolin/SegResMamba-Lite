/**
 * ============================================================
 *  App —— 应用根组件与页面布局
 * ============================================================
 * 职责：
 *  1. 定义应用顶层布局：顶部导航栏（AppBar）+ 内容路由区 + 底部页脚；
 *  2. 使用 react-router-dom 定义三条路由：主页 /、历史记录 /history、任务管理 /jobs；
 *  3. 【M3 响应式】根据屏幕宽度自动切换布局：
 *       - 桌面端（≥900px）：顶部导航按钮 + 主页"左-中-右"三栏布局；
 *       - 移动端（<900px）：底部导航 Tab + 主页"操作面板抽屉 + 全屏主视图"。
 *  4. 全局错误信息在顶部以红色条幅展示。
 */
import { useEffect, useRef, useState } from 'react';
import { Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import {
  Box, AppBar, Toolbar, Typography, IconButton, Tooltip, Button, Chip, Paper,
  Drawer, BottomNavigation, BottomNavigationAction, Fab,
  Accordion, AccordionSummary, AccordionDetails
} from '@mui/material';
import {
  DarkMode as DarkModeIcon,
  LightMode as LightModeIcon,
  Info as InfoIcon,
  Settings as SettingsIcon,
  Help as HelpIcon,
  Home as HomeIcon,
  History as HistoryIcon,
  Schedule as ScheduleIcon,
  Tune as TuneIcon,
  Close as CloseIcon,
  PlayArrow as PlayIcon,
  ExpandMore as ExpandMoreIcon
} from '@mui/icons-material';
import {
  Dialog, DialogTitle, DialogContent, DialogActions, Divider, List, ListItem, ListItemIcon, ListItemText
} from '@mui/material';
import FileUploader from './components/Controls/FileUploader';
import ControlPanel from './components/Controls/ControlPanel';
import Legend from './components/Controls/Legend';
import InferencePanel from './components/Controls/InferencePanel';
import MobileDisplayControls from './components/Controls/MobileDisplayControls';
import LabelFileZone from './components/Controls/LabelFileZone';
import MPRViewer from './components/Viewer/MPRViewer';
import AnalysisReport from './components/Reports/AnalysisReport';
import HistoryPage from './components/History/HistoryPage';
import HistoryDetailPage from './components/History/HistoryDetailPage';
import JobsPage from './components/Jobs/JobsPage';
import ModeSwitch from './components/Controls/ModeSwitch';
import ServerSettingsDialog from './components/Controls/ServerSettingsDialog';
import { useViewerStore } from './store/viewerStore';
import { useThemeStore } from './store/themeStore';
import { useModeStore } from './store/modeStore';
import { useIsMobile } from './hooks/useIsMobile';
import { checkBackendReachable } from './services/api/ping';
import { LEGEND_ORDER } from './config/labels';

/**
 * 顶部导航栏配置：路径 / 标签 / 图标（桌面端 AppBar 与移动端 BottomNav 共用）
 */
const NAV_ITEMS = [
  { path: '/', label: '主页', icon: <HomeIcon /> },
  { path: '/history', label: '历史记录', icon: <HistoryIcon /> },
  { path: '/jobs', label: '任务管理', icon: <ScheduleIcon /> },
];

/**
 * 桌面端主页（左-中-右三栏布局）
 *
 *  - 左栏（宽 280）：上传 / 推理 / 标签 / 图例，以及已加载数据的信息卡；
 *  - 中栏（弹性）：MPR 四宫格主视图；
 *  - 右栏（宽 320）：Accordion 折叠面板：显示调节（默认展开）+ 量化分析报告（默认折叠）。
 */
function DesktopMainPage() {
  // 读取 MRI 数据（用于数据信息卡）。
  // 逐字段订阅：全量订阅会让推理进度每跳动一次就重渲染整棵树
  const mriData = useViewerStore((s) => s.mriData);
  // 右栏两个折叠面板各自独立的开关状态：
  //   -「显示调节」默认 true（最常用，避免右栏太空）
  //   -「量化分析报告」默认 false（按需打开，不记忆）
  // 这里用 useState 完全受控，不依赖 Accordion 内部 defaultExpanded，
  // 以规避 dev server（HMR）+ 中文路径 watcher 失灵导致 UI 不更新的坑。
  const [showDisplayPanel, setShowDisplayPanel] = useState(true);
  const [showReportPanel, setShowReportPanel] = useState(false);

  return (
    <Box sx={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
      {/* 左侧边栏 */}
      <Box sx={{
        width: 280, flexShrink: 0, bgcolor: 'var(--bg-panel)',
        borderRight: '1px solid var(--border)', padding: '15px',
        overflowY: 'auto', position: 'sticky', top: 56,
        alignSelf: 'flex-start', maxHeight: 'calc(100vh - 56px)'
      }}>
        <FileUploader />
        <InferencePanel />
        <LabelFileZone />
        <Legend />
        {/* 已加载数据信息卡（仅当 MRI 加载成功时显示） */}
        {mriData && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'var(--bg-app)', borderRadius: 1, fontSize: '0.875rem' }}>
            <Typography variant="subtitle2" gutterBottom sx={{ color: 'var(--accent)' }}>
              数据信息
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {/* 数据维度（X × Y × Z） */}
              <Typography variant="body2" color="text.secondary">
                维度: {mriData.header.dims[1] || 0} x {mriData.header.dims[2] || 0} x {mriData.header.dims[3] || 0}
              </Typography>
              {/* 体素间距（X × Y，mm） */}
              <Typography variant="body2" color="text.secondary">
                间距: {(mriData.header.pixDims[1] || 0).toFixed(2)} x {(mriData.header.pixDims[2] || 0).toFixed(2)} mm
              </Typography>
              {/* 数据类型编码 */}
              <Typography variant="body2" color="text.secondary">
                数据类型: {mriData.header.datatypeCode}
              </Typography>
            </Box>
          </Box>
        )}
      </Box>

      {/* 中央主视图区域 */}
      <Box sx={{
        flex: 1, bgcolor: 'var(--bg-app)', display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        padding: '20px', overflow: 'auto'
      }}>
        <MPRViewer />
      </Box>

      {/* 右侧面板：两个面板默认折叠，点击展开，默认状态不记忆 */}
      <Box sx={{
        width: 320, flexShrink: 0, bgcolor: 'var(--bg-panel)',
        borderLeft: '1px solid var(--border)', padding: '15px',
        overflowY: 'auto', position: 'sticky', top: 56,
        alignSelf: 'flex-start', maxHeight: 'calc(100vh - 56px)'
      }}>
        {/* 显示调节：完全受控，默认展开。点击标题切换开合。 */}
        <Accordion
          expanded={showDisplayPanel}
          onChange={(_, isExp) => setShowDisplayPanel(isExp)}
          disableGutters
          elevation={0}
          square
          sx={{
            bgcolor: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: 1,
            mb: 1,
            '&:before': { display: 'none' },
            '& .MuiAccordionSummary-root': { minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TuneIcon fontSize="small" />
              <Typography sx={{ fontWeight: 600, color: 'var(--text-primary)' }}>显示调节</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <ControlPanel />
          </AccordionDetails>
        </Accordion>

        {/* 量化分析报告：完全受控，默认折叠。 */}
        <Accordion
          expanded={showReportPanel}
          onChange={(_, isExp) => setShowReportPanel(isExp)}
          disableGutters
          elevation={0}
          square
          sx={{
            bgcolor: 'transparent',
            border: '1px solid var(--border)',
            borderRadius: 1,
            mb: 1,
            '&:before': { display: 'none' },
            '& .MuiAccordionSummary-root': { minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{
                width: 8, height: 8, borderRadius: '50%', bgcolor: 'var(--accent)'
              }} />
              <Typography sx={{ fontWeight: 600, color: 'var(--text-primary)' }}>量化分析报告</Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <AnalysisReport />
          </AccordionDetails>
        </Accordion>
      </Box>
    </Box>
  );
}

/**
 * 移动端主页（M3 重构：模仿参考版手机排版）
 *
 * 布局（纵向卡片、页面直接展示，不再把功能全藏进抽屉）：
 *  - 顶部：状态摘要 + 「上传 / 推理」按钮；
 *  - 卡片1：三平面视图（1 大 + 2 小，MobileTriView）；
 *  - 卡片2：显示调节（亮度/对比度/透明度 + 区域显隐）—— 折叠卡片，默认展开；MRI 数据载入后自动折叠；
 *  - 卡片3：量化分析报告（完整表格）—— 折叠卡片，默认折叠，点击展开；
 *  - 底部抽屉只承载「上传 / 推理」流程（文件上传、标签、模型推理、图例）；
 *  - 右下悬浮按钮，随时可重新打开操作抽屉。
 */
function MobileMainPage() {
  const [drawerOpen, setDrawerOpen] = useState(false);
  // 移动端两个折叠面板的开合状态：
  //   - 显示调节：默认展开；当 MRI 首次加载成功时，自动折叠（让出三平面视图）。
  //   - 量化分析报告：默认折叠，按需展开。本次会话内不持久化（不写 localStorage）。
  const [showMobileDisplayPanel, setShowMobileDisplayPanel] = useState(true);
  const [showMobileReportPanel, setShowMobileReportPanel] = useState(false);
  // 跟踪"是否已经做过自动折叠"，避免 MRI 切换/反复加载时把用户手动展开状态再压回去。
  const hasAutoCollapsedRef = useRef(false);
  // 从 store 中拿到 MRI 数据；只要有值，就触发一次"自动折叠显示调节"。
  const mriData = useViewerStore((s) => s.mriData);
  useEffect(() => {
    if (mriData && !hasAutoCollapsedRef.current) {
      setShowMobileDisplayPanel(false);
      hasAutoCollapsedRef.current = true;
    }
  }, [mriData]);
  // 页面状态摘要（用于顶部状态行与图例角标）
  const segData = useViewerStore((s) => s.segData);
  const inferenceStatus = useViewerStore((s) => s.inference.status);

  const hasMri = !!mriData;
  const hasSeg = !!segData;
  const openDrawer = () => setDrawerOpen(true);

  // 当前状态文案 / 颜色
  const statusText = hasSeg ? '分割完成，可查看量化分析'
    : hasMri ? 'MRI 已加载，可上传分割标签或运行推理'
    : '先上传 MRI，即可查看三平面视图';
  const statusColor = hasSeg ? '#4caf50' : hasMri ? 'var(--accent)' : 'var(--text-secondary)';

  return (
    <>
      <Box sx={{ px: 1.5, pt: 0.5, pb: 2, width: '100%', maxWidth: 560, mx: 'auto' }}>
        {/* ── 顶部：状态摘要 + 打开操作面板 ── */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
          <Box sx={{ flexGrow: 1, minWidth: 0 }}>
            <Typography
              variant="subtitle1"
              sx={{ fontWeight: 700, lineHeight: 1.3, color: 'var(--text-primary)', fontSize: '0.95rem' }}
            >
              脑肿瘤分割工作台
            </Typography>
            <Typography
              variant="caption"
              sx={{
                display: 'block', color: statusColor, fontWeight: 500,
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}
            >
              {statusText}
            </Typography>
          </Box>
          <Button
            variant="contained"
            size="small"
            startIcon={<TuneIcon />}
            onClick={openDrawer}
            sx={{
              flexShrink: 0,
              bgcolor: 'var(--accent)',
              '&:hover': { bgcolor: 'var(--accent-hover)' },
              textTransform: 'none',
              borderRadius: '20px',
              px: 2,
            }}
          >
            上传 / 推理
          </Button>
        </Box>

        {/* ── 卡片1：三平面视图（1 大 + 2 小） —— 压缩高度以容纳量化分析 ── */}
        <Paper
          elevation={0}
          sx={{
            p: 1, mb: 1, bgcolor: 'var(--bg-panel)',
            borderRadius: 3, border: '1px solid var(--border)',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mb: 0.5 }}>
            <Typography variant="subtitle2" sx={{ color: 'var(--accent)', fontWeight: 700, fontSize: '0.78rem' }}>
              三平面视图
            </Typography>
            <Box sx={{ flexGrow: 1 }} />
            <Box
              sx={{
                px: 1, py: 0.25, borderRadius: '999px', fontSize: '0.6rem', fontWeight: 600,
                bgcolor: hasSeg ? 'rgba(76,175,80,.14)' : 'var(--accent-soft)',
                color: hasSeg ? '#4caf50' : 'var(--accent)',
              }}
            >
              {hasSeg ? '分割叠加' : hasMri ? '原始图像' : '未加载'}
            </Box>
          </Box>
          <MPRViewer />
        </Paper>

        {/* ── 卡片2：显示调节（折叠，默认折叠，避免空白页 + 按需展开） ── */}
        <Accordion
          expanded={showMobileDisplayPanel}
          onChange={(_, isExp) => setShowMobileDisplayPanel(isExp)}
          disableGutters
          elevation={0}
          square
          sx={{
            bgcolor: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: 3,
            mb: 1,
            '&:before': { display: 'none' },
            '& .MuiAccordionSummary-root': { minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } },
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <TuneIcon fontSize="small" />
              <Typography sx={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.8rem' }}>
                显示调节
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <MobileDisplayControls />
          </AccordionDetails>
        </Accordion>

        {/* ── 卡片2：分割标签图例（紧贴三视图下方，无标题紧凑一行展示） ── */}
        <Paper
          elevation={0}
          sx={{
            p: 1, mb: 1, bgcolor: 'var(--bg-panel)',
            borderRadius: 3, border: '1px solid var(--border)',
          }}
        >
          {/* 三个色块横向居中排列，超窄屏自动换行；字号缩小到 0.65rem */}
          <Box
            sx={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 1.25,
            }}
          >
            {/* 颜色与名称统一取自 config/labels，避免与视图叠加 / 导出配色不一致 */}
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

        {/* ── 卡片3：量化分析报告（默认折叠，与桌面保持同一交互） ── */}
        <Accordion
          expanded={showMobileReportPanel}
          onChange={(_, isExp) => setShowMobileReportPanel(isExp)}
          disableGutters
          elevation={0}
          square
          sx={{
            bgcolor: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: 3,
            mb: 1,
            '&:before': { display: 'none' },
            '& .MuiAccordionSummary-root': { minHeight: 40, '& .MuiAccordionSummary-content': { my: 0.5 } },
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
          }}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: 'var(--accent)' }} />
              <Typography sx={{ fontWeight: 700, color: 'var(--text-primary)', fontSize: '0.8rem' }}>
                量化分析报告
              </Typography>
            </Box>
          </AccordionSummary>
          <AccordionDetails sx={{ p: 0 }}>
            <AnalysisReport />
          </AccordionDetails>
        </Accordion>
      </Box>

      {/* 悬浮操作按钮：随时重新打开上传/推理抽屉 */}
      <Fab
        size="medium"
        aria-label="上传或推理"
        onClick={openDrawer}
        sx={{
          position: 'fixed',
          right: 16,
          bottom: 80,
          zIndex: 1050,
          bgcolor: 'var(--accent)',
          color: '#fff',
          '&:hover': { bgcolor: 'var(--accent-hover)' },
        }}
      >
        <TuneIcon />
      </Fab>

      {/* ── 底部抽屉：只承载 上传/推理 流程（文件/标签/模型/图例） ── */}
      <Drawer
        anchor="bottom"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        PaperProps={{ sx: { height: '82vh', bgcolor: 'var(--bg-app)', borderTopLeftRadius: 16, borderTopRightRadius: 16 } }}
      >
        <Box sx={{
          overflowY: 'auto',
          p: 1.5,
          pb: 4,
          '& .MuiPaper-root': { p: 1, mb: 1, '&:last-child': { mb: 0 } },
          '& .MuiPaper-root .MuiTypography-h6': { fontSize: '0.9rem', fontWeight: 600 },
          '& .MuiPaper-root .MuiTypography-subtitle1': { fontSize: '0.85rem' },
          '& .MuiPaper-root .MuiTypography-subtitle2': { fontSize: '0.8rem' },
          '& .MuiPaper-root .MuiTypography-body1': { fontSize: '0.8rem' },
          '& .MuiPaper-root .MuiTypography-body2': { fontSize: '0.75rem' },
          '& .MuiPaper-root .MuiTypography-caption': { fontSize: '0.7rem' },
          '& .MuiPaper-root .MuiButton-root': { fontSize: '0.75rem', py: 0.5 },
          '& .MuiPaper-root .MuiOutlinedInput-input': { fontSize: '0.8rem', padding: '6px 10px' },
          '& .MuiPaper-root .MuiInputLabel-root': { fontSize: '0.8rem' },
          '& .MuiPaper-root .MuiFormHelperText-root': { fontSize: '0.7rem' },
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="h6" sx={{ flexGrow: 1, color: 'var(--accent)' }}>
              操作面板
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
              上传 · 推理（亮度/分析已常驻主页）
            </Typography>
            <IconButton onClick={() => setDrawerOpen(false)}><CloseIcon /></IconButton>
          </Box>
          <FileUploader />
          <InferencePanel />
          <LabelFileZone />
          {/* 分割标签图例已常驻主页（"脑图例"卡），此处不再重复 */}
          {/* 推理状态小提示：完成后点按外部关闭，回主页查看分析 */}
          {inferenceStatus === 'completed' && (
            <Typography variant="caption" color="success.main" sx={{ display: 'block', textAlign: 'center', mt: 1 }}>
              ✓ 推理完成，请关闭面板查看三平面视图与量化分析
            </Typography>
          )}
        </Box>
      </Drawer>
    </>
  );
}

/**
 * 主页入口：按屏幕宽度选择桌面三栏 / 移动单列布局
 */
function MainPage() {
  const isMobile = useIsMobile();
  return isMobile ? <MobileMainPage /> : <DesktopMainPage />;
}

/**
 * 移动端底部导航
 * 用 position: fixed + bottom: 0 浮在视口底部（不被页面内容滚动出视野，
 * 始终可见可点）。内容区加 paddingBottom 留出空间避免被遮。
 */
function BottomNav() {
  const location = useLocation();
  const navigate = useNavigate();
  return (
    <BottomNavigation
      value={location.pathname}
      onChange={(_e, value) => navigate(value)}
      showLabels
      sx={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 1100,           // 略低于 Drawer(1200) 但高于普通内容
        bgcolor: 'var(--bg-panel)',
        borderTop: '1px solid var(--border)',
      }}
    >
      {NAV_ITEMS.map(item => (
        <BottomNavigationAction
          key={item.path}
          label={item.label}
          icon={item.icon}
          value={item.path}
          sx={{ '&.Mui-selected': { color: 'var(--accent)' } }}
        />
      ))}
    </BottomNavigation>
  );
}

/**
 * 应用根组件
 * 负责顶部导航、路由分发、全局错误条幅与移动端底部导航。
 */
function App() {
  // 读取全局错误（用于顶部错误条）。同理只订阅 error 一个字段
  const error = useViewerStore((s) => s.error);
  // 主题模式与切换（亮色 / 深色）
  const { theme, toggleTheme } = useThemeStore();
  // 屏幕断点：移动端使用底部导航、抽屉布局
  const isMobile = useIsMobile();
  // 服务器设置对话框 + 使用说明对话框
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  // 获取当前路径与路由跳转方法
  const location = useLocation();
  const navigate = useNavigate();
  // 启动时探活：当前是「在线」但后端不可达时，自动切回「离线」模式，
  // 避免控制台一直 500 报错 + 用户体验顺畅（移动端默认离线）
  useEffect(() => {
    if (useModeStore.getState().mode !== 'online') return;
    // 卸载守卫：cleanup 后不再切换模式（避免卸载后 setState）
    let cancelled = false;
    // 1.5s 超时由 checkBackendReachable 内部处理（AbortController + setTimeout）
    checkBackendReachable(1500)
      .then((ok) => {
        // 探活失败（网络错误 / 非 2xx / 超时）→ 自动切回离线；组件已卸载则忽略
        if (!ok && !cancelled) {
          console.warn('[App] 在线后端不可达，已自动切换到离线模式');
          useModeStore.getState().setMode('offline');
        }
      })
      .catch(() => { /* checkBackendReachable 内部已兜底，不会 reject */ });
    // 卸载清理：置 cancelled，阻止卸载后的模式切换
    return () => { cancelled = true; };
  }, []);

  return (
    <Box sx={{
      minHeight: '100vh', bgcolor: 'var(--bg-app)',
      color: 'var(--text-primary)', display: 'flex', flexDirection: 'column',
      /* 移动端 fixed BottomNav 占约 56px；为内容留出 padding 避免被遮 */
      pb: isMobile ? '64px' : 0,
    }}>
      {/* 顶部导航栏 */}
      <AppBar position="sticky" sx={{ bgcolor: 'var(--bg-panel)', flexShrink: 0 }}>
        <Toolbar sx={{ gap: 0.5, px: isMobile ? 1 : 2 }}>
          {/* 系统标题（移动端缩小 + 不截断，确保"脑肿瘤分割可视化系统"完整显示） */}
          <Typography
            variant="h6"
            component="div"
            sx={{
              flexGrow: 0, color: 'var(--accent)',
              mr: isMobile ? 0.5 : 3,
              fontSize: isMobile ? '0.85rem' : undefined,
              whiteSpace: 'nowrap',
              fontWeight: 600,
            }}
          >
            脑肿瘤分割可视化系统
          </Typography>

          {/* 桌面端导航按钮（移动端隐藏，移入底部导航） */}
          {!isMobile && (
            <Box sx={{ flexGrow: 1, display: 'flex', gap: 0.5 }}>
              {NAV_ITEMS.map(item => (
                <Button
                  key={item.path}
                  startIcon={item.icon}
                  onClick={() => navigate(item.path)}
                  sx={{
                    color: location.pathname === item.path ? 'var(--accent)' : 'var(--text-secondary)',
                    borderBottom: location.pathname === item.path ? '2px solid var(--accent)' : '2px solid transparent',
                    borderRadius: 0,
                    textTransform: 'none',
                    px: 2,
                    '&:hover': { color: 'var(--accent)', bgcolor: 'rgba(0,173,181,0.08)' }
                  }}
                >
                  {item.label}
                </Button>
              ))}
            </Box>
          )}
          {isMobile && <Box sx={{ flexGrow: 1 }} />}

          {/* 亮色 / 深色主题切换（移动端只图标，桌面端显示文字） */}
          {isMobile ? (
            <Tooltip title={theme === 'dark' ? '切换亮色' : '切换深色'}>
              <IconButton color="inherit" size="small" onClick={toggleTheme} sx={{ mr: 0.5 }}>
                {theme === 'dark' ? <LightModeIcon fontSize="small" /> : <DarkModeIcon fontSize="small" />}
              </IconButton>
            </Tooltip>
          ) : (
            <Chip
              icon={theme === 'dark' ? <LightModeIcon /> : <DarkModeIcon />}
              label={theme === 'dark' ? '切换亮色' : '切换深色'}
              onClick={toggleTheme}
              clickable
              color="primary"
              variant="outlined"
              size="small"
              sx={{ mr: 1 }}
            />
          )}

          {/* 使用说明（始终显示，移动端用户引导核心） */}
          <Tooltip title="使用说明">
            <IconButton color="inherit" size={isMobile ? 'small' : 'medium'} onClick={() => setHelpOpen(true)}>
              <HelpIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          {/* 服务器设置（移动端也显示，小尺寸图标） */}
          <Tooltip title="服务器设置">
            <IconButton color="inherit" size={isMobile ? 'small' : 'medium'} onClick={() => setSettingsOpen(true)}>
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>

          {/* 项目信息（占位） */}
          {!isMobile && (
            <Tooltip title="项目信息">
              <IconButton color="inherit"><InfoIcon /></IconButton>
            </Tooltip>
          )}

          {/* 在线 / 离线模式切换开关 */}
          <ModeSwitch />
        </Toolbar>
      </AppBar>

      {/* 全局错误条幅（有错误时显示） */}
      {error && (
        <Box sx={{ p: 2, bgcolor: 'var(--bg-disabled)', borderBottom: '1px solid var(--border)', borderLeft: '4px solid #f44336' }}>
          <Typography color="error">错误: {error}</Typography>
        </Box>
      )}

      {/* 路由分发：主页 / 历史记录 / 任务管理 */}
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/history/:id" element={<HistoryDetailPage />} />
        <Route path="/jobs" element={<JobsPage />} />
      </Routes>

      {/* 移动端底部导航（M3 新增） */}
      {isMobile && <BottomNav />}

      {/* 服务器设置对话框（M5：在线模式后端地址配置） */}
      <ServerSettingsDialog open={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* 使用说明对话框（移动端 / 新用户引导） */}
      <Dialog open={helpOpen} onClose={() => setHelpOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle sx={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 1 }}>
          <HelpIcon fontSize="small" />
          使用说明
        </DialogTitle>
        <DialogContent dividers>
          <Typography variant="subtitle2" sx={{ mb: 1, color: 'var(--text-primary)' }}>三步开始：</Typography>
          <List dense>
            <ListItem>
              <ListItemIcon><TuneIcon fontSize="small" sx={{ color: 'var(--accent)' }} /></ListItemIcon>
              <ListItemText
                primary="① 切换到「离线」模式"
                secondary="点顶部右侧滑块（开关推左=离线）。离线时无需后端，模型内置在 App 中。"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon><HomeIcon fontSize="small" sx={{ color: 'var(--accent)' }} /></ListItemIcon>
              <ListItemText
                primary="② 上传 MRI 文件"
                secondary="点「操作面板」→ 「文件上传」选 .nii / .nii.gz。推荐上传 4 个模态（T1 原始/T1 增强/T2 加权/FLAIR）精度最佳。"
              />
            </ListItem>
            <ListItem>
              <ListItemIcon><PlayIcon fontSize="small" sx={{ color: 'var(--accent)' }} /></ListItemIcon>
              <ListItemText
                primary="③ 点「运行推理」"
                secondary="几秒后三平面 viewer 会显示分割结果（彩色掩码）。滑轮/按钮可切切片。"
              />
            </ListItem>
          </List>
          <Divider sx={{ my: 1 }} />
          <Typography variant="subtitle2" sx={{ mb: 1, color: 'var(--text-primary)' }}>底部导航：</Typography>
          <List dense>
            <ListItem><ListItemText primary="主页 / 历史记录 / 任务管理" secondary="历史记录自动保存（离线模式存本地，在线模式读后端）" /></ListItem>
          </List>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setHelpOpen(false)} variant="contained" sx={{ bgcolor: 'var(--accent)', '&:hover': { bgcolor: 'var(--accent-hover)' } }}>
            知道了
          </Button>
        </DialogActions>
      </Dialog>

      {/* 底部页脚（移动端隐藏，节省空间） */}
      {!isMobile && (
        <Box sx={{
          py: 2, textAlign: 'center', borderTop: '1px solid var(--border)',
          color: 'var(--text-disabled)', fontSize: '0.875rem', bgcolor: 'var(--bg-panel)'
        }}>
          <Typography variant="body2">
            脑肿瘤分割可视化系统 v1.0 | 基于 SegResNet-Mamba 模型 | SRTP 项目成果展示
          </Typography>
        </Box>
      )}
    </Box>
  );
}

export default App;
