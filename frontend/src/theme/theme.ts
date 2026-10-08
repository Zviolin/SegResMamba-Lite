/**
 * ============================================================
 *  theme —— MUI 亮色 / 深色两套主题
 * ============================================================
 * 职责：定义 MUI 组件的两套配色（light / dark）：
 *  - palette.mode 让 MUI 内置组件（Paper/Alert/Chip/Select 等）自动适配；
 *  - 自定义布局颜色（背景/边框/文字）统一走 index.css 的 CSS 变量，
 *    由 AppThemeProvider 设置 <html data-theme> 驱动，二者保持同步。
 *
 * 说明：主色保持项目品牌色 #00adb5；亮色模式下使用更深的 #008f9a
 * 以保障白底下的文字/图标对比度。
 */
import { createTheme, type Theme } from '@mui/material/styles';

/** 亮色主题：MUI 组件浅色外观 */
export const lightTheme: Theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#008f9a' },
    background: { default: '#f5f6f8', paper: '#ffffff' },
    text: { primary: '#212121', secondary: '#5f6368' }
  }
});

/** 深色主题：MUI 组件深色外观（与原设计一致） */
export const darkTheme: Theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#00adb5' },
    background: { default: '#121212', paper: '#1e1e1e' },
    text: { primary: '#e0e0e0', secondary: '#888888' }
  }
});
