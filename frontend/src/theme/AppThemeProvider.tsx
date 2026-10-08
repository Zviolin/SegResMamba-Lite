/**
 * ============================================================
 *  AppThemeProvider —— 应用主题 Provider
 * ============================================================
 * 职责：
 *  1. 读取 themeStore 当前主题，生成对应 MUI 主题并包裹子组件；
 *  2. 同步设置 <html data-theme="light|dark">，驱动 index.css 的
 *     CSS 变量（自定义 sx 配色随主题切换）；
 *  3. 注入 CssBaseline（MUI 全局样式重置，保证两套主题一致）。
 */
import { useMemo, useLayoutEffect, type ReactNode } from 'react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { useThemeStore } from '../store/themeStore';
import { lightTheme, darkTheme } from './theme';

/**
 * 应用主题 Provider
 * @param children 子组件树
 */
export default function AppThemeProvider({ children }: { children: ReactNode }) {
  const theme = useThemeStore((s) => s.theme);

  // 根据主题生成 MUI 主题（memo 避免无谓重建）
  const muiTheme = useMemo(() => (theme === 'dark' ? darkTheme : lightTheme), [theme]);

  // 同步 CSS 变量主题：渲染前先设置 data-theme，避免首次闪烁
  useLayoutEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return (
    <ThemeProvider theme={muiTheme}>
      <CssBaseline />
      {children}
    </ThemeProvider>
  );
}
