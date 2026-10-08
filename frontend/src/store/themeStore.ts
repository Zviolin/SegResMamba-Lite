/**
 * ============================================================
 *  themeStore —— 界面主题（亮色 / 深色）
 * ============================================================
 * 职责：
 *  1. 保存当前主题（light / dark），持久化到 localStorage；
 *  2. 首次访问无记录时跟随系统偏好（prefers-color-scheme）；
 *  3. 提供 toggleTheme() 供顶部按钮一键切换。
 *
 * 主题机制（两层联动）：
 *  - MUI ThemeProvider（components/theme/AppThemeProvider）控制 MUI 组件配色；
 *  - index.css 中的 CSS 变量（--bg-* / --text-* / --accent）控制自定义 sx 配色。
 * 两层由本 store 的 theme 值同步驱动。
 */
import { create } from 'zustand';

/** 主题模式：dark=深色（默认，与原设计一致），light=亮色 */
export type ThemeMode = 'light' | 'dark';

/** localStorage 持久化键名（刷新页面后保持上次选择） */
const THEME_STORAGE_KEY = 'brainseg-theme';

/**
 * 读取初始主题
 * 优先级：持久化值 > 系统偏好 > 默认深色。
 */
function readInitialTheme(): ThemeMode {
  try {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'light' || saved === 'dark') return saved;
  } catch {
    // localStorage 不可用（隐私模式等）时走默认
  }
  try {
    // 无记录时跟随系统偏好（仅桌面浏览器有效，webview 缺失时回退深色）
    if (window.matchMedia?.('(prefers-color-scheme: light)').matches) {
      return 'light';
    }
  } catch {
    // 忽略
  }
  return 'dark';
}

/** themeStore 状态结构 */
interface ThemeState {
  /** 当前主题 */
  theme: ThemeMode;
  /** 设置主题（同时持久化） */
  setTheme: (theme: ThemeMode) => void;
  /** 一键切换（深色 ↔ 亮色） */
  toggleTheme: () => void;
}

/** 全局主题 store（Zustand） */
export const useThemeStore = create<ThemeState>((set, get) => ({
  theme: readInitialTheme(),
  setTheme: (theme) => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // 持久化失败不影响内存态切换
    }
    set({ theme });
  },
  toggleTheme: () => {
    const next = get().theme === 'dark' ? 'light' : 'dark';
    get().setTheme(next);
  }
}));
