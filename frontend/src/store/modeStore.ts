/**
 * ============================================================
 *  modeStore —— 应用运行模式（在线 / 离线）
 * ============================================================
 * 职责：
 *  1. 保存当前模式（online / offline），持久化到 localStorage；
 *  2. 提供 setMode() 供顶部开关一键切换。
 *
 * 具体用哪个推理引擎由业务层（hooks/useInference）按当前 mode 直接选择
 * onlineProvider / offlineProvider，本 store 不再持有引擎引用。
 */
import { create } from 'zustand';

/** 应用模式：online=在线（后端推理），offline=离线（本地推理） */
export type AppMode = 'online' | 'offline';

/** localStorage 持久化键名（刷新页面后保持上次选择） */
const MODE_STORAGE_KEY = 'brainseg-mode';

/**
 * 读取初始模式
 * 优先取持久化值；localStorage 不可用（隐私模式等）时回退为「离线」模式。
 *
 * 设计：默认 offline —— 移动端/打包 APK 的主场景是离线使用（断网可用，
 * 模型内置），避免一打开就请求 /api/models 导致无后端时控制台一片 500 报错。
 * 用户切到「在线」后会被持久化，下次启动仍保持。
 */
function readInitialMode(): AppMode {
  try {
    const saved = localStorage.getItem(MODE_STORAGE_KEY);
    if (saved === 'online' || saved === 'offline') return saved;
  } catch {
    // localStorage 不可用时回退
  }
  return 'offline';
}

/** modeStore 状态结构 */
interface ModeState {
  /** 当前模式 */
  mode: AppMode;
  /** 切换模式（同时持久化） */
  setMode: (mode: AppMode) => void;
}

/** 全局模式 store（Zustand） */
export const useModeStore = create<ModeState>((set) => ({
  mode: readInitialMode(),
  setMode: (mode) => {
    try {
      localStorage.setItem(MODE_STORAGE_KEY, mode);
    } catch {
      // 持久化失败（如隐私模式）不影响内存态切换
    }
    set({ mode });
  }
}));
