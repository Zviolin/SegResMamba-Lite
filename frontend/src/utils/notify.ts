/**
 * ============================================================
 *  notify —— 统一错误上报工具
 * ============================================================
 * 各组件/钩子的错误统一经此入口写入全局错误 store，
 * 由 App 顶栏红色条幅展示，避免各页面自建错误流。
 */
import { useViewerStore } from '../store/viewerStore';

/**
 * 把 unknown 异常解析为可读文案（供 alert / 提示文案复用，避免各处写 e?.message 再配 any）
 */
export function errMessage(err: unknown, fallback = '操作失败，请重试'): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  return fallback;
}

/** notifyError：统一错误上报入口——解析 unknown 异常为文案并写入全局错误 store（顶栏 Banner 显示）。 */
export function notifyError(err: unknown, fallback = '操作失败，请重试'): void {
  useViewerStore.getState().actions.setError(errMessage(err, fallback));
}
