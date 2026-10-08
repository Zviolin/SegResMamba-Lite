/**
 * storage/historyProviders —— 在线/离线历史 Provider 实现
 * ============================================================
 * 与 inference 侧 onlineProvider/offlineProvider 同构：
 *  remoteHistoryProvider：在线，走后端 /api/history；
 *  localHistoryProvider：离线，走 IndexedDB。
 * 由 historyGateway 按当前模式选择。
 */
import { getHistory, deleteHistory, renameHistory } from '../api/apiClient';
import { getLocalHistory, deleteLocalHistory, renameLocalHistory } from './localHistory';
import type { HistoryProvider } from './types';

/** 在线历史提供者（后端 API） */
export const remoteHistoryProvider: HistoryProvider = {
  getRecords: getHistory,
  deleteRecord: deleteHistory,
  renameRecord: renameHistory,
};

/** 离线历史提供者（IndexedDB） */
export const localHistoryProvider: HistoryProvider = {
  getRecords: getLocalHistory,
  deleteRecord: deleteLocalHistory,
  renameRecord: renameLocalHistory,
};
