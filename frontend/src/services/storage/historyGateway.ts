/**
 * ============================================================
 *  storage/historyGateway —— 历史记录统一入口（模式判断收敛层）
 * ============================================================
 * 职责：屏蔽"在线 / 离线 / 本地文件夹"差异，向上层提供统一历史接口。
 *
 * 决策链（自上而下）：
 *  1. 在线模式  → 后端 /api/history（remote）
 *  2. 离线模式 + 已连接"本地文件夹" → File System Access 真实文件
 *  3. 离线模式 + 未连接文件夹 → IndexedDB（浏览器内置，默认兜底）
 *
 * 导出：
 *  - 历史 CRUD：getHistoryRecords / saveHistoryRecord / deleteHistoryRecord
 *  - 读取含 Blob：getRecordStorage（详情回放 / 导出用）
 *  - 文件夹开关：connectFolderStorage / disconnectFolderStorage / getFolderStorageStatus
 */
import { useModeStore } from '../../store/modeStore';
import { remoteHistoryProvider, localHistoryProvider } from './historyProviders';
import { getLocalHistory, getLocalHistoryById, saveLocalHistory, deleteLocalHistory } from './localHistory';
import {
  chooseDataDirectory,
  getFolderInfo,
  disconnectDataDirectory,
  isFileSystemAccessSupported,
} from './folderHandle';
import {
  saveFolderRecord,
  listFolderRecords,
  readFolderRecord,
  deleteFolderRecord,
  renameFolderRecord,
} from './folderHistoryStore';
import type { HistoryRecord } from '../../types';
import { errMessage } from '../../utils/notify';

/** 当前是否为离线模式（从 zustand 直接读，避免 Hook 限制） */
function isOffline(): boolean {
  return useModeStore.getState().mode === 'offline';
}

/** 是否已连接本地文件夹存储（离线且句柄有效） */
async function folderActive(): Promise<boolean> {
  if (!isOffline()) return false;
  const info = await getFolderInfo();
  return info.active;
}

/**
 * 当前生效存储的统一能力描述
 *
 * 三路存储（后端 / 本地文件夹 / IndexedDB）的能力并不一致（在线不能存 Blob、
 * 文件夹能读写真实文件），差异在这里抹平成同一形状。
 *
 * 背景：原先「在线 / 文件夹 / IndexedDB」的路由判断在下面每个 CRUD 函数里
 * 各写一遍，新增一种存储或调整优先级要改四处。收敛后路由判断只出现一次。
 */
interface StorageAdapter {
  getRecords(): Promise<HistoryRecord[]>;
  save(record: Omit<HistoryRecord, 'id'>, segGz?: Blob, mriGz?: Blob): Promise<boolean>;
  remove(id: number): Promise<void>;
  rename(id: number, fileName: string): Promise<{ ok: boolean; reason?: string }>;
  /** 记录不存在时 record 为 undefined（与对外接口保持一致） */
  getWithBlobs(id: number): Promise<{ record: HistoryRecord | undefined; segGz?: Blob; mriGz?: Blob }>;
}

/** 在线：走后端 /api/history；记录由后端自行落库，无本地 Blob */
const remoteAdapter: StorageAdapter = {
  getRecords: () => remoteHistoryProvider.getRecords(),
  save: async () => false,
  remove: (id) => remoteHistoryProvider.deleteRecord(id),
  rename: async (id, fileName) => {
    if (!remoteHistoryProvider.renameRecord) {
      return { ok: false, reason: '当前在线模式不支持重命名' };
    }
    try {
      await remoteHistoryProvider.renameRecord(id, fileName);
      return { ok: true };
    } catch (e) {
      // 后端若未实现 PATCH /api/history/:id 会返回 4xx/5xx，把原因透传便于排查
      return {
        ok: false,
        reason: `在线模式：${errMessage(e, '重命名失败')}（请确认后端已实现 PATCH /api/history/:id）`,
      };
    }
  },
  getWithBlobs: async () => ({ record: undefined }),
};

/** 离线 + 已连接文件夹：读写真实文件 */
const folderAdapter: StorageAdapter = {
  getRecords: () => listFolderRecords(),
  save: async (record, segGz, mriGz) => {
    try {
      await saveFolderRecord(record, segGz, mriGz);
      return true;
    } catch (e) {
      console.warn('[history] 写入文件夹失败:', e);
      return false;
    }
  },
  remove: (id) => deleteFolderRecord(id),
  rename: async (id, fileName) => {
    try {
      await renameFolderRecord(id, fileName);
      return { ok: true };
    } catch (e) {
      return {
        ok: false,
        reason: `文件夹模式：${errMessage(e, '重命名失败')}（请检查本地文件夹句柄是否仍然有效）`,
      };
    }
  },
  getWithBlobs: async (id) => {
    const found = await readFolderRecord(id);
    return found
      ? { record: found.record, segGz: found.segGz, mriGz: found.mriGz }
      : { record: undefined };
  },
};

/** 离线 + 未连接文件夹：IndexedDB 兜底 */
const localAdapter: StorageAdapter = {
  getRecords: () => getLocalHistory(),
  save: async (record, segGz, mriGz) => {
    try {
      await saveLocalHistory(record, segGz, mriGz);
      return true;
    } catch (e) {
      console.warn('[history] 写入 IndexedDB 失败:', e);
      return false;
    }
  },
  remove: (id) => deleteLocalHistory(id),
  rename: async (id, fileName) => {
    if (!localHistoryProvider.renameRecord) {
      return { ok: false, reason: '本地存储不支持重命名' };
    }
    try {
      await localHistoryProvider.renameRecord(id, fileName);
      return { ok: true };
    } catch (e) {
      return { ok: false, reason: `本地存储：${errMessage(e, '重命名失败')}` };
    }
  },
  getWithBlobs: async (id) => {
    const full = await getLocalHistoryById(id);
    if (!full) return { record: undefined };
    const { segDataGz, mriDataGz, ...pureRecord } =
      full as HistoryRecord & { segDataGz?: Blob; mriDataGz?: Blob };
    return { record: pureRecord, segGz: segDataGz, mriGz: mriDataGz };
  },
};

/**
 * 按当前状态挑出可用的存储实现 —— 路由判断只写这��
 *
 * 优先级：在线 → 后端；离线 + 已连文件夹 → 真实文件；离线 → IndexedDB。
 */
async function currentStorage(): Promise<StorageAdapter> {
  if (!isOffline()) return remoteAdapter;
  if (await folderActive()) return folderAdapter;
  return localAdapter;
}

/**
 * 连接一个本地存储文件夹（弹窗选择）
 * @returns {active:是否成功; name:文件夹名; supported:浏览器是否支持}
 */
export async function connectFolderStorage(): Promise<{ active: boolean; name: string | null; supported: boolean }> {
  if (!isFileSystemAccessSupported()) {
    return { active: false, name: null, supported: false };
  }
  const handle = await chooseDataDirectory();
  if (!handle) return { active: false, name: null, supported: true };
  return { active: true, name: handle.name, supported: true };
}

/** 断开文件夹（不清数据） */
export async function disconnectFolderStorage(): Promise<void> {
  await disconnectDataDirectory();
}

/** 查询当前文件夹存储状态 */
export async function getFolderStorageStatus(): Promise<{ active: boolean; name: string | null }> {
  return getFolderInfo();
}

/** 是否可用 File System Access（桌面 Chromium 提示用） */
export function folderStorageSupported(): boolean {
  return isFileSystemAccessSupported();
}

/**
 * 获取历史记录列表（在线→后端；离线→文件夹或 IndexedDB）
 */
export async function getHistoryRecords(): Promise<HistoryRecord[]> {
  return (await currentStorage()).getRecords();
}

/**
 * 保存一条推理历史（离线：文件夹优先，否则 IndexedDB；在线：后端自行落库 no-op）
 * @returns 是否成功写入本地（在线返回 false）
 */
export async function saveHistoryRecord(
  record: Omit<HistoryRecord, 'id'>,
  segDataGz?: Blob,
  mriDataGz?: Blob
): Promise<boolean> {
  return (await currentStorage()).save(record, segDataGz, mriDataGz);
}

/**
 * 删除一条历史记录（按当前生效存储路由）
 */
export async function deleteHistoryRecord(id: number): Promise<void> {
  return (await currentStorage()).remove(id);
}

/**
 * 重命名一条历史记录（按当前生效存储路由）。
 * 成功返回 { ok: true }；任一实现不支持时返回 { ok: false, reason }，便于上层提示用户。
 */
export async function renameHistoryRecord(
  id: number,
  fileName: string
): Promise<{ ok: boolean; reason?: string }> {
  // 各实现自己负责把失败转成 { ok, reason }，网关不再逐路兜错
  return (await currentStorage()).rename(id, fileName);
}

/**
 * 读取一条记录的完整内容（含可回放 Blob）
 * @returns record 不存在时 record 为 undefined；segGz/mriGz 可能缺失
 */
export async function getRecordStorage(id: number): Promise<{
  record: HistoryRecord | undefined;
  segGz?: Blob;
  mriGz?: Blob;
}> {
  return (await currentStorage()).getWithBlobs(id);
}
