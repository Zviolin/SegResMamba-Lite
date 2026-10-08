/**
 * ============================================================
 *  storage/folderHandle —— File System Access 目录句柄管理
 * ============================================================
 * 让用户把"数据真实存到能看到的本地文件夹"：
 *  - 浏览器（Chrome/Edge）通过 showDirectoryPicker() 让用户选择一个目录；
 *  - 把目录句柄持久化到 IndexedDB（brainseg-fs），下次打开无需重选（浏览器
 *    会依据权限策略让用户再确认一次）；
 *  - 提供查询/请求读写权限、断开文件夹等能力。
 *
 * 说明：File System Access API 仅桌面 Chrome / Edge 支持；iOS Safari 不支持。
 * 手机端后续在 Capacitor App 内用 @capacitor/filesystem 适配同一套接口。
 */
const DB_NAME = 'brainseg-fs';
const DB_VERSION = 1;
const STORE = 'handles';
const KEY = 'dirHandle';

/** 打开句柄存储库 */
function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** 当前环境是否支持 File System Access（桌面 Chromium） */
export function isFileSystemAccessSupported(): boolean {
  return typeof window !== 'undefined' && 'showDirectoryPicker' in window;
}

/** 保存目录句柄到 IndexedDB */
async function persistDirHandle(handle: FileSystemDirectoryHandle): Promise<void> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(handle, KEY);
    await new Promise<void>((res, rej) => {
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  } finally {
    db.close();
  }
}

/** 读取持久化的目录句柄（不存在返回 null） */
export async function getSavedDirHandle(): Promise<FileSystemDirectoryHandle | null> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, 'readonly');
    const store = tx.objectStore(STORE);
    return await new Promise<FileSystemDirectoryHandle | null>((resolve, reject) => {
      const req = store.get(KEY) as IDBRequest<FileSystemDirectoryHandle | undefined>;
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => reject(req.error);
    });
  } finally {
    db.close();
  }
}

/** 清除持久化的目录句柄（断开存储） */
async function clearSavedDirHandle(): Promise<void> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(KEY);
    await new Promise<void>((res, rej) => {
      tx.oncomplete = () => res();
      tx.onerror = () => rej(tx.error);
    });
  } finally {
    db.close();
  }
}

/** 目录权限状态：granted=已授予 / denied=被拒 / prompt=可请求但未授予 / unknown=接口不可用 */
type DirPermissionState = 'granted' | 'denied' | 'prompt' | 'unknown';

/**
 * 只查询当前权限状态，绝不弹窗请求。
 *
 * 关键：requestPermission 必须在用户手势（点击）里调用才有效；
 * 在「页面加载 → 拉历史 → 判断是否走文件夹模式」这类异步链路里请求会被浏览器
 * 直接拒绝，旧实现因此误判成"句柄失效"并把它删掉，用户看到文件夹莫名断开。
 * 判断"当前是否可用文件夹存储"一律用本函数。
 */
async function queryDirPermission(
  handle: FileSystemDirectoryHandle,
  mode: 'read' | 'readwrite' = 'readwrite'
): Promise<DirPermissionState> {
  try {
    // Chromium 实现扩展了 queryPermission，TS lib 未收录，故 unknown 断言
    const h = handle as unknown as { queryPermission?: (opts?: unknown) => Promise<string> };
    if (h.queryPermission) {
      const s = await h.queryPermission({ mode });
      if (s === 'granted' || s === 'denied' || s === 'prompt') return s;
    }
    return 'unknown';
  } catch (e) {
    console.warn('[fs] 目录权限查询失败:', e);
    return 'unknown';
  }
}

/**
 * 校验 / 请求目录读写权限（返回 true = 已获得权限）
 *
 * @param request 未授予时是否弹窗请求（默认 true）。
 *   只有用户手势回调里才该传 true；纯判断场景请用 queryDirPermission()，
 *   否则无手势的 requestPermission 会被浏览器拒绝。
 */
export async function ensureDirPermission(
  handle: FileSystemDirectoryHandle,
  mode: 'read' | 'readwrite' = 'readwrite',
  request = true
): Promise<boolean> {
  const state = await queryDirPermission(handle, mode);
  if (state === 'granted') return true;
  if (!request || state === 'denied') return false;
  try {
    // Chromium 扩展接口，TS lib 未收录，故 unknown 断言
    const h = handle as unknown as { requestPermission?: (opts?: unknown) => Promise<string> };
    if (h.requestPermission) {
      const r = await h.requestPermission({ mode });
      return r === 'granted';
    }
    return false;
  } catch (e) {
    console.warn('[fs] 目录权限请求失败:', e);
    return false;
  }
}

/**
 * 弹窗让用户选择/新建一个目录并持久化
 * @returns 目录句柄 + 名称；用户取消或失败返回 null
 */
export async function chooseDataDirectory(): Promise<FileSystemDirectoryHandle | null> {
  if (!isFileSystemAccessSupported()) return null;
  try {
    const picker = (window as unknown as { showDirectoryPicker?: (opts?: { mode?: string }) => Promise<FileSystemDirectoryHandle> });
    if (!picker.showDirectoryPicker) return null;
    const handle = await picker.showDirectoryPicker({ mode: 'readwrite' });
    if (handle && typeof handle.name === 'string') {
      await persistDirHandle(handle as FileSystemDirectoryHandle);
      return handle as FileSystemDirectoryHandle;
    }
    return null;
  } catch (e) {
    // 用户取消选择时 name !== AbortError 也静默处理；其它错误打印
    if ((e as { name?: string })?.name !== 'AbortError') console.warn('[fs] 选择目录失败:', e);
    return null;
  }
}

/**
 * 获取当前"已连接文件夹"信息
 * @returns active=是否配置了文件夹；name=文件夹名
 *
 * 关键修复：如果浏览器残留的句柄已失效（权限被拒/句柄过期），主动清掉，
 * 避免后续读写一直走「文件夹模式」分支失败。
 */
export async function getFolderInfo(): Promise<{ active: boolean; name: string | null }> {
  const handle = await getSavedDirHandle();
  if (!handle) return { active: false, name: null };
  try {
    // 只查询、不请求：本函数会在「拉历史列表」等无用户手势的链路里被调用，
    // 此时 requestPermission 必然失败，旧实现会据此把句柄删掉（表现为文件夹自己断开）。
    const state = await queryDirPermission(handle, 'read');
    if (state === 'granted') return { active: true, name: handle.name };
    // 只有明确被拒绝才清理；'prompt'（尚未授予）保留句柄，等用户下次点击时再请求
    if (state === 'denied') {
      console.warn('[fs] 已保存的文件夹句柄权限被拒绝，主动断开');
      await clearSavedDirHandle();
    }
    return { active: false, name: null };
  } catch (e) {
    console.warn('[fs] 获取文件夹权限异常，断开:', e);
    await clearSavedDirHandle();
    return { active: false, name: null };
  }
}

/** 断开：清除句柄（不清除已写数据） */
export async function disconnectDataDirectory(): Promise<void> {
  await clearSavedDirHandle();
}
