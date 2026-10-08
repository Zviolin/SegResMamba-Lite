/**
 * ============================================================
 *  storage/localHistory —— 离线本地历史记录（IndexedDB）
 * ============================================================
 * 职责：离线模式下推理历史的本地持久化（增/查/删）。
 * 断网时应用仍可查看历史推理记录，联网后与后端 /api/history 并存。
 *
 * 存储：IndexedDB（容量大，适合长期保存记录），库名 brainseg-offline，
 * 对象仓库 history（主键 id 自增）。
 *
 * M4 里程碑：实现离线数据闭环的第一环（历史记录本地化）。
 */
import type { HistoryRecord } from '../../types';

const DB_NAME = 'brainseg-offline';
const DB_VERSION = 1;
const STORE_NAME = 'history';

/** 打开（或创建）IndexedDB 连接 */
function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/** 事务包装：把 IDBRequest 转为 Promise */
function requestToPromise<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * 保存一条本地历史记录
 * @param record 记录内容（不含 id，由 IndexedDB 自增生成）
 * @param segDataGz 可选：分割结果压缩包（.nii.gz 的 Blob，历史页可导出/回放）
 * @param mriDataGz 可选：原始 MRI 文件字节（Blob，历史详情页回放三平面视图用）
 * @returns 生成的主键 id
 */
export async function saveLocalHistory(
  record: Omit<HistoryRecord, 'id'>,
  segDataGz?: Blob,
  mriDataGz?: Blob
): Promise<number> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.add({
      ...record,
      segDataGz,
      mriDataGz,
    } as HistoryRecord & { segDataGz?: Blob; mriDataGz?: Blob }) as IDBRequest<number>;
    return await requestToPromise(req);
  } finally {
    db.close();
  }
}

/**
 * 读取单条历史记录（含分割压缩包 segDataGz 与原始 MRI mriDataGz）
 * @param id 记录主键
 */
export async function getLocalHistoryById(id: number): Promise<
  (HistoryRecord & { segDataGz?: Blob; mriDataGz?: Blob }) | undefined
> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    return await requestToPromise(store.get(id) as IDBRequest<(HistoryRecord & { segDataGz?: Blob; mriDataGz?: Blob }) | undefined>);
  } finally {
    db.close();
  }
}

/**
 * 获取全部本地历史记录（按创建时间倒序）
 */
export async function getLocalHistory(): Promise<HistoryRecord[]> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const records = await requestToPromise(store.getAll() as IDBRequest<HistoryRecord[]>);
    // 按创建时间倒序（最新在前）
    records.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
    return records;
  } finally {
    db.close();
  }
}

/**
 * 删除一条本地历史记录
 * @param id 记录主键
 */
export async function deleteLocalHistory(id: number): Promise<void> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    await requestToPromise(store.delete(id) as IDBRequest<undefined>);
  } finally {
    db.close();
  }
}

/**
 * 重命名一条本地历史记录（仅修改 fileName 字段，不动 seg/mri 数据）
 * @param id 记录主键
 * @param fileName 新文件名（不含扩展名；空字符串会被拒绝）
 */
export async function renameLocalHistory(id: number, fileName: string): Promise<void> {
  const trimmed = (fileName || '').trim();
  if (!trimmed) throw new Error('文件名不能为空');
  const db = await openDb();
  try {
    // 注意：get 和 put 必须共用同一个 readwrite 事务，否则事务会自动 commit
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const txStore = tx.objectStore(STORE_NAME);
    const existing = await requestToPromise(
      txStore.get(id) as IDBRequest<(HistoryRecord & { segDataGz?: Blob; mriDataGz?: Blob }) | undefined>
    );
    if (!existing) throw new Error('记录不存在');
    existing.fileName = trimmed;
    const putReq = txStore.put(existing) as IDBRequest<number>;
    await requestToPromise(putReq);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
      tx.onabort = () => reject(tx.error || new Error('IndexedDB 事务被中止'));
    });
  } catch (e) {
    console.error('[renameLocalHistory] 失败:', { id, fileName: trimmed, err: e });
    throw e;
  } finally {
    db.close();
  }
}
