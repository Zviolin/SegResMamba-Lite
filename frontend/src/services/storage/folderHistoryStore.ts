/**
 * ============================================================
 *  storage/folderHistoryStore —— 把历史记录真实写入本地文件夹
 * ============================================================
 * 在用户选中的文件夹根目录下创建 `brainseg-records/`：
 *
 *   <用户选择的文件夹>/
 *   └─ brainseg-records/
 *      └─ rec_<id>/
 *         ├─ record.json     # 记录元信息（文件名/模型/体积/时间...）
 *         ├─ mri.nii.gz      # 原始 MRI（gzip 时）或 mri.nii（未压缩）
 *         └─ seg.nii.gz      # 分割结果（始终 gzip）
 *
 * 全部是标准文件，用户可在资源管理器中直接看到/拷贝/用其它 NIfTI 工具打开。
 *
 * 注意：本模块依赖 File System Access（桌面 Chrome/Edge）。
 * 手机 App 场景改用 @capacitor/filesystem 实现同接口（结构相同）。
 */
import {
  getSavedDirHandle,
  ensureDirPermission,
} from './folderHandle';
import type { HistoryRecord } from '../../types';

/** 根子目录名 */
const ROOT_NAME = 'brainseg-records';
/** 记录目录前缀 */
const PREFIX = 'rec_';

/** 快速判断 Blob 是否为 gzip（前两字节 1F 8B） */
async function sniffGzip(blob: Blob): Promise<boolean> {
  try {
    const head = new Uint8Array(await blob.slice(0, 2).arrayBuffer());
    return head.length === 2 && head[0] === 0x1f && head[1] === 0x8b;
  } catch {
    return false;
  }
}

/** 获取（必要时创建）记录根目录 */
async function getRootDir(handle: FileSystemDirectoryHandle, create: boolean): Promise<FileSystemDirectoryHandle> {
  return handle.getDirectoryHandle(ROOT_NAME, { create });
}

/** 写一个 Blob 到目录下的文件 */
async function writeBlob(dir: FileSystemDirectoryHandle, name: string, blob: Blob): Promise<void> {
  const fileHandle = await dir.getFileHandle(name, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(blob);
  await writable.close();
}

/** 读取目录下的文件为 Blob（不存在返回 undefined） */
async function readBlob(dir: FileSystemDirectoryHandle, name: string): Promise<Blob | undefined> {
  try {
    const fh = await dir.getFileHandle(name);
    return await fh.getFile();
  } catch {
    return undefined;
  }
}

/** 建立可用目录连接：取句柄 → 校验权限；失败返回 null */
async function readyRoot(): Promise<FileSystemDirectoryHandle | null> {
  const handle = await getSavedDirHandle();
  if (!handle) return null;
  const ok = await ensureDirPermission(handle, 'readwrite');
  if (!ok) return null;
  return handle;
}

/** 由 record.fileName 得到一个安全的 MRI 文件名（记录原名，去掉路径危险字符） */
function safeMriName(fileName: string, isGz: boolean): string {
  const base = (fileName || '').split(/[\\/]/).pop() || 'mri';
  // 去掉 Windows 非法字符（含 \u0000-\u001f 控制字符，故显式关闭该规则告警）；再补后缀
  // eslint-disable-next-line no-control-regex
  const cleaned = base.replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_').trim();
  const hasExt = /\.(nii\.gz|nii)$/i.test(cleaned);
  const suffix = isGz ? '.nii.gz' : '.nii';
  return hasExt ? cleaned : cleaned.replace(/\.[^.]*$/, '') + suffix;
}

/**
 * 保存一条历史记录到本地文件夹
 * @returns 生成的记录 id；失败抛错
 */
export async function saveFolderRecord(
  record: Omit<HistoryRecord, 'id'>,
  segGz?: Blob,
  mriGz?: Blob
): Promise<number> {
  const root = await readyRoot();
  if (!root) throw new Error('未连接本地存储文件夹或权限不足');

  const recordsDir = await getRootDir(root, true);
  // 本地文件记录用时间戳做 id（IndexedDB 用自增，两边互不冲突：只用当前生效的存储）
  const id = Date.now();
  const dir = await recordsDir.getDirectoryHandle(`${PREFIX}${id}`, { create: true });

  // 写元信息（含 id；mriFileName 用于精确恢复原始名）
  await writeBlob(dir, 'record.json', new Blob([JSON.stringify({ ...record, id })], { type: 'application/json' }));
  // 写分割（始终 .nii.gz）
  if (segGz) await writeBlob(dir, 'seg.nii.gz', segGz);
  // 写原始 MRI（按是否 gzip 决定后缀）
  if (mriGz) {
    const isGz = await sniffGzip(mriGz);
    await writeBlob(dir, safeMriName(record.fileName || '', isGz), mriGz);
  }
  return id;
}

/** 列出文件夹内的历史记录（按创建时间倒序） */
export async function listFolderRecords(): Promise<HistoryRecord[]> {
  const root = await readyRoot();
  if (!root) return [];
  let recordsDir: FileSystemDirectoryHandle;
  try {
    recordsDir = await getRootDir(root, false);
  } catch {
    return []; // 尚无记录目录
  }
  const out: HistoryRecord[] = [];
  // @ts-expect-error FileSystemDirectoryHandle.values() 迭代器类型在 lib.dom 中缺省
  for await (const entry of recordsDir.values()) {
    if (entry.kind !== 'directory' || !entry.name.startsWith(PREFIX)) continue;
    try {
      const dir = await recordsDir.getDirectoryHandle(entry.name);
      const metaBlob = await readBlob(dir, 'record.json');
      if (!metaBlob) continue;
      const parsed = JSON.parse(await metaBlob.text());
      if (parsed && typeof parsed.id !== 'undefined') out.push(parsed as HistoryRecord);
    } catch (e) {
      console.warn('[folder] 读取记录失败:', entry.name, e);
    }
  }
  out.sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
  return out;
}

/** 读取某条记录（含 Blob：segGz / mriGz） */
export async function readFolderRecord(id: number): Promise<{ record: HistoryRecord; segGz?: Blob; mriGz?: Blob } | null> {
  const root = await readyRoot();
  if (!root) return null;
  let recordsDir: FileSystemDirectoryHandle;
  try {
    recordsDir = await getRootDir(root, false);
  } catch {
    return null;
  }
  let dir: FileSystemDirectoryHandle;
  try {
    dir = await recordsDir.getDirectoryHandle(`${PREFIX}${id}`);
  } catch {
    return null;
  }

  const metaBlob = await readBlob(dir, 'record.json');
  if (!metaBlob) return null;
  const record = JSON.parse(await metaBlob.text()) as HistoryRecord;

  // 找 MRI 文件：record.json 之外的第一个非 seg 文件
  let mriGz: Blob | undefined;
  // @ts-expect-error values() 迭代器类型缺省
  for await (const entry of dir.values()) {
    if (entry.kind === 'file' && entry.name !== 'record.json' && entry.name !== 'seg.nii.gz') {
      const fh = await dir.getFileHandle(entry.name);
      mriGz = await fh.getFile();
      break;
    }
  }
  const segGz = await readBlob(dir, 'seg.nii.gz');
  return { record, segGz, mriGz };
}

/** 删除文件夹内某条记录（整目录） */
export async function deleteFolderRecord(id: number): Promise<void> {
  const root = await readyRoot();
  if (!root) return;
  try {
    const recordsDir = await getRootDir(root, false);
    await recordsDir.removeEntry(`${PREFIX}${id}`, { recursive: true });
  } catch (e) {
    console.warn('[folder] 删除记录失败:', id, e);
  }
}

/**
 * 重命名文件夹模式某条记录：覆盖 record.json 中的 fileName；如果存在 MRI 文件，则同步改名。
 */
export async function renameFolderRecord(id: number, fileName: string): Promise<void> {
  const trimmed = (fileName || '').trim();
  if (!trimmed) throw new Error('文件名不能为空');
  const root = await readyRoot();
  if (!root) throw new Error('未连接本地存储文件夹或权限不足');
  const recordsDir = await getRootDir(root, false);
  const dir = await recordsDir.getDirectoryHandle(`${PREFIX}${id}`);
  const metaBlob = await readBlob(dir, 'record.json');
  if (!metaBlob) throw new Error('记录元信息不存在');
  const record = JSON.parse(await metaBlob.text()) as HistoryRecord;
  record.fileName = trimmed;
  await writeBlob(dir, 'record.json', new Blob([JSON.stringify(record)], { type: 'application/json' }));
  // 同步改名原始 MRI 文件（如果存在且原本以记录 fileName 为名）
  // @ts-expect-error values() 迭代器类型缺省
  for await (const entry of dir.values()) {
    if (entry.kind === 'file' && entry.name !== 'record.json' && entry.name !== 'seg.nii.gz') {
      const isGz = await sniffGzip(await (await dir.getFileHandle(entry.name)).getFile());
      const newName = safeMriName(trimmed, isGz);
      if (newName !== entry.name) {
        try {
          const oldFh = await dir.getFileHandle(entry.name);
          const file = await oldFh.getFile();
          const newFh = await dir.getFileHandle(newName, { create: true });
          const writable = await newFh.createWritable();
          await writable.write(await file.arrayBuffer());
          await writable.close();
          await dir.removeEntry(entry.name);
        } catch (e) {
          console.warn('[folder] 同步 MRI 文件名失败:', e);
        }
      }
      break;
    }
  }
}
