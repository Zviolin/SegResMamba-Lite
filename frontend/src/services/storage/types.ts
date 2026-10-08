/**
 * ============================================================
 *  storage/types —— 存储层统一接口（与 inference/types 同构）
 * ============================================================
 * 在线/离线历史采用同一 HistoryProvider 接口，与推理侧 InferenceProvider
 * 设计模式一致：业务层不感知实现，由网关按模式路由。
 */
import type { HistoryRecord } from '../../types';

/** 历史记录存储提供者（在线=后端 API，离线=IndexedDB） */
export interface HistoryProvider {
  /** 获取全部历史记录（按创建时间倒序） */
  getRecords(): Promise<HistoryRecord[]>;
  /** 删除一条记录 */
  deleteRecord(id: number): Promise<void>;
  /** 重命名一条记录（仅改 fileName）。可选：未实现时函数可省略；调用方需自行兜底 */
  renameRecord?(id: number, fileName: string): Promise<void>;
}
