/**
 * ============================================================
 *  jobs/jobService —— 异步任务服务（与 inference / storage 同构）
 * ============================================================
 * 背景：原先 JobsPage 直接 import apiClient 调 getJobs / getJobStatus，
 * 表现层越过服务层直接摸 HTTP 层；而同项目的历史记录走的是 historyGateway。
 * 两条数据访问路径并存，分层就被架空了 —— 照着 Jobs 写的代码会一起越层。
 *
 * 这里把任务相关的数据访问收敛进服务层：
 *  - 组件只依赖 JobService 接口，不感知 HTTP；
 *  - 离线模式暂无本地任务队列，返回 null 由上层明确提示（而不是静默空列表）。
 */
import { getJobs, getJobStatus, submitJob } from '../api/apiClient';
import { useModeStore, type AppMode } from '../../store/modeStore';
import type { InferenceRequest, JobStatus } from '../../types';

/** 任务服务接口（与 InferenceProvider / HistoryProvider 同构） */
export interface JobService {
  /** 拉取全部任务 */
  list(): Promise<JobStatus[]>;
  /** 查询单个任务状态 */
  status(id: string): Promise<JobStatus>;
  /** 提交一个异步推理任务 */
  submit(request: InferenceRequest): Promise<JobStatus>;
}

/** 在线实现：走后端 /api/jobs */
const onlineJobService: JobService = {
  list: getJobs,
  status: getJobStatus,
  submit: submitJob,
};

/**
 * 获取当前模式可用的任务服务
 * @param mode 当前运行模式；不传时从 modeStore 直接读（非 Hook 场景用）
 * @returns 在线返回实现；离线暂无本地队列，返回 null
 */
export function getJobService(mode?: AppMode): JobService | null {
  const current = mode ?? useModeStore.getState().mode;
  return current === 'online' ? onlineJobService : null;
}
