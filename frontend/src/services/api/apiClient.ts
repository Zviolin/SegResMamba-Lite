/**
 * ============================================================
 *  apiClient —— 后端 API 调用层
 * ============================================================
 * 本模块封装所有对后端（Spring Boot 中间层，端口 8080）的 HTTP 请求。
 *
 * 后端地址解析优先级（M5 打包支持）：
 *  1. 用户运行时设置（localStorage 键 'api-base-url'，可在"服务器设置"里改，
 *     手机 APK 场景填开发机局域网 IP，如 http://192.168.1.5:8080）；
 *  2. 构建时环境变量 VITE_API_BASE_URL（.env 配置）；
 *  3. 默认 '/api'（开发环境由 Vite 代理转发到 localhost:8080）。
 *
 * 包含两类接口：
 *  1. 同步推理相关：文件上传、模型列表、推理执行、掩码下载、状态查询；
 *  2. Java 中间层新增：历史记录增删查、异步任务提交与状态查询。
 *
 * 所有函数均返回 Promise，出错时抛出 Error（含中文错误信息）。
 *
 * 统一收敛点：
 *  - 超时：全部请求经 request() 带 AbortController 超时，避免后端挂起时
 *    请求永久 pending、UI 卡在 loading 而无法取消；
 *  - 错误：按 HTTP 状态码给出有指向性的文案（原先所有非 2xx 都抛同一句，
 *    401 / 404 / 5xx 无法区分，排查成本高）。
 */
import type { ModelInfo, InferenceResult, InferenceRequest, HistoryRecord, JobStatus } from '../../types';

/** localStorage 中用户自定义后端地址的键名 */
export const API_BASE_STORAGE_KEY = 'api-base-url';

/** 构建时环境变量（.env 中 VITE_API_BASE_URL，可为空） */
const DEFAULT_API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) || '/api';

/** 默认请求超时（毫秒） */
const DEFAULT_TIMEOUT_MS = 30_000;
/** 上传 / 推理单独放宽：大文件在慢网络下可能需要更久 */
const LONG_TIMEOUT_MS = 120_000;

/**
 * 解析当前生效的后端 API 基础地址
 * 优先用户设置 → 环境变量 → '/api'（Vite 代理）。
 * 用户设置存 localStorage 时以 "/" 结尾，这里去尾斜杠。
 */
export function resolveApiBase(): string {
  try {
    const saved = localStorage.getItem(API_BASE_STORAGE_KEY);
    if (saved && saved.trim()) return saved.trim().replace(/\/+$/, '');
  } catch {
    // localStorage 不可用时忽略
  }
  return DEFAULT_API_BASE;
}

/** 后端接口基础地址（每次请求前动态解析，支持运行时修改） */
function getApiBase(): string {
  return resolveApiBase();
}

/**
 * 统一请求入口：拼 base + path，并附加超时控制
 *
 * 为什么不用 AbortSignal.timeout()：它在 iOS 16 以下不支持，
 * 而本项目要打进 Capacitor 的移动端 WebView，得照顾老版本。
 * @throws 超时或网络不通时抛出带中文说明的 Error（不再把底层 TypeError 透给上层）
 */
async function request(
  path: string,
  init?: RequestInit,
  timeoutMs: number = DEFAULT_TIMEOUT_MS
): Promise<Response> {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    return await fetch(`${getApiBase()}${path}`, { ...init, signal: ac.signal });
  } catch (e) {
    if (e instanceof DOMException && e.name === 'AbortError') {
      throw new Error('请求超时：后端迟迟无响应，请检查网络或后端地址是否正确');
    }
    throw new Error('无法连接后端：请检查网络是否通畅、后端是否已启动');
  } finally {
    clearTimeout(timer);
  }
}

/**
 * 非 2xx 响应的统一错误出口
 *
 * 先按状态码给出有指向性的文案，其余情况尽量取后端返回的 { message }，
 * 都拿不到时才回退到调用方传入的通用文案（并附上状态码便于定位）。
 */
async function throwHttpError(response: Response, fallback: string): Promise<never> {
  switch (response.status) {
    case 401:
    case 403:
      throw new Error('没有访问权限（登录已过期或未授权）');
    case 404:
      throw new Error('接口不存在（404）：后端地址或接口版本可能不对');
    case 408:
    case 504:
      throw new Error('后端响应超时，请稍后重试');
    case 400:
    case 409:
    case 422: {
      // 业务类错误：优先取后端返回的 { message }，拿不到再退回通用文案
      const business = await response.json().catch(() => null) as { message?: string } | null;
      throw new Error(business?.message || `${fallback}（HTTP ${response.status}）`);
    }
    default: {
      if (response.status >= 500) {
        throw new Error(`后端服务异常（HTTP ${response.status}），请稍后重试`);
      }
      const data = await response.json().catch(() => null) as { message?: string } | null;
      throw new Error(data?.message || `${fallback}（HTTP ${response.status}）`);
    }
  }
}

/**
 * 上传 MRI 文件到服务端
 *
 * @param file 用户选择的原始 MRI 文件（.nii / .nii.gz）
 * @returns Promise<{ filePath: string }> 上传成功后返回文件在服务端的存储路径
 * @throws 上传失败（HTTP 非 2xx）或超时时抛出 Error
 */
export async function uploadMriFile(file: File): Promise<{ filePath: string }> {
  // 构造 multipart/form-data 表单，字段名为 'file'
  const formData = new FormData();
  formData.append('file', file);

  // POST 到 /api/upload/mri，浏览器自动设置 multipart 边界
  const response = await request(
    '/upload/mri',
    { method: 'POST', body: formData },
    LONG_TIMEOUT_MS
  );

  if (!response.ok) await throwHttpError(response, '文件上传失败');

  // 返回体形如 { filePath: '/uploads/xxx.nii.gz' }
  return response.json();
}

/**
 * 获取可用模型列表
 *
 * @returns Promise<ModelInfo[]> 模型信息数组
 * @throws 获取失败或后端不可达时抛出 Error
 */
export async function getAvailableModels(): Promise<ModelInfo[]> {
  const response = await request('/models');

  if (!response.ok) await throwHttpError(response, '获取模型列表失败');

  return response.json();
}

/**
 * 同步执行模型推理
 *
 * @param request 推理请求参数（文件路径、模态、模型类型）
 * @returns Promise<InferenceResult> 推理结果（掩码路径、耗时、体积）
 * @throws 后端返回错误信息时抛出其中的 message，否则抛出带状态码的 Error
 */
export async function runInference(req: InferenceRequest): Promise<InferenceResult> {
  const response = await request(
    '/inference/run',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    },
    LONG_TIMEOUT_MS
  );

  if (!response.ok) await throwHttpError(response, '推理失败');

  return response.json();
}

/**
 * 下载分割结果掩码文件
 *
 * @param filePath 服务端掩码文件路径（需 encodeURIComponent 转义）
 * @returns Promise<Blob> 掩码文件的二进制内容
 * @throws 下载失败或超时时抛出 Error
 */
export async function downloadMaskFile(filePath: string): Promise<Blob> {
  const response = await request(`/download/mask?path=${encodeURIComponent(filePath)}`);

  if (!response.ok) await throwHttpError(response, '下载分割结果失败');

  return response.blob();
}

// ── Java 中间层新增 API ──

/**
 * 获取推理历史记录列表
 *
 * @returns Promise<HistoryRecord[]> 历史记录数组
 * @throws 获取失败时抛出 Error
 */
export async function getHistory(): Promise<HistoryRecord[]> {
  const response = await request('/history');
  if (!response.ok) await throwHttpError(response, '获取历史记录失败');
  return response.json();
}

/**
 * 删除一条历史记录
 *
 * @param id 历史记录 ID
 * @throws 删除失败时抛出 Error
 */
export async function deleteHistory(id: number): Promise<void> {
  const response = await request(`/history/${id}`, { method: 'DELETE' });
  if (!response.ok) await throwHttpError(response, '删除历史记录失败');
}

/**
 * 重命名一条历史记录（仅修改 fileName 字段）。
 * 后端需实现 `PATCH /api/history/{id}`；若后端未实现，会返回非 2xx 状态码，前端据此提示。
 *
 * @param id 历史记录 ID
 * @param fileName 新文件名（不含扩展名）
 */
export async function renameHistory(id: number, fileName: string): Promise<void> {
  const trimmed = (fileName || '').trim();
  if (!trimmed) throw new Error('文件名不能为空');
  const response = await request(`/history/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fileName: trimmed }),
  });
  if (!response.ok) await throwHttpError(response, '重命名历史记录失败（后端可能不支持该操作）');
}

/**
 * 提交异步推理任务
 *
 * @param req 推理请求参数（文件路径、模态、模型类型）
 * @returns Promise<JobStatus> 提交后返回的任务状态（含 jobId）
 * @throws 提交失败时抛出 Error
 */
export async function submitJob(req: InferenceRequest): Promise<JobStatus> {
  const response = await request('/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  if (!response.ok) await throwHttpError(response, '提交任务失败');
  return response.json();
}

/**
 * 获取全部异步任务列表
 *
 * @returns Promise<JobStatus[]> 任务数组
 * @throws 获取失败时抛出 Error
 */
export async function getJobs(): Promise<JobStatus[]> {
  const response = await request('/jobs');
  if (!response.ok) await throwHttpError(response, '获取任务列表失败');
  return response.json();
}

/**
 * 查询单个异步任务状态
 *
 * @param id 任务 ID
 * @returns Promise<JobStatus> 任务最新状态
 * @throws 查询失败时抛出 Error
 */
export async function getJobStatus(id: string): Promise<JobStatus> {
  const response = await request(`/jobs/${id}`);
  if (!response.ok) await throwHttpError(response, '获取任务状态失败');
  return response.json();
}
