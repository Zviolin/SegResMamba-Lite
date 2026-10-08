/**
 * ============================================================
 *  ping —— 后端可达性探测
 * ============================================================
 * 供 App 启动自检等场景使用：地址与 apiClient 一致
 * （resolveApiBase 解析用户配置的后端 base），避免硬编码 '/api'。
 */
import { resolveApiBase } from './apiClient';

/** checkBackendReachable：探测后端是否可达（resolveApiBase + /api/models，超时默认 1500ms）；可达返回 true。 */
export async function checkBackendReachable(timeoutMs = 1500): Promise<boolean> {
  const ac = new AbortController();
  // 超时后中止请求（AbortError 走 catch，视为不可达）
  const timer = setTimeout(() => ac.abort(), timeoutMs);
  try {
    // 与 getAvailableModels 相同的探测端点：/models
    const r = await fetch(`${resolveApiBase()}/models`, { signal: ac.signal });
    return r.ok;
  } catch {
    // 网络错误 / 非 2xx 之外的中止等异常一律视为不可达
    return false;
  } finally {
    clearTimeout(timer);
  }
}
