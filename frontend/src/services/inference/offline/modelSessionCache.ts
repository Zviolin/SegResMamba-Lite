/**
 * ============================================================
 *  inference/offline/modelSessionCache —— 推理会话缓存与生命周期
 * ============================================================
 * 职责：
 *  1. 按模型文件懒加载 InferenceSession 并缓存（复用，避免重复加载）；
 *  2. 失败允许重试、SIMD 失败自动降级重试；
 *  3. 提供 releaseSessions()：切换模式/卸载时主动释放，防止 ORT-Web 内存泄露。
 *
 * 注意：session 是 WASM 对象，禁止放入 zustand/localStorage（无法序列化），
 * 仅在本模块内部以 Map 持有。
 */
import type { OrtSession } from '../../../types/ort';
import { OFFLINE_MODEL_FILES, OFFLINE_DEFAULT_MODEL } from '../../../config/appConfig';
import { ensureOrt, downgradeSimd } from './ortEnv';

/** 推理会话缓存（key: 模型文件路径） */
const sessionCache = new Map<string, Promise<OrtSession>>();

/** 获取（或创建）指定模型的推理会话：懒加载 + 缓存 + SIMD 降级重试 */
export async function getSession(modelType: string): Promise<OrtSession> {
  const modelPath = OFFLINE_MODEL_FILES[modelType] || OFFLINE_MODEL_FILES[OFFLINE_DEFAULT_MODEL];
  let cached = sessionCache.get(modelPath);
  if (!cached) {
    cached = (async () => {
      ensureOrt();
      // 先 fetch → arrayBuffer 再 create，避免 ORT URL/external data 解析问题
      const res = await fetch(modelPath);
      if (!res.ok) {
        throw new Error(`离线模型加载失败: HTTP ${res.status} (${modelPath})`);
      }
      const buf = await res.arrayBuffer();
      try {
        return await window.ort!.InferenceSession.create(buf, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
      } catch (e) {
        // SIMD 加载失败（低端设备）→ 降级为非 SIMD 重试一次
        downgradeSimd();
        ensureOrt();
        console.warn('[offline] 降级为非 SIMD 重试创建 session:', e);
        return await window.ort!.InferenceSession.create(buf, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
      }
    })();
    sessionCache.set(modelPath, cached);
    // 失败时允许重试（从缓存移除，下次重新创建）
    cached.catch(() => { sessionCache.delete(modelPath); });
  }
  return cached;
}

/**
 * 释放全部推理会话（切换模式 / 组件卸载时调用）。
 * 显式 release() 通知 ORT 释放 WASM 内存，防止内存泄露与手机 OOM。
 */
export async function releaseSessions(): Promise<void> {
  const entries = Array.from(sessionCache.entries());
  sessionCache.clear();
  for (const [, promise] of entries) {
    try {
      const session = await promise;
      if (session.release) await session.release();
    } catch {
      // 释放失败不影响主流程
    }
  }
}
