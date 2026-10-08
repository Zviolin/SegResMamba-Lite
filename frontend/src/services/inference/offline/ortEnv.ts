/**
 * ============================================================
 *  inference/offline/ortEnv —— ORT 运行时环境配置
 * ============================================================
 * 职责：初始化 window.ort 的 WASM 运行环境（线程/SIMD/Worker/路径），
 * 并提供 SIMD 低端设备降级能力。
 *
 * 说明：配置项集中在 config/appConfig.ts 的 ORT_CONFIG；
 * 本模块不依赖任何 UI/React，可独立测试。
 */
import { ORT_CONFIG } from '../../../config/appConfig';

let ortConfigured = false;
/** 是否启用 SIMD：低端设备加载失败时降级 false 并重试 */
let useSimd = ORT_CONFIG.simd;

/**
 * 初始化 ORT 运行时（幂等；SIMD 降级后再次调用会重新配置）。
 * 与 www 保持兼容：numThreads=1 / simd / proxy / wasmPaths。
 */
export function ensureOrt(): void {
  const ort = window.ort;
  if (!ort) {
    throw new Error('ORT 运行时未加载（请检查 index.html 中的 /vendor/ort/ort.min.js）');
  }
  if (ortConfigured) return;
  ort.env.wasm = ort.env.wasm || {};
  ort.env.wasm.numThreads = ORT_CONFIG.numThreads;
  ort.env.wasm.simd = useSimd;
  ort.env.wasm.proxy = ORT_CONFIG.proxy;
  // ★ 必须用绝对 URL：proxy=true 时 ORT 内部 Worker 的 fetch 不能用相对路径，
  //   否则 WorkerGlobalScope 解析 '/vendor/ort/ort-wasm-simd.wasm' 失败。
  //   www 的做法：用 new URL(..., location.href).href 拼成绝对 URL。
  const wasmBase = new URL(ORT_CONFIG.wasmDir, window.location.href).href;
  ort.env.wasm.wasmPaths = {
    'ort-wasm.wasm': wasmBase + 'ort-wasm.wasm',
    'ort-wasm-simd.wasm': wasmBase + 'ort-wasm-simd.wasm',
  };
  ortConfigured = true;
}

/**
 * 降级为非 SIMD 模式（下次 ensureOrt 时生效）。
 * 调用方应在 session 创建失败时先调用本函数再重试。
 */
export function downgradeSimd(): void {
  if (!useSimd) return;
  console.warn('[ort] SIMD WASM 加载失败，降级为非 SIMD 模式');
  useSimd = false;
  ortConfigured = false;
}
