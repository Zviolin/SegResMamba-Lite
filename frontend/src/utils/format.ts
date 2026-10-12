/**
 * ============================================================
 *  utils/format —— 通用展示格式化函数
 * ============================================================
 * 收敛历史记录页 / 任务管理页各自重复定义的格式化函数，
 * 保证两处（以及后续新页面）对耗时、时间、体积的展示口径一致。
 */

/**
 * 格式化耗时：小于 1 秒显示毫秒，否则显示秒（保留 1 位小数）
 * @param ms 耗时（毫秒）
 */
export function formatTime(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * 格式化 ISO 时间为本地中文时间字符串
 * @param iso ISO 8601 时间字符串
 */
export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString('zh-CN');
}

/**
 * 格式化体积统计映射为可读字符串
 * 例如 { WT: 123.4, TC: 45.6 } -> "WT: 123.4cm3  TC: 45.6cm3"
 * @param volumes 模型返回的各区域体积（cm³）；空值返回 '-'
 */
export function formatVolume(volumes: Record<string, number> | null | undefined): string {
  if (!volumes) return '-';
  return Object.entries(volumes)
    .map(([k, v]) => `${k}: ${v.toFixed(1)}cm3`)
    .join('  ');
}
