/**
 * ============================================================
 *  useIsMobile —— 屏幕断点检测 Hook
 * ============================================================
 * 职责：判断当前视口是否为移动端（宽度 < MOBILE_BREAKPOINT）。
 * 用于 App 层选择桌面三栏布局 / 移动单列布局。
 *
 * 断点阈值收敛在 config/appConfig.ts（MOBILE_BREAKPOINT），避免硬编码。
 */
import { useEffect, useState } from 'react';
import { MOBILE_BREAKPOINT } from '../config/appConfig';

/** 是否移动端布局 */
export function useIsMobile(): boolean {
  const [isMobile, setIsMobile] = useState(() => {
    if (typeof window === 'undefined') return false;
    return window.innerWidth < MOBILE_BREAKPOINT;
  });

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < MOBILE_BREAKPOINT);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return isMobile;
}
