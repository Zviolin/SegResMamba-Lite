/**
 * vitest 配置：单测环境 node（纯函数测试，无 DOM 依赖）
 */
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
  },
});
