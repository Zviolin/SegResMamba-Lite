import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

/**
 * ESLint flat config
 *
 * 设计：
 *   - 类型错误仍然按 error 处理（noUnusedLocals 等已经在 tsconfig 里）；
 *   - 这里只对源码做风格/正确性检查；
 *   - `no-explicit-any` 先降为 warn，给团队留处理时间（后续 PR 逐步收敛）；
 *   - 文件名清理/字节级校验场景需要匹配控制字符（\x00\x1f 等），
 *     用 `allowUnsafeRegexConstructor` 豁免正则字面量检查。
 */
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      // eslint-plugin-react-hooks v5+：
      //   `recommended` 仍指向 rc 旧版配置（向后兼容）；
      //   `recommended-latest` 是 flat config 推荐写法。
      reactHooks.configs['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
    rules: {
      // 先告警，等"中度问题"那一波 PR 收敛后再升回 error
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      // 文件名清洗 / 十六进制扫描等场景需要匹配 \x00、\x1f 等控制字符；
      // 这是"已知输入边界"，安全风险由调用方控制，这里走默认警告而非错误。
      'no-control-regex': 'warn',
    },
  },
])
