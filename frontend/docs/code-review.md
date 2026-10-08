# src 代码审核报告（三遍）

> 审核范围：`src/**`（54 个 .ts/.tsx，约 9500 行）
> 审核日期：2026-09-27
> 基线检查：`tsc -b` 通过（0 error）｜`eslint` 0 error / 26 warning｜`vitest` 16 passed
## 修复进度（2026-09-28 更新）

复检：`tsc -b` 0 error｜`eslint` **0 warning**（原 26 条 any 全部收紧）｜`vitest` 16 passed｜`npm run build` 成功

| 条目 | 状态 |
|---|---|
| 第一遍 A：7 个零引用导出 | ✅ 已删 6 个（submitJob 保留：异步任务是既定功能，已在 JobsPage 如实说明入口未开放） |
| 第一遍 B：useModeProvider / generateReportData / actions.reset | ✅ 全部删除，相关 import 与注释同步清理 |
| 第一遍 C1：异步任务空头支票 | ✅ JobsPage 空态文案改为「当前版本主页仅支持同步推理，异步提交入口尚未开放」 |
| 第一遍 C2：Shift/Ctrl 滚轮 | ✅ PlaneViewer 已实现（Shift 调窗位 ±20、Ctrl/⌘ 调窗宽 ±50） |
| 第一遍 C3：点击视图定位 | ✅ PlaneViewer 已实现点击换算体素坐标并联动另两个平面 |
| 第一遍 C4：在线模式文件夹开关误导 | ✅ 在线时 Chip 置灰显示「后端存储（在线）」，点击给出说明 |
| 第一遍 D：axios 依赖 | ⚠️ 未卸载（需 npm uninstall axios，会改 package-lock，待确认） |
| 第二遍 P0-1：GridExport 颜色不一致 | ✅ utils/blend.ts 作为唯一混色实现，视图与导出共用，颜色取自 LABEL_COLORS |
| 第二遍 P0-2：render 期副作用 | ✅ 改 useEffect 重绘，依赖含区域显隐 |
| 第二遍 P0-3：runModelInference 缺 mode | ✅ 依赖补齐 |
| 第二遍 P1-4：冠状位初始切片 | ✅ 改用 ny/2 |
| 第二遍 P1-5：PlaneViewer 切片数字 | ✅ 改为选择器订阅 |
| 第二遍 P1-6：除零 NaN | ✅ percentOf() 保护 |
| 第二遍 P1-7：统计重复计算 | ✅ 收敛到 AnalysisReport（带 (segData, mriData) 去重 ref） |
| 第二遍 P1-8：文件夹权限时机 | ✅ 新增 queryDirPermission() 只查不请求；仅 denied 才清句柄 |
| 第二遍 P1-9：模型列表重复请求 | ✅ loadModels 改为函数内读取当前模型 |
| 第二遍 P1-10：ORT IO 名硬编码 | ✅ 改用 session.inputNames[0] / outputNames[0]，缺失时明确报错 |
| 第二遍 P1-11：MRI 入 IndexedDB 无上限 | ✅ 新增 80MB 上限，超出只存分割结果（详情页自动降级伪 MRI 灰度） |
| 第二遍 P2-12：切在线不重新探活 | ⚠️ 未改（体验增强，非缺陷） |
| 第三遍：调试 console 残留 | ✅ 移除（含 MobileDisplayControls 版本水印） |
| 第三遍：图例 3 份实现 | ✅ 新增 LEGEND_ORDER，App 与历史详情页改为数据驱动，硬编码颜色消除 |
| 第三遍：全量 store 订阅 | ⚠️ 未改（性能重构，涉及 7 个文件，建议单独一轮） |
| 第三遍：格式化函数重复 | ⚠️ 未抽公共 utils（纯重构，不影响功能） |
| 第三遍：alert 与 notifyError 混用 | ⚠️ 部分收敛（错误文案统一走 errMessage()，弹窗仍为 alert） |
| 第三遍：根目录垃圾文件 | ⚠️ 未删（3 个 _debug_screenshot*.png、eslint-report.json、vite.config.ts.timestamp-*.mjs，待确认） |

---

## 第一遍：写了但没用上的代码

### A. 完全零引用的导出（可直接删）

| 位置 | 符号 | 说明 |
|---|---|---|
| `api/apiClient.ts` | `uploadSegFile` | 分割标签上传后端，全程未调用（标签只本地解析） |
| `api/apiClient.ts` | `getInferenceStatus` | 同步推理状态轮询，未调用 |
| `api/apiClient.ts` | `getHistoryDetail` | 详情走 `historyGateway.getRecordStorage`，未调用 |
| `api/apiClient.ts` | `submitJob` | **异步任务提交，全项目无入口**（见 C1） |
| `services/inference/offline/modelSessionCache.ts` | `sessionCount` | 诊断用，无人调用 |
| `services/inference/offline/ortEnv.ts` | `isSimdEnabled` | 诊断用，无人调用 |
| `services/storage/folderHistoryStore.ts` | `disconnectFolder` | 网关走的是 `folderHandle.disconnectDataDirectory`，本函数是重复实现 |

### B. 有实现但无调用方的"孤儿"代码

| 位置 | 符号 | 说明 |
|---|---|---|
| `store/modeStore.ts` | `useModeProvider()` | 注释称其为"业务组件获取引擎的推荐入口"，实际 `useInference` 自己写了 `mode === 'online' ? onlineProvider : offlineProvider`，本 hook 零调用 |
| `utils/statistics.ts` | `generateReportData()` | 报告结构化函数，但 `AnalysisReport` 自己内联算了一遍占比，本函数零调用 |
| `store/viewerStore.ts` | `actions.reset()` | 整店重置，无任何 UI 入口 |

### C. UI/文案承诺了但根本没实现的功能

1. **异步任务提交是空头支票**：`JobsPage` 空态写着"通过主页面提交推理时可选择异步模式"，但主页 `InferencePanel` 只有同步 `runModelInference`，没有任何异步入口。`submitJob` 从未被调用 → 该页面永远只能看到空列表。
2. **快捷键没接**：`Legend` 提示"Shift+滚轮调节亮度 | Ctrl+滚轮调节对比度"，但 `PlaneViewer` 的 wheel 回调只读 `e.deltaY`，完全没判断 `shiftKey/ctrlKey`。
3. **"点击视图定位"没接**：`PositionInfoPanel` 底部提示"点击视图定位"，但 canvas 没有任何点击 → 坐标写入的实现。
4. **在线模式下的"本地文件夹"开关是误导**：`HistoryPage` 无论什么模式都显示"浏览器内置存储 / 文件夹"Chip，点击可连文件夹并显示已连接；但 `historyGateway.getHistoryRecords()` 在线分支直接 `return remoteHistoryProvider.getRecords()`，压根不看 folder → 连了也白连。

### D. 未使用的依赖与类型

- **axios**：`package.json` 里有，`src/` 里 0 处引用（apiClient 全部用 `fetch`）→ 可卸载，减少体积。
- **`InferenceStatus` 类型**：`types/index.ts:199` 定义后无人使用（各处都写 `InferenceState['status']`）。
- **`NiftiHeader`** 的 `affine / qform_code / sform_code / magic / sizeof_hdr` 5 个字段声明后从未被读取。

---

## 第二遍：功能 bug 与逻辑错误

### P0（建议尽快修）

1. **导出的网格图颜色跟屏幕不一致**（`GridExport.tsx:305-318`）
   硬编码 `NCR=(30,100,255)`、`ED=(30,255,100)`、`ET=(255,50,50)`，而 `config/labels.ts` 规定的是 `NCR=(100,181,246)`、`ED=(129,199,132)`、`ET=(229,115,115)`，且标签值 1/2/3/4 也是硬编码。`labels.ts` 注释明确写了"视图叠加、图例、勾选控件都必须从这里取色"——这里直接违反了。用户导出的 PNG/PDF 和界面看到的分割颜色是两套。
   → 复用 `LABEL_COLORS`，并把混色逻辑抽成公共函数供 `useViewSync` 和 `GridExport` 共用。

2. **GridExport 在 render 阶段做副作用**（`GridExport.tsx:363-379`）
   - `useState(() => { ...renderGrid(canvasRef.current) })`：在 state 初始化器里画 canvas；
   - 渲染期直接改 `prevDataRef.current` 并 `setTimeout` 重绘（应在 `useEffect` 里做）；
   - StrictMode 下会双调用，可能出现重复绘制/闪烁。
   - **且 `dataKey` 漏了 `showNecrosis / showEdema / showEnhancing`**：区域显隐开关改了之后网格图不会重绘。

3. **`runModelInference` 闭包缺 `mode` 依赖**（`useInference.ts:162`，ESLint 已警告）
   函数体里用了 `mode === 'offline'` 决定是否保存历史，但 `useCallback` 依赖数组没有 `mode`。推理过程中切换模式 → 用的是旧值 → 历史记录存错位置（该存 IndexedDB 的没存 / 不该存的存了）。

### P1

4. **冠状位初始切片算错**（`useNiftiData.ts:59-63`）
   `slicePosition.coronal` 是 **Y 轴索引**（上限 ny-1，通常 239），却赋成了 `nz/2`（77）。既不是中间层，注释也自相矛盾（注释说"与轴位共用 nz/2"）。应为 `Math.floor(ny/2)`。

5. **PlaneViewer 底部切片数字不订阅状态**（`PlaneViewer.tsx:194`）
   渲染里直接 `useViewerStore.getState().slicePosition[plane]` 读快照，本身不订阅。目前靠父组件 `PlanePanel` 整个 store 订阅才"碰巧"更新；一旦在别处复用（如独立嵌入）就永远不会刷新。

6. **除零导致 NaN**（`AnalysisReport.tsx:241,252`）
   `statistics.totalVoxels === 0`（分割结果全背景）时，占比 = `0/0 = NaN`，`LinearProgress value={NaN}`、显示 "NaN%"。需加保护。

7. **统计被重复计算两次**
   `useInference` 推理成功后算一次（第 144 行），`AnalysisReport` 的 `useEffect` 又算一次。240×240×155 ≈ 900 万体素遍历两遍，纯浪费。建议只保留 `AnalysisReport` 那一处（它依赖 mriData+segData，覆盖更全）。

8. **文件夹权限请求缺用户手势**（`folderHandle.ts:97` + `historyGateway.folderActive()`）
   `fetchHistory → folderActive → getFolderInfo → ensureDirPermission → requestPermission`，整条链在异步加载中触发，没有用户手势。Chrome 通常会拒绝 → 走到 catch → 静默清除句柄，用户看到"文件夹突然自己断开了"。

9. **在线模式模型列表重复请求**（`useInference.ts:91,94`）
   `loadModels` 依赖 `inference.selectedModel`，首次 `setSelectedModel` 后依赖变化 → `loadModels` 重建 → `useEffect` 再跑 → 又一次 `/api/models`。每次挂载至少两次请求。

10. **ORT 输入/输出名硬编码**（`offlineProvider.ts:123-127`）
    `session.run({ input: tensor })` 和 `outputs.output` 写死了名字。模型 IO 名一旦不是 `input/output`（换模型时很常见）直接抛 undefined。应该用 `session.inputNames[0] / session.outputNames[0]`。

11. **离线历史把整份原始 MRI 复制进 IndexedDB**（`useInference.ts:170`）
    每次推理都 `new Blob([mriFile.arrayBuffer()])` 整份入库，单个 MRI 几十 MB，无条数上限、无清理策略，长期使用会撑爆配额且无提示。

### P2

12. `App.tsx` 的后端探活只在挂载时执行一次；用户从离线切到在线后不会重新探活，`ModeSwitch` 也不会提示"后端不可达"。
13. `historyGateway.getRecordStorage()` 在线模式直接返回 `{ record: undefined }`，在线记录的详情页只能报错，没有降级说明。

---

## 第三遍：重复代码、架构与规范

### 重复实现（同一件事写了多份）

| 重复内容 | 出现位置 | 建议 |
|---|---|---|
| **分割图例** | `Legend.tsx`（用 LABEL_COLORS）、`App.tsx` MobileMainPage（**颜色硬编码** `#E57373/#81C784/#64B5F6`）、`HistoryDetailPage`（又一份） | 抽 `<SegLegend />` 组件，三处复用 |
| 文件状态 Chip + 大小格式化 | `FileUploader.tsx` 与 `LabelFileZone.tsx` 各一份，代码几乎逐行相同 | 抽 `utils/format.ts` |
| `formatTime / formatDate / formatVolume` | `HistoryPage.tsx` 与 `JobsPage.tsx` 各一份 | 同上 |
| 切片混色渲染 | `useViewSync` 与 `GridExport.renderGrid` | 抽公共函数（附带的收益：修掉 P0-1 的颜色不一致） |
| 获取推理引擎 | `modeStore.useModeProvider` 与 `useInference` 内联三元 | 二选一 |
| 断开文件夹 | `folderHandle.disconnectDataDirectory` 与 `folderHistoryStore.disconnectFolder` | 删后者 |
| ServerSettingsDialog | `App.tsx` 和 `ModeSwitch` 各挂一个实例 | 收敛到一处 |

### 性能

- **大量无选择器整体订阅**：`App.tsx`（DesktopMainPage）、`useInference`、`ControlPanel`、`LabelFileZone`、`AnalysisReport`、`PositionInfoPanel`、`MobileTriView` 都是 `useViewerStore()` 全量订阅。任何一个字段变化（含推理进度每 1% 的跳动）都会让整棵子树重渲染，`GridExport` 尤其重（会重画整张网格）。
  → 改成 `useViewerStore((s) => s.xxx)` 选择器订阅（项目里 `MobileDisplayControls`、`MobileMainPage` 已经这么做了，两派风格混用）。
- `useInference` 里 `useViewerStore().actions` 调用了两次，可合并。
- 三个 `PlaneViewer` 共用同一个 `getCurrentSliceImage`（依赖含 `slicePosition`），任一切片变化会触发三个平面全部重绘。

### 规范与卫生

- **生产代码里的调试残留**：`MobileDisplayControls.tsx:17` 的版本水印 `console.log('[MobileDisplayControls] v... loaded')`；`offlineProvider.ts` 4 处 `console.log`（文件名、维度、模态、类别统计）。全项目 5 处 `console.log/debug`。
- **`void c0;`**（`offlineProvider.ts:135`）：为规避未使用变量写的占位，应直接不解构 c0。
- **错误处理三套并存**：`notifyError`（顶栏 Banner）/ `inference.error`（面板内 Alert）/ 原生 `alert()`（7 处，集中在 `InferencePanel.handleExport` 与 `HistoryPage`）。
- **20 处 `: any`**，ESLint 26 条 warning 基本由此产生（主要在 storage 层与页面事件回调）。
- **根目录垃圾文件**：`_debug_screenshot.png`、`_debug_screenshot_d7.png`、`_debug_screenshot_single_btn.png`、`eslint-report.json`、`vite.config.ts.timestamp-*.mjs`（Vite 临时文件）。
- **`docs/功能概述.md` 停留在 2026-09-07**，与现在的移动端重构、离线多模态、文件夹存储、历史回放页已明显脱节；建议同步或标注版本。

---

## 建议修复顺序

1. **P0**：GridExport 颜色不一致 + render 期副作用 + `runModelInference` 缺 `mode` 依赖
2. **P1**：冠状位初始切片、PlaneViewer 订阅、除零 NaN、重复统计、文件夹权限时机、ORT IO 名、IndexedDB 容量
3. **清理**：删 7 个零引用导出 + `useModeProvider` + `generateReportData` + `actions.reset`，卸载 axios
4. **重构**：抽公共图例 / 格式化 / 混色函数，全量订阅改选择器订阅，移除调试 console
