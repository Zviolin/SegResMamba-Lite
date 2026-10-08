# 前端交接说明（1002）

**这是什么**
脑肿瘤 MRI 分割的前端项目，技术栈是 React + TypeScript + Vite。离线能在浏览器里直接跑 ONNX 模型做分割，在线模式对接后端的异步推理接口。

**环境要求**
- 需要装 Node.js，建议用 20 或 22 的 LTS 版本（开发时用的 22）。
- 包管理器用 npm 就够了，不用额外装别的。

**怎么装依赖**
不用一个一个去下，进到这个目录跑一句就行：
`npm install`
它会按 package.json 把下面这些依赖全部自动装上，包括版本都锁在 package-lock.json 里。

**主要依赖（列出来只是让你心里有数，npm install 会一次性装好）**
- react / react-dom：界面框架
- react-router-dom：页面路由，比如 /jobs 任务页、历史记录页
- zustand：全局状态管理
- axios：发 HTTP 请求，在线模式连后端用
- @mui/material、@mui/icons-material、@emotion/react、@emotion/styled：UI 组件库和样式方案
- nifti-reader-js、pako：解析 .nii / .nii.gz 医学影像，以及解压
- jspdf：把分析报告导出成 PDF
- onnxruntime-web：写在依赖清单里，但**运行时实际用的是 public/vendor/ort 下的 UMD 全局**，推理引擎不是从 npm 这个包提供的，别被名字误导

**离线资源已经随包带了，不用另下**
public/models/ 下是分割模型（.onnx 文件），public/vendor/ort/ 下是 ORT 运行时（.js + .wasm）。这两块是浏览器离线推理必需的，已经打进压缩包，接收方不用单独下载。

**常用命令**
- npm run dev：起开发服务器，默认 5173 端口；加 --host 能让同一个局域网下的其他设备访问
- npm run build：打包到 dist/，交给部署用
- npm run preview：本地预览 build 出来的结果
- npm run test：跑单元测试（vitest）
- npm run lint / npm run typecheck：代码规范检查和 TypeScript 类型检查

**对接后端要注意**
在线模式的接口地址在前端设置里填（ServerSettingsDialog），默认走 /api/*。前端目前没有对后端返回做统一信封解析，如果后端字段有变动，改 src/services/api/apiClient.ts 这一处就行。

**备注：1002**
