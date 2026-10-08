# 脑肿瘤分割可视化系统

基于 **SegResNet-Mamba** 的脑肿瘤（胶质瘤）MRI 智能分割与三维可视化系统（SRTP 项目成果）。

纯前端即可运行：离线模式内置 ONNX 模型，无需后端即可完成「上传 MRI → 推理分割 → 三平面查看 → 量化分析」的完整演示。

## 核心能力

- **双引擎模式**：在线（后端 API）／离线（本地 ONNX Runtime Web + WASM）一键切换，断网可用
- **多模态输入**：BraTS 四模态（T1 / T1ce / T2 / FLAIR），4 模态精度优于单模态
- **三平面 MPR 可视化**：桌面 2×2 四宫格；移动端 1 大 + 2 小三视图（点小图切换、滑动/滚轮翻片）
- **显示调节**：亮度（窗位）／对比度（窗宽）／叠加透明度／区域显隐，主页常驻
- **量化分析**：肿瘤总体积、受影响切片数、NCR/ED/ET 区域分布明细（体积 + 占比）
- **历史记录**：列表 → 详情完整回放（MRI + 分割），支持导出 .nii.gz
- **本地文件存储**：数据可真实写入用户选择的文件夹（`brainseg-records/`），不再只存浏览器数据库
- **响应式**：桌面 / 手机双排版，亮暗主题

## 快速开始

```bash
npm install      # 安装依赖
npm run dev      # 启动开发服务器 → http://localhost:5173
npm run build    # 生产构建 → dist/
npm test         # 单元测试（vitest）
```

> 手机端预览：浏览器宽度 < 900px 自动进入移动排版；真机调试可运行 `npm run dev -- --host` 后用局域网 IP 访问。
> 也可双击根目录 `start-dev.bat` 一键启动（自动安装依赖）。

环境要求：Node.js 20 或 22（LTS）。

## 使用三步

1. 切换到「离线」模式（顶栏开关推左）
2. 「上传 / 推理」选择 MRI（推荐补 4 模态提升精度）→ 运行推理
3. 在三平面视图查看分割结果，拖动亮度/对比度，查看量化分析报告

## 目录结构

```
├─ src/
│  ├─ api/               # 后端 API 客户端
│  ├─ components/        # 界面组件
│  │  ├─ Controls/       #   上传 / 推理 / 显示控制 / 图例 / 设置
│  │  ├─ History/        #   历史列表 + 详情回放页
│  │  ├─ Jobs/           #   任务管理
│  │  ├─ Reports/        #   量化分析报告
│  │  └─ Viewer/         #   MPR 三平面视图
│  ├─ config/            # 应用配置（断点 / ORT / 离线模型）
│  ├─ hooks/             # 业务 hooks（推理编排 / NIfTI 加载 / 视图同步）
│  ├─ services/          # 服务层（推理引擎、历史存储）
│  │  ├─ inference/      #   在线 / 离线推理引擎
│  │  └─ storage/        #   历史存储（IndexedDB / 本地文件夹 / 网关）
│  ├─ store/             # Zustand 全局状态
│  ├─ theme/             # 亮暗主题
│  ├─ types/             # 全局类型
│  └─ utils/             # NIfTI 解析 / 导出 / 统计工具
├─ public/               # 静态资源（模型 .onnx、ORT WASM、favicon）
├─ docs/                 # 文档
└─ dist/                 # 构建产物（git 忽略，可重新生成）
```

## 文档

- 逐界面逐功能说明：[docs/功能概述.md](docs/功能概述.md)
- 需求与分析：[docs/需求与分析文档.md](docs/需求与分析文档.md)
- 前后端 API 契约：[docs/api-contract.md](docs/api-contract.md)
- 环境搭建与运行：[docs/HANDOFF-1002.md](docs/HANDOFF-1002.md)

## 技术栈

React 18 · TypeScript · Vite · MUI 5 · Zustand · ONNX Runtime Web · NIfTI reader/writer · vitest
