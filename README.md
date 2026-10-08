# SegResMamba-Lite - WebAPP

> 脑肿瘤分割 Web 应用（前端 + 后端 + Docker 部署）

## 📋 项目概述

本目录是 `SegResMamba-Lite` 项目的 **Web 应用配套**分支（`webapp` 分支）。

基于 V10（最终模型，论文级 token 级稀疏 MoA，1.42M 参数），提供：
- 📤 **上传**脑部 MRI（nii.gz 格式）
- 🎯 **自动分割**肿瘤区域（WT/TC/ET）
- 📊 **可视化** 3D 分割结果
- 📥 **下载**报告（CSV/PNG）

## 🏗️ 架构设计

```
WebAPP/
├── frontend/                  # 前端（React/Vue + Three.js）
│ ├── src/
│ │ ├── components/         # 组件
│ │ │ ├── UploadPanel/     # 上传组件
│ │ │ ├── SegmentationView/  # 分割可视化（3D）
│ │ │ └── ReportPanel/      # 报告面板
│ │ ├── pages/
│ │ ├── App.js
│ │ └── index.js
│ ├── public/
│ ├── package.json
│ └── Dockerfile
│
├── backend/                   # 后端（FastAPI + MONAI）
│ ├── app/
│ │ ├── api/                # REST API
│ │ │ ├── upload.py        # 上传接口
│ │ │ ├── segment.py      # 分割接口
│ │ │ └── report.py       # 报告接口
│ │ ├── models/             # 模型加载
│ │ │ ├── loader.py       # V10 加载
│ │ │ └── inference.py    # 推理封装
│ │ ├── utils/              # 工具
│ │ └── main.py
│ ├── requirements.txt
│ └── Dockerfile
│
├── docker-compose.yml        # 一键启动
├── nginx.conf                # 反向代理
├── .env.example              # 环境变量示例
└── README.md
```

## 🚀 快速开始（待开发）

```bash
# 启动全部服务
docker-compose up -d

# 访问
# 前端：http://localhost:3000
# 后端：http://localhost:8000
# API 文档：http://localhost:8000/docs
```

## 🔧 技术栈（待开发）

| 模块 | 技术 |
|------|------|
| 前端 | React + TypeScript + Three.js + Material-UI |
| 后端 | FastAPI + MONAI + PyTorch + CUDA |
| 数据库 | PostgreSQL（病例管理）|
| 缓存 | Redis（任务队列）|
| 反向代理 | Nginx |
| 容器化 | Docker + Docker Compose |
| 模型推理 | V10 + ONNX Runtime |

## 📦 API 接口（规划）

### 上传 MRI

```http
POST /api/v1/upload
Content-Type: multipart/form-data

file: brain_mri.nii.gz
```

### 执行分割

```http
POST /api/v1/segment
Content-Type: application/json

{
  "case_id": "BraTS_001",
  "model": "v10",
  "init_filters": 20
}
```

### 获取报告

```http
GET /api/v1/report/{case_id}
```

## 📊 模型选择（V10 为最终模型）

| 场景 | 推荐版本 | 理由 |
|------|---------|------|
| **稳定生产** | **V10** | 最终模型：全局 Dice 0.8878、HD95 7.90、LW Dice 0.8304 全系列最高（官方 374mm 满罚口径） |
| **批量处理** | V10 + ONNX | 推理速度优化 |

> 注：V8 权重未留存，无法部署；完整版本对比见 experiments 分支 `SegResMamba-Lite/论文实验章节.md`。

## 📚 关联项目

- 模型代码：[experiments 分支](https://github.com/Zviolin/SegResMamba-Lite/tree/experiments)
- Android 客户端：[App 分支](https://github.com/Zviolin/SegResMamba-Lite/tree/App)

---

**状态**: 🚧 占位（待开发）
**License**: MIT | **Author**: Zviolin