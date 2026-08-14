# SegResMamba-Lite - Android App

> 脑肿瘤分割 Android 移动应用（Gradle + Kotlin）

## 📋 项目概述

本目录是 `SegResMamba-Lite` 项目的 **Android 移动应用配套**分支（`android` 分支）。

基于 V6/V8 MoA 模型，提供：
- 📱 移动端 MRI 查看
- 🎯 离线肿瘤分割（ONNX Runtime）
- 📊 3D 分割可视化
- 💾 病例本地存储

## 🏗️ 项目结构

```
Android/
├── app/                         # 主应用模块
│ ├── src/main/
│ │ ├── java/com/zviolin/segresmamba/
│ │ │ ├── MainActivity.kt
│ │ │ ├── ui/                # UI 组件
│ │ │ │ ├── upload/         # 上传界面
│ │ │ │ ├── segment/        # 分割界面
│ │ │ │ ├── viewer/         # 3D 可视化
│ │ │ │ └── settings/       # 设置
│ │ │ ├── ml/                # 机器学习
│ │ │ │ ├── ModelLoader.kt  # 模型加载
│ │ │ │ ├── ONNXInference.kt  # ONNX 推理
│ │ │ │ └── Preprocess.kt   # 预处理
│ │ │ ├── data/              # 数据层
│ │ │ │ ├── RoomDB          # 本地数据库
│ │ │ │ └── Repository
│ │ │ └── utils/             # 工具
│ │ ├── res/                  # 资源文件
│ │ └── AndroidManifest.xml
│ ├── build.gradle.kts          # 模块构建脚本
│ └── proguard-rules.pro
│
├── gradle/                      # Gradle Wrapper
│ └── wrapper/
│
├── build.gradle.kts             # 项目级构建脚本
├── settings.gradle.kts
├── gradle.properties
├── gradlew                       # Unix 启动脚本
├── gradlew.bat                  # Windows 启动脚本
└── README.md
```

## 🚀 快速开始（待开发）

```bash
# 克隆分支
git clone -b android https://github.com/Zviolin/SegResMamba-Lite.git

# 用 Android Studio 打开项目根目录

# 同步 Gradle
./gradlew build

# 安装到设备
./gradlew installDebug
```

## 🔧 技术栈（待开发）

| 模块 | 技术 |
|------|------|
| 语言 | Kotlin 1.9+ |
| UI | Jetpack Compose + Material 3 |
| 3D 渲染 | SceneView / Filament |
| ML 推理 | ONNX Runtime Mobile |
| 数据库 | Room |
| 异步 | Coroutines + Flow |
| 网络 | Retrofit + OkHttp |
| 依赖注入 | Hilt |
| 构建 | Gradle 8.x |

## 📦 模型转换（V6 → ONNX）

```python
# 在 experiments 分支执行
import torch
from models import get_model

model = get_model(version='v6', init_filters=20)
model.load_state_dict(torch.load('best_metric_model.pth'))
model.eval()

# 导出 ONNX
dummy = torch.randn(1, 4, 64, 64, 64)
torch.onnx.export(
    model, dummy, 'v6_2.0mm.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'},
                  'output': {0: 'batch'}}
)
```

将 `v6_2.0mm.onnx` 放入 `app/src/main/assets/`。

## 📱 核心功能

### 1. 上传 MRI

```kotlin
class UploadFragment : Fragment() {
    private fun pickFile() {
        // 调用系统文件选择器
        // 支持 .nii.gz 格式
    }
}
```

### 2. ONNX 推理

```kotlin
class ONNXInference(context: Context) {
    private val session: OrtSession
    
    fun segment(mriVolume: FloatArray): SegmentationResult {
        // 1. 预处理（归一化 + padding）
        // 2. ONNX 推理
        // 3. 后处理（argmax + threshold）
        // 4. 返回分割结果
    }
}
```

### 3. 3D 可视化

```kotlin
class ViewerFragment : Fragment() {
    private fun renderSegmentation(seg: SegmentationResult) {
        // 使用 SceneView 渲染 3D 分割结果
        // 支持旋转、缩放、平移
    }
}
```

## 📊 V6 vs V8 模型选择

| 场景 | 推荐版本 | 理由 |
|------|---------|------|
| **移动端（生产）** | **V6 + ONNX** | 体积小、稳定、Dice 0.89 |
| **学术 Demo** | V8 + ONNX | α 自由自适应创新 |
| **超轻量场景** | V2 + ONNX | 1.42M 极小 |

## 🧪 测试用例

```bash
# 单元测试
./gradlew test

# 仪器测试
./gradlew connectedAndroidTest
```

## 📚 关联项目

- 模型代码：[experiments 分支](../experiments)
- Web 应用：[webapp 分支](../webapp)

---

**状态**: 🚧 占位（待开发）
**Min SDK**: 24 (Android 7.0) | **Target SDK**: 34 (Android 14)
**License**: MIT | **Author**: Zviolin