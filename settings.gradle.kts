// Settings file for Android project
// 待开发：完整的模块配置将在此文件中定义

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "SegResMamba-Lite"
include(":app")

// 待添加模块：
// include(":ml")          // ONNX 推理模块
// include(":data")        // Room 数据库模块
// include(":ui-components")  // 共享 UI 组件