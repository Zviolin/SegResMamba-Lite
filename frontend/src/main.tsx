/**
 * ============================================================
 *  main —— React 应用入口文件
 * ============================================================
 * 职责：创建 React 根节点，挂载 <App />，并包裹：
 *  1. React.StrictMode —— 开发阶段开启严格模式（双渲染检测副作用问题）；
 *  2. BrowserRouter —— 提供前端路由上下文（HTML5 History 模式）；
 *  3. AppThemeProvider —— 提供亮色/深色两套主题（MUI + CSS 变量联动）。
 * 同时引入全局样式 index.css。
 */
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.tsx'
import AppThemeProvider from './theme/AppThemeProvider.tsx'
import './index.css'

// 找到 index.html 中 id 为 root 的挂载节点，创建 React 根
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* AppThemeProvider 提供亮色/深色主题上下文 */}
    <AppThemeProvider>
      {/* BrowserRouter 提供路由上下文，配合 App 中的 <Routes> 使用 */}
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AppThemeProvider>
  </React.StrictMode>,
)
