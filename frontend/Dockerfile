# ============================================================
#  脑肿瘤分割可视化前端 —— 多阶段构建生产镜像
#  阶段 1 builder : Node 20 固定版本编译（npm ci + tsc + vite build）
#  阶段 2 runtime : nginx-unprivileged 非 root 用户静态服务（8080 端口）
#  构建：docker build -t brain-tumor-frontend:1.0.0 .
#  运行：docker run -d -p 8080:8080 brain-tumor-frontend:1.0.0
# ============================================================

# ---- 阶段 1：构建（固定版本，保证可重现构建） ----
FROM node:20.19-slim AS builder
WORKDIR /app

# 先复制依赖清单并安装：依赖未变时命中层缓存，源码改动不触发重装
COPY package.json package-lock.json .npmrc ./
RUN npm ci

# 再复制源码与构建配置
COPY . .
RUN npm run build

# ---- 阶段 2：运行（nginx 非 root 镜像，监听 8080） ----
FROM nginxinc/nginx-unprivileged:1.27-alpine

# 覆盖默认站点配置：SPA 回退 / gzip / WASM MIME / 分层缓存策略
COPY --from=builder /app/docker/nginx.conf /etc/nginx/conf.d/default.conf
# 静态产物 + 离线推理模型（ONNX）与 ORT WASM 运行时
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 8080

# 容器级健康检查：首页可访问即视为健康
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/ >/dev/null 2>&1 || exit 1
