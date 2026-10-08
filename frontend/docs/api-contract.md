# 前后端对接 API 契约（前端 → 后端）

适用范围：本项目（脑肿瘤 MRI 分割前端 frontend-fyx）的**在线模式**所调用的全部后端接口。离线模式（浏览器内 ONNX 推理，不依赖后端）的运作机制见 §7，供后端同学理解整体架构。

状态：本契约与 `src/services/api/apiClient.ts` 当前代码一一对应。标注「建议补充」的接口为前端 UI 已规划、待后端实现。

---

## 1. 通用约定

**后端技术栈（前端假定）**：Spring Boot，端口 8080。这是前端单方面的假设，后端实际技术栈不受限，只要接口和字段对得上即可。

**基础路径的解析优先级**：前端按下面顺序决定请求打到哪。
- 第一优先：用户运行时在"服务器设置"里填的地址，存在 localStorage 键 `api-base-url`（如手机场景填 `http://192.168.1.5:8080`）。
- 第二优先：构建时环境变量 `VITE_API_BASE_URL`（在 .env 里配）。
- 默认：`/api`，开发期由 Vite 代理转发到 `localhost:8080`；部署/手机直连时通过上面的设置覆盖。

完整 URL 示例：`http://localhost:8080/api/models`。

**鉴权**：目前无。如果以后要加，前端会在 `apiClient.ts` 统一加 `Authorization` 头，不影响接口形态。

**请求体格式**：JSON 接口用 `application/json`；文件上传用 `multipart/form-data`。

**成功响应**：HTTP 2xx + JSON 体（具体字段见各接口和文末类型定义）。

**失败响应与错误文案（重要，请后端注意）**：前端不是对所有错误都读后端返回的 `message`。`apiClient` 是按 HTTP 状态码分流的，规则如下：

- 400 / 409 / 422（业务类错误）：前端**会读取**后端返回的 `{ "message": "..." }` 并展示给用户。
- 401 / 403：前端显示内置文案「没有访问权限（登录已过期或未授权）」，**忽略后端 message**。
- 404：前端显示「接口不存在（404）：后端地址或接口版本可能不对」，**忽略后端 message**。
- 408 / 504：前端显示「后端响应超时，请稍后重试」，**忽略后端 message**。
- 5xx：前端显示「后端服务异常（HTTP 5xx），请稍后重试」，**忽略后端 message**。
- 其他未列出的状态码：前端会尝试读 `{message}`，读不到再回退到内置通用文案。

结论：想让自定义错误提示透传给用户，请走 400 / 409 / 422 返回带 `message` 的 JSON；用 401/403/404/5xx 的话，提示文案是前端写死的，后端的 message 不会显示。

**超时时长**：前端统一在请求层带超时（基于 AbortController，兼容老移动端 WebView）。
- 普通请求：30 秒。
- 上传文件（`/api/upload/mri`）和同步推理（`/api/inference/run`）：120 秒（大文件慢网络需要更久）。
如果后端前面挂了网关或反向代理，代理侧的超时请设得比上面更长，否则会出现代理先断、前端还没超时的情况。

**时间格式**：ISO 8601 字符串，例如 `2026-09-28T21:00:00`。

**探活说明**：前端复用 `GET /api/models` 判断后端是否在线——这个接口返回 HTTP 2xx 即视为在线（返回空数组也算在线，代码只判断状态码）。切换在线模式、启动时会调用它。

---

## 2. 接口清单

下面 12 个接口是前端在线模式会调用的。前 10 个代码已实现；后 2 个标「建议补充」，是前端 UI 已规划但后端可本期一起做、不做也能跑通现有功能。

- `POST /api/upload/mri` — 上传 MRI 文件（前端函数 `uploadMriFile`），已实现
- `GET /api/models` — 模型列表，兼作探活（函数 `getAvailableModels`），已实现
- `POST /api/inference/run` — 同步推理（函数 `runInference`），已实现
- `GET /api/download/mask?path=` — 下载分割掩码（函数 `downloadMaskFile`），已实现
- `GET /api/history` — 历史记录列表（函数 `getHistory`），已实现
- `DELETE /api/history/{id}` — 删除历史（函数 `deleteHistory`），已实现
- `PATCH /api/history/{id}` — 重命名历史（函数 `renameHistory`），已实现
- `POST /api/jobs` — 提交异步任务（函数 `submitJob`），已实现
- `GET /api/jobs` — 异步任务列表（函数 `getJobs`），已实现
- `GET /api/jobs/{id}` — 查询任务状态（函数 `getJobStatus`），已实现
- `POST /api/jobs/{id}/cancel` — 取消异步任务，建议补充
- `GET /api/history/{id}` — 按 id 取单条历史，建议补充

所有路径前缀 `/api` 取决于上面"基础路径"的解析结果；开发期 Vite 代理把它转给 `localhost:8080`。

---

## 3. 接口详述

### 3.1 上传 MRI 文件
`POST /api/upload/mri`

请求：`multipart/form-data`，字段名 `file`（**必填**，单个文件，`.nii` 或 `.nii.gz`）。

响应 200：`{ "filePath": "/uploads/xxx.nii.gz" }`。`filePath` 是文件在服务端的存储路径，后续推理和下载都引用它。

错误：非 2xx → 前端提示「文件上传失败」（具体文案见 §1 错误规则）。

### 3.2 获取模型列表（兼探活）
`GET /api/models`

请求：无。

响应 200：`ModelInfo[]`（数组，结构见 §4.1）。返回 2xx 即视为后端在线（空数组也算在线）。

错误：非 2xx → 前端提示「获取模型列表失败」。

### 3.3 同步推理
`POST /api/inference/run`

请求体 `application/json`，字段如下：
- `mriFilePath`：**必填**，字符串，值为 3.1 上传接口返回的路径。
- `modality`：**必填**，字符串，取值必须是 `T1ce`、`FLAIR`、`T1`、`T2` 之一（BraTS 标准模态），后端请按这个值做校验。
- `modelType`：**可选**，字符串，缺省时由后端自选模型。取值与 `ModelInfo.modelType` 一致（如 `SEGRESNET_MAMBA`）。

请求示例：
```json
{ "mriFilePath": "/uploads/xxx.nii.gz", "modality": "T1ce", "modelType": "SEGRESNET_MAMBA" }
```

响应 200：`InferenceResult`（结构见 §4.2），示例：
```json
{
  "maskFilePath": "/outputs/xxx_mask.nii.gz",
  "modelUsed": "SEGRESNET_MAMBA",
  "executionTimeMs": 1234,
  "volumes": { "WT": 123.45, "TC": 45.6, "ET": 12.3 }
}
```

错误：非 2xx → 按 §1 规则，业务错误取 `message`，否则提示「推理失败」。

### 3.4 下载分割掩码
`GET /api/download/mask?path={encodeURIComponent(filePath)}`

请求：查询参数 `path`，**需 URL 编码**，值就是 `InferenceResult.maskFilePath`。

响应 200：`application/octet-stream` 二进制（`.nii.gz`），前端转成 Blob 后解析。

错误：非 2xx → 前端提示「下载分割结果失败」。

### 3.5 历史记录列表
`GET /api/history`

请求：无。

响应 200：`HistoryRecord[]`（结构见 §4.3）。

错误：非 2xx → 前端提示「获取历史记录失败」。

### 3.6 删除历史
`DELETE /api/history/{id}`

路径参数：`id`（整数主键）。

响应 204：无内容。

错误：非 2xx → 前端提示「删除历史记录失败」。

### 3.7 重命名历史
`PATCH /api/history/{id}`

路径参数：`id`（整数主键）。

请求体 `application/json`：`{ "fileName": "新名称" }`，`fileName` **必填且非空**（不含扩展名）。

响应 204：无内容。

错误：非 2xx → 前端提示「重命名历史记录失败（后端可能不支持该操作）」。如果后端暂未实现，返回 4xx 即可，前端会提示但不崩。

### 3.8 提交异步任务
`POST /api/jobs`

请求体：`InferenceRequest`（同 §3.3，三个字段规则一致）。

响应 200：`JobStatus`（结构见 §4.4，含 `jobId`）。

错误：非 2xx → 前端提示「提交任务失败」。

### 3.9 异步任务列表
`GET /api/jobs`

请求：无。

响应 200：`JobStatus[]`。

错误：非 2xx → 前端提示「获取任务列表失败」。

### 3.10 查询单个任务状态
`GET /api/jobs/{id}`

路径参数：`id`（任务 ID 字符串）。

响应 200：`JobStatus`。

错误：非 2xx → 前端提示「获取任务状态失败」。

### 3.11 取消异步任务（建议补充）
`POST /api/jobs/{id}/cancel`

说明：前端任务页目前只有「查看」没有「取消」，需要这个接口才能取消运行中的任务。

建议响应 200：`{ "jobId": "...", "status": "FAILED", "cancelled": true }`，或 204。

若后端暂不实现，任务页的取消按钮可先隐藏。

### 3.12 按 id 取单条历史（建议补充）
`GET /api/history/{id}`

说明：在线模式推理的历史存在后端库，前端点开历史详情时当前只从浏览器本地读（读不到会提示「无法回放」）。实现这个接口后，在线历史就能从后端拉取 mask 回放。

建议响应 200：`HistoryRecord`（含可下载的 `maskFilePath`）。

若后端暂不实现，在线历史详情仍只能看元数据、不能回放图像。

---

## 4. 公共类型定义（前后端共用字段）

以下 JSON 是各接口的请求/响应结构，字段名大小写必须一致。

**ModelInfo（模型信息）**
```jsonc
{
  "modelType": "SEGRESNET_MAMBA",   // 唯一标识，作请求参数
  "modelName": "SegResNet-Mamba",   // 显示名
  "modelVersion": "1.0.0",
  "description": "脑肿瘤分割模型",
  "supportedModalities": ["T1ce", "FLAIR", "T1", "T2"]
}
```

**InferenceResult（推理结果）**
```jsonc
{
  "maskFilePath": "/outputs/xxx_mask.nii.gz", // 下载接口用
  "modelUsed": "SEGRESNET_MAMBA",
  "executionTimeMs": 1234,
  "volumes": { "WT": 123.45, "TC": 45.6, "ET": 12.3 } // cm³，可空
}
```

**InferenceRequest（推理请求）**
```jsonc
{
  "mriFilePath": "/uploads/xxx.nii.gz", // 必填，由上传接口返回
  "modality": "T1ce",                   // 必填，枚举：T1ce / FLAIR / T1 / T2
  "modelType": "SEGRESNET_MAMBA"        // 可选，缺省后端自选
}
```

**HistoryRecord（历史记录）**
```jsonc
{
  "id": 12,
  "jobId": "job-abc" | null,           // 同步推理为 null
  "fileName": "patient_001",
  "modelType": "SEGRESNET_MAMBA",
  "modelUsed": "SegResNet-Mamba",
  "executionTimeMs": 1234,
  "volumes": { "WT": 123.45 } | null,
  "status": "COMPLETED",               // COMPLETED / FAILED
  "error": null,                       // 失败原因
  "createdAt": "2026-09-28T21:00:00"
}
```

**JobStatus（异步任务状态）**
```jsonc
{
  "jobId": "job-abc",
  "status": "QUEUED",  // QUEUED / RUNNING / COMPLETED / FAILED / TIMEOUT
  "progress": 45,      // 0-100
  "result": null,      // 完成后为 InferenceResult
  "error": null,
  "createdAt": "2026-09-28T21:00:00",
  "finishedAt": null   // 未完成为 null
}
```

---

## 5. 后端落地提示

1. 字段名必须一致：前端严格按上面大小写取字段（例如 `maskFilePath` 不是 `mask_path`）。改字段名需同步改 `src/types/index.ts` 与 `apiClient.ts`。
2. 错误提示怎么透传：想让用户看到自定义错误，请走 400 / 409 / 422 返回 `{ "message": "..." }`；401/403/404/5xx 的文案是前端写死的，后端 message 不会显示（详见 §1）。
3. 掩码文件是 `.nii.gz`：`volumes` 里的 `WT/TC/ET` 是 BraTS 标准三区体积（cm³），前端分析报告直接使用。
4. 两个建议接口（取消任务 / 按 id 取历史）不影响现有功能跑通，但决定「任务能否取消」「在线历史能否回放」两个交互是否完整，建议本期一起实现。
5. CORS：若前端直连 `http://IP:8080`（非 Vite 代理），后端需允许跨域；开发期走 Vite 代理（默认 `/api`）则无需。

---

## 6. 联调 curl 示例（给后端快速自检）

探活 + 取模型列表：
```bash
curl http://localhost:8080/api/models
```

上传 MRI 文件（字段名必须是 `file`）：
```bash
curl -F "file=@patient001.nii.gz" http://localhost:8080/api/upload/mri
```

同步推理（拿上面返回的文件路径填 mriFilePath）：
```bash
curl -X POST http://localhost:8080/api/inference/run \
  -H "Content-Type: application/json" \
  -d '{"mriFilePath":"/uploads/xxx.nii.gz","modality":"T1ce","modelType":"SEGRESNET_MAMBA"}'
```

拉历史记录、删历史、查任务：
```bash
curl http://localhost:8080/api/history
curl -X DELETE http://localhost:8080/api/history/12
curl http://localhost:8080/api/jobs
curl http://localhost:8080/api/jobs/job-abc
```

---

## 7. 离线模式说明（无后端，供参考）

背景：前端默认就是离线模式（`modeStore` 默认值 `offline`，选择记在 localStorage 键 `brainseg-mode`，刷新后保持）。离线时所有推理和历史存储都在浏览器内完成，完全不请求任何后端接口——也就是说，本文档前面 §2 的 12 个接口在离线模式下**一个都不会被调用**。前端一开始就默认离线，是为了避免在无后端时一打开就狂报 500。

**离线推理**：在浏览器里用 ONNX Runtime Web 跑模型。运行时在 `public/vendor/ort/`，模型在 `public/models/`（两个 `.onnx`）。用户上传 `.nii` / `.nii.gz` 后在本地完成分割，**MRI 数据和分割结果不会上传任何服务器**，纯本地处理。

**离线历史存储（三级路由，见 `historyGateway.ts`）**：
- 在线模式：走后端 `/api/history`。
- 离线 + 已连接本地文件夹：用 File System Access API（仅桌面 Chromium 系浏览器支持）把历史和原始文件写成真实文件，存在用户选的目录里。
- 离线 + 未连接文件夹：用 IndexedDB（浏览器内置数据库）兜底，默认就是这个。

**离线模式的几个差异点（后端同学需知）**：
- 离线没有"任务队列"概念。`JobService` 在离线时返回 `null`，任务页会明确提示「离线模式暂不支持任务队列」——因为本地推理是同步的、即时出结果，不需要异步排队。
- 离线历史详情能完整回放（数据就在本地 IndexedDB 或文件夹里）；而在线历史如果不实现 §3.12 的 `GET /api/history/{id}`，就只能看元数据、不能回放图像。
- 想测离线能力不需要任何后端：`npm run dev` 后默认就是离线，传一个脑 MRI 文件就能跑分割。

**给后端的结论**：如果部署环境能保证联网、且希望历史/任务统一由后端管理，就让用户切到在线模式并对接 §2 的接口；如果部署在断网或隐私敏感场景，离线模式可独立运行，无需后端配合，但此时历史只存在用户本机、不进后端库。
