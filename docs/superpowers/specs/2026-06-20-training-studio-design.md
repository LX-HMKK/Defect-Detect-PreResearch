# Training Studio 设计文档

> **日期**: 2026-06-20  
> **主题**: FastAPI + Alpine.js SPA 新增“训练工作室”页面，实现本地上传训练样本、前端启动训练、实时 loss/AUROC 监控。  
> **状态**: 待实现  

---

## 1. 背景与目标

### 1.1 缺口来源
对照《综合项目任务书》第四章“可视化验证平台开发”与第六章验收标准，当前 Phase 2 UI（FastAPI + Alpine.js SPA）已实现：
- 单模型推理与结果可视化
- 四模型并行对比
- 亮/暗主题切换

但**缺失以下验收功能**：
1. 训练样本选择与管理（上传、预览、筛选）
2. 模型训练模块（选择算法、配置超参数、启动训练、实时监控 loss/学习率/AUROC）
3. 批量图像推理入口

### 1.2 本设计范围
本设计聚焦补齐前两项，形成 **Training Studio** 独立页面：
- **训练样本管理**：支持本地上传正常样本图片，自动格式化为 MVTec AD 目录结构，展示样本画廊。默认使用全部上传样本训练；用户可在画廊中反选排除部分样本，受 150 张上限约束。
- **训练配置**：选择算法（PatchCore / PaDiM / FRE / DRAEM）、设置 epoch / batch size / learning rate / seed。
- **训练启动与监控**：点击“开始训练”后通过 SSE 实时推送 loss、learning rate、val_image_AUROC、当前 epoch、ETA；训练完成后自动保存 checkpoint 并刷新模型列表。
- **批量推理入口**：训练完成后在监控卡显示跳转按钮，携带模型/数据集信息到 Section 2 推理页。**本设计仅做跳转触发，批量推理 UI 本身不在本设计范围内。**

**不在本设计范围内**：
- 超参数自动搜索 / AutoML
- 分布式多 GPU 训练
- 模型版本管理与回滚
- 用户权限与多租户

---

## 2. 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 页面形式 | **新增独立全屏 snap 页** | 与任务书“模型训练模块”一一对应，演示动线最清晰。 |
| 布局 | **左侧配置浮卡 + 右侧工作区** | 与现有 Apple Cinematic Pro 设计语言一致，信息层次清晰。 |
| 样本来源 | **本地上传 + 自动格式化** | 满足任务书“从本地加载训练图像”要求，且无需改动现有 `data/` 目录。 |
| 数字输入 | **简洁圆角输入框** | 避免步进器/下拉组合的杂乱感，用户可直接输入。 |
| 训练通信 | **SSE 流式推送** | 与现有 `/api/predict`、`/api/compare` 一致，避免 WebSocket 复杂度。 |
| 后端执行 | **asyncio.to_thread + 线程池** | 与现有推理端点一致，避免阻塞事件循环。 |
| 训练引擎 | **复用 `AnomalyDetectionTrainer`** | 保证训练逻辑与命令行脚本一致，减少重复代码。 |

---

## 3. 页面结构

### 3.1 四页 snap 结构

当前为 3 页，新增训练页后变为 4 页：

| 页码 | id | 内容 |
|------|-----|------|
| 1/4 | `#s0` | Hero / 算法介绍（现有） |
| 2/4 | `#s1-training` | **Training Studio（新增）** |
| 3/4 | `#s2` | 单模型推理（现有，页码后移） |
| 4/4 | `#s3` | 四模型对比（现有，页码后移） |

需同步修改：
- `index.html` 中 `.snap-dots` 圆点数量（3 → 4）
- `app.js` 中 `sectionCount`（3 → 4）
- 键盘 ↑↓ 导航范围
- 右侧进度环标签

### 3.2 训练页布局

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING STUDIO                                            │
│  训练你的检测模型                                            │
├──────────────────┬──────────────────────────────────────────┤
│                  │                                          │
│  [配置卡]        │  [样本画廊卡]                            │
│  ─────────────   │  已选择 N 张  [全选][清空]               │
│  算法 pill 标签   │  ┌────┐┌────┐┌────┐┌────┐...           │
│  拖拽上传区       │  │img ││img ││img ││img │              │
│  参数输入框       │  └────┘└────┘└────┘└────┘              │
│  开始训练按钮     │                                          │
│                  │  [实时监控卡]                            │
│                  │  Loss 曲线区        Epoch   val AUROC    │
│                  │                     12/100    0.87       │
│                  │                                          │
└──────────────────┴──────────────────────────────────────────┘
```

### 3.3 组件清单

| 组件 | 位置 | 职责 |
|------|------|------|
| `AlgorithmSelector` | 配置卡 | 药丸标签单选 4 种算法 |
| `SampleUploader` | 配置卡 | 拖拽/点击上传，显示 150 张上限计数 |
| `TrainingConfigForm` | 配置卡 | epoch / batch / lr / seed 输入 |
| `TrainingStartButton` | 配置卡 | 启动训练，训练中变“停止训练” |
| `SampleGallery` | 右侧上 | 缩略图网格，支持勾选与删除 |
| `TrainingMonitor` | 右侧下 | 实时曲线 + 大字号指标 |
| `TrainingHistoryStrip` | 页面底部 | 最近训练任务缩略状态 |

---

## 4. 视觉与交互细节

### 4.1 配色与材质
延续现有 Phase 2 Apple 风格：
- 背景：深色渐变 + 环境光呼吸
- 卡片：`rgba(255,255,255,0.06)` + `backdrop-filter: blur(20px)` + 1px 边缘高光
- 主按钮：蓝渐变 `#2997ff → #5ac8fa`，药丸圆角
- 状态徽章：绿色脉冲点“训练中”，蓝色“就绪”
- 输入框：深色填充 `rgba(0,0,0,0.2)`，圆角 12px，聚焦 1px 高光边框

### 4.2 算法标签颜色
沿用现有算法色标：
- PatchCore: `#2997ff`
- PaDiM: `#30d158`
- FRE: `#ff9f0a`
- DRAEM: `#bf5af2`

### 4.3 训练状态机

```
idle ──上传样本──▶ uploaded ──点击开始──▶ training ──完成──▶ completed
                    │                       │
                    └────────停止───────────┘
                                        error
```

- `idle`：未上传或已清空
- `uploaded`：有样本但未训练
- `training`：训练进行中，显示进度与曲线
- `completed`：训练完成，可跳转推理
- `error`：训练失败，显示错误信息

---

## 5. 数据流

### 5.1 用户操作流程

1. 用户进入 Training Studio 页
2. 选择算法（默认 PatchCore）
3. 拖拽或点击上传训练样本图片
4. 后端保存到临时目录并按 MVTec AD 结构格式化
5. 前端展示样本画廊，用户可勾选/删除
6. 调整训练参数（默认从 `configs/config.yaml` 读取）
7. 点击“开始训练”
8. 后端启动训练线程，SSE 推送实时指标
9. 训练完成：保存 checkpoint，计算最优阈值，刷新模型列表
10. 用户可跳转 Section 2 进行推理

### 5.2 后端训练流程

```
/api/train (POST, SSE)
    │
    ▼
asyncio.to_thread(_run_training)
    │
    ▼
创建临时数据集目录
    │
    ▼
AnomalyDetectionTrainer(config_path, data_path, category, model_name)
    │
    ▼
trainer.setup()  → datamodule + model
    │
    ▼
trainer.train(max_epochs)
    │  └── Engine.fit() 过程中通过回调 yield 指标
    ▼
trainer.evaluate()  → 计算阈值 + 保存 JSON
    │
    ▼
返回 completed 事件
```

### 5.3 关键约束
- **GPU 串行**：一次只能跑一个训练任务。若已有训练在进行中，后端返回 `409 Conflict`。
- **样本上限**：上传时若超过 150 张，前端提示；后端再次校验。
- **临时目录隔离**：每次上传生成 `uploads/training_<uuid>/`，避免与现有数据集混淆。
- **训练可中断**：提供停止训练按钮，通过 `trainer.engine` 或进程标志位实现优雅中断。

---

## 6. API 设计

### 6.1 新增端点

#### `POST /api/upload-samples`
上传训练样本图片，返回格式化后的临时数据集信息。

**请求**: `multipart/form-data`
```
files: List[UploadFile]  # 图片文件
model: str                # 可选，用于组织目录
```

**响应**:
```json
{
  "session_id": "uuid",
  "dataset_path": ".cache/uploads/training_uuid",
  "category": "training_uuid",
  "total": 120,
  "max_allowed": 150,
  "samples": ["img_001.png", "img_002.png", ...]
}
```

#### `POST /api/train` (SSE)
启动训练，返回 SSE 流。

**请求体**:
```json
{
  "model": "patchcore",
  "dataset_path": ".cache/uploads/training_uuid",
  "category": "training_uuid",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.0001,
  "seed": 42
}
```

**SSE 事件类型**:
- `status`: 训练状态变化
- `metric`: 训练指标（loss, AUROC 等）
- `log`: 训练日志文本
- `completed`: 训练完成，含结果摘要
- `error`: 训练错误

#### `POST /api/train/stop`
停止当前训练任务。

**响应**:
```json
{ "status": "stopped" }
```

#### `GET /api/train-status`
查询当前是否有训练任务在运行。

**响应**:
```json
{
  "running": true,
  "model": "patchcore",
  "dataset": "training_uuid",
  "started_at": "2026-06-20T14:30:00",
  "current_epoch": 12,
  "total_epochs": 100
}
```

### 6.2 复用/扩展现有端点
- `GET /api/models`：返回可用模型与数据集，训练完成后需刷新。
- `/api/predict`、`/api/compare`：训练完成后模型列表应包含新训练的 checkpoint。

---

## 7. SSE 协议

### 7.1 事件格式

```
event: metric
data: {"epoch": 12, "train_loss": 0.0342, "learning_rate": 0.0001, "val_image_AUROC": 0.8712, "eta_seconds": 120}

event: log
data: {"message": "Epoch 12/100 - train_loss: 0.0342", "level": "info"}

event: completed
data: {"model": "patchcore", "dataset": "training_uuid", "image_AUROC": 0.98, "pixel_AUROC": 0.95, "checkpoint_path": "results/patchcore/training_uuid/latest.ckpt"}

event: error
data: {"message": "CUDA out of memory", "code": "OOM"}
```

### 7.2 前端消费
沿用现有 `InferenceRunner` / `CompareRunner` 模式，新增 `TrainingRunner`：
- 建立 EventSource
- 解析 event/data 行
- 更新 Alpine `trainingState`
- 实时绘制曲线（可用 SVG 或 lightweight chart 库）
- 完成/错误时关闭连接并更新 UI

---

## 8. 后端实现细节

### 8.1 复用 `AnomalyDetectionTrainer`
训练核心继续由 `modules/algorithm/trainer.py` 的 `AnomalyDetectionTrainer` 承担，确保与命令行脚本 `scripts/run_training.py` 行为一致。

需要新增/调整：
- 支持传入自定义 `data_path` 指向临时上传目录
- 训练过程中通过回调或日志捕获实时指标
- 支持优雅停止（stop flag）
- 训练完成后将结果保存到 `results/<model>/<category>/`，与现有命令行训练输出目录一致，使 `/api/models` 与 `demo.py` 的模型发现逻辑无需修改即可识别新模型

### 8.1.1 模型可见性
Training Studio 训练得到的 checkpoint 必须能被 Section 2 单模型推理和 Section 3 四模型对比使用。实现方式：
- 输出目录沿用现有约定 `results/<model>/<category>/`
- 结果 JSON 文件命名为 `<model>_<category>_results.json`
- 最优阈值写入 JSON，供 UI 在推理时直接使用
- 启动 UI 时 `get_available_datasets()` 和 `MODEL_CONFIGS` 扫描 `results/` 目录，自动包含新训练的数据集

### 8.2 训练回调与指标捕获
使用 PyTorch Lightning 回调在 `on_train_epoch_end` 和 `on_validation_epoch_end` 时收集指标：
- `train_loss`
- `learning_rate`
- `val_image_AUROC`
- `val_pixel_AUROC`（若可用）
- 当前 epoch
- 预估剩余时间 ETA

回调将指标写入线程安全的 `queue.Queue`；SSE 生成器在 `_run_training` 线程中从队列读取并 `yield` 事件。前端通过 `EventSource` 接收并更新曲线与数字指标。

### 8.3 临时目录与清理
- 上传目录：`.cache/uploads/training_<uuid>/`
- 训练结果目录：`results/<model>/training_<uuid>/`
- 完成训练后保留 checkpoint 和 JSON，上传源文件可配置保留时间（默认保留）

### 8.4 单任务锁
使用全局 `asyncio.Lock` 或线程事件标志，确保同时只有一个训练任务在 GPU 上运行：
- 新训练请求若检测到已有任务运行，返回 `409` 并提示“已有训练任务进行中”。
- 提供 `/api/train-status` 查询当前训练状态。

---

## 9. 错误处理

| 错误场景 | 前端表现 | 后端处理 |
|----------|----------|----------|
| 上传非图片文件 | 提示“请上传图片文件” | FastAPI 校验 MIME type |
| 上传超过 150 张 | 提示并阻止继续上传 | 后端校验总数 |
| GPU 已被占用 | 提示“已有训练任务进行中” | 返回 409 |
| CUDA OOM | SSE `error` 事件 + 释放显存 | try/except + torch.cuda.empty_cache() |
| 训练中断 | 状态回到 `uploaded`，保留已训练权重 | 优雅停止 |
| 无样本启动训练 | 按钮禁用 + 提示“请先上传样本” | 前端校验 |
| 参数非法 | 输入框红色高亮 | 前端 + 后端双重校验 |

---

## 10. 与现有代码集成

### 10.1 修改文件
- `modules/ui/static/index.html`
  - 新增 `#s1-training` section
  - 调整 snap-dots 数量为 4
  - 引入新 JS/CSS 资源
- `modules/ui/static/js/app.js`
  - `totalPages` 3 → 4
  - 新增 `trainingState` Alpine store
- `modules/ui/static/js/training.js`（新增）
  - `TrainingRunner`
  - 样本上传逻辑
  - 训练状态机
  - 实时曲线绘制
- `modules/ui/static/css/app.css`
  - 新增训练页相关样式
- `modules/ui/server.py`
  - 新增 `/api/upload-samples`
  - 新增 `/api/train` SSE
  - 新增 `/api/train/stop`
  - 新增 `/api/train-status`

### 10.2 新增文件
- `modules/ui/static/js/training.js`
- 可选：`modules/ui/static/js/chart.js`（轻量曲线绘制）

### 10.3 复用组件
- 算法色标 CSS 变量
- 现有玻璃质感卡片样式（`backdrop-filter: blur(...)`）
- SSE 解析工具 `InferenceRunner` 模式
- `AnomalyDetectionTrainer`
- `MVTecFormatter`（用于格式化上传样本）

---

## 11. 测试计划

### 11.1 单元测试
- `tests/test_training_api.py`（新增）
  - 上传样本 API 校验
  - 训练参数校验
  - SSE 事件解析

### 11.2 集成测试
- 端到端：上传 → 启动训练 → 接收 SSE → 完成 → 检查 checkpoint 与 JSON
- 使用 `pytest` + 小数据集（5 张图 + 1 epoch）加速

### 11.3 UI 测试
- 检查训练页是否在 snap 导航中正确显示
- 检查按钮状态机
- 检查 SSE 断开后 UI 不卡死

### 11.4 回归测试
- 确保原有 3 页推理/对比功能不受影响
- 确保亮/暗主题切换覆盖新增元素

---

## 12. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 训练任务耗时长，用户等待焦虑 | 体验差 | SSE 实时推送 + ETA 显示 + 可中断 |
| GPU 串行导致并发训练请求失败 | 用户困惑 | 全局任务锁 + 清晰提示 |
| 大量图片上传导致内存/磁盘压力 | 系统卡死 | 单文件大小限制 + 总数限制 + 异步保存 |
| 实时曲线绘制性能差 | 页面卡顿 | 限制数据点数量，使用 CSS/SVG 而非 Canvas 重绘 |
| 训练中断后状态不一致 | 模型不可用 | 优雅停止 + 状态回滚 |

---

## 13. 后续可扩展

- 批量推理入口：训练完成后在监控卡显示“用此模型批量推理”按钮，跳转 Section 2 并预填模型。
- 训练历史：持久化训练记录到 JSON/数据库，支持复跑。
- 超参数预设：为不同算法提供推荐参数模板。
- 早停可视化：在监控区显示早停触发线。

---

## 14. 附录：默认参数来源

各算法默认训练参数从 `configs/config.yaml` 读取：

| 参数 | 默认值来源 |
|------|-----------|
| epoch | `training.epochs.{model}` |
| batch_size | `data.train_batch_size`（DRAEM 用 `data.draem_batch_size`） |
| learning_rate | `training.optimizer.lr` |
| seed | `training.seed` |
| image_size | `data.image_size` |

---

> **下一步**: 本设计文档经审阅批准后，进入 `superpowers:writing-plans` 生成详细实现计划。
