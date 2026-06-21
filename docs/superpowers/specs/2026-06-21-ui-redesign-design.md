# UI 重设计规范：训练工作室与单模型推理页

## 背景与目标

基于现有 FastAPI + Alpine.js SPA（四页 scroll-snap 结构），对以下三处进行改造：

1. 训练工作室（第二页）不再要求用户上传图片，改为直接选择 `/data` 下已知数据集，并接入左上角全局数据集选择器。
2. 单模型推理页（第三页）Step 2 增加“自训练模型选择”分支，自训练模型不再混入左上角数据集下拉栏。
3. 单模型推理页（第三页）Step 3 只能从所选数据集的 `test/` 目录中选择图片，不再开放自由上传。

## 1. 整体架构与数据流变化

### 1.1 数据源统一

- 所有标准数据集均来自 `data_root`（`configs/config.yaml` 中的 `paths.data_root`，默认 `./data`）。
- 每个数据集目录为 MVTec AD 格式：`train/good/`、`test/<defect>/`、`ground_truth/<defect>/`。
- 训练仅使用 `train/good/`；评估使用完整 `test/`（含正常和异常样本），从而得到真实 AUROC/AUPR/F1。
- 自训练模型保存在 `results/{model}/{ModelName}/user/{category}/vX/`。

### 1.2 全局状态

- `app.selectedDataset`：当前选中的标准数据集类别（如 `bottle`）。
- `app.selfTrainedModels`：当前所有可用的自训练模型缓存。
- `app.availableDatasets`：标准数据集列表。

### 1.3 页面间状态同步

- 左上角数据集选择器、`training.selectedDataset`（第二页）、`inference.dataset`（第三页分支 A）三者双向同步。
- 第三页切换到“自训练模型”分支时，选择某个模型后，将其对应数据集回写到全局 `app.selectedDataset`。

## 2. 第二页（训练工作室）UI 重设计

### 2.1 布局

左侧区域：

- 当前数据集标题，与左上角选择器双向同步。
- 训练样本预览网格：只读显示 `train/good/` 下前 12 张缩略图（按文件名升序）。
- 训练样本计数（例如“训练样本：128 张”）。
- 保留“排除样本”能力：用户可在预览网格中勾选排除某些训练样本（仅影响训练，不删除文件）。

右侧区域：

- 算法选择（PatchCore / PaDiM / FRE / DRAEM）。
- 基础参数：epochs、batch_size、learning_rate、seed。
- 高级参数折叠面板（保持现有 `advanced_params`）。
- 训练按钮 + 实时监控曲线。

### 2.2 行为

- 进入第二页时，`training.selectedDataset` 从 `app.selectedDataset` 初始化。
- 在本页切换数据集时，立即写回 `app.selectedDataset`。
- 如果当前全局选中的是 `user/xxx`（历史遗留或异常状态），自动回退到第一个可用标准数据集，并在页面顶部以提示条形式告知用户。
- 点击训练后，前端发送 `dataset` 类别名，后端拼出完整 `data_path`。

### 2.3 后端接口调整

- `POST /api/train` 的 payload 从 `dataset_path` 改为 `dataset`：

  ```json
  {
    "model": "patchcore",
    "dataset": "bottle",
    "epochs": 1,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "seed": 42,
    "excluded_samples": [],
    "advanced_params": {}
  }
  ```

- 后端 `run_training_job()` 使用 `data_root / dataset` 构造路径，不再调用 `format_uploaded_samples()`。
- 训练完成后触发 `training-completed` 事件，刷新全局模型/数据集列表。

## 3. 第三页（单模型推理）Step 2 分支设计

### 3.1 三步流水线

- Step 1：选择算法（PatchCore / PaDiM / FRE / DRAEM）。
- Step 2：选择数据来源：
  - 分支 A：标准数据集（与全局选择器同步）。
  - 分支 B：自训练模型（仅列出当前算法下已训练模型）。
- Step 3：从对应数据集的 `test/` 目录中选择图片并推理。

### 3.2 分支 A：标准数据集

- 下拉列表与左上角全局数据集选择器共享状态。
- 选择后，`inference.dataset` 与 `app.selectedDataset` 同步。

### 3.3 分支 B：自训练模型

- 进入该分支时，调用 `GET /api/self-trained-models?model=patchcore` 加载列表。
- 每个选项显示：`{display_name} — {category} (v{version})`。
- 选择模型后，自动将模型对应的 `category` 同步到 `app.selectedDataset`。
- 模型路径示例：`results/patchcore/Patchcore/user/bottle/v1`。

### 3.4 Step 3：图片选择

- 只允许从对应数据集的 `test/` 目录中选择图片。
- 新增 `GET /api/test-images?dataset=bottle` 返回图片相对路径列表（按相对路径字母顺序排序，包含所有 `test/` 子目录中的图片）。
- UI 形式：下拉选择 + 缩略图预览（方案 C）。
- 默认选择排序后的第一张图片。
- 如果 `test/` 为空，禁用选择并提示。

### 3.5 后端接口调整

- 新增 `GET /api/self-trained-models`：

  ```json
  {
    "models": [
      {
        "path": "results/patchcore/Patchcore/user/bottle/v1",
        "category": "bottle",
        "version": 1,
        "display_name": "patchcore-custom-001"
      }
    ]
  }
  ```

- 新增 `GET /api/test-images`：

  ```json
  {
    "images": [
      "test/good/000.png",
      "test/scratch/001.png"
    ]
  }
  ```

- 修改 `POST /api/predict` payload：

  ```json
  {
    "model": "patchcore",
    "source": "pretrained",
    "dataset": "bottle",
    "image": "test/scratch/001.png"
  }
  ```

  或

  ```json
  {
    "model": "patchcore",
    "source": "self_trained",
    "self_trained_path": "results/patchcore/Patchcore/user/bottle/v1",
    "dataset": "bottle",
    "image": "test/scratch/001.png"
  }
  ```

## 4. 安全与边界情况

### 4.1 路径校验

- `dataset` 必须是 `data_root` 下已存在的目录名。
- `image` 必须解析到 `{data_root}/{dataset}/test/` 下的真实文件。
- `self_trained_path` 必须解析到 `results/{model}/{ModelName}/user/{category}/vX/` 下的有效 checkpoint。
- 拒绝 `..`、绝对路径、空字符串。

### 4.2 空状态

- 数据集 `test/` 为空：Step 3 禁用并提示“暂无测试图片”。
- 当前算法下无自训练模型：分支 B 禁用并提示“请先在训练工作室训练”。

### 4.3 训练并发

- 保持 `TrainingTaskManager` 单任务锁。
- 训练中禁用训练按钮。

### 4.4 状态清理

- 切换数据集时，重置 Step 3 图片为默认第一张。
- 切换算法时，若处于自训练分支，重新加载模型列表并清空无效选择。

## 5. 测试策略

### 5.1 后端测试

- 更新 `tests/test_training_api.py`：
  - `/api/train` 使用标准数据集路径启动训练（mock trainer）。
  - `/api/self-trained-models` 正确扫描自训练目录。
  - `/api/test-images` 只返回 `test/` 下图片并拒绝越界路径。
  - `/api/predict` 拒绝 `test/` 外图片路径。

### 5.2 前端交互测试

- 第二页切换数据集，左侧预览同步变化。
- 第二页训练后，第三页分支 B 出现新模型。
- 第三页分支 A 从标准数据集 test/ 选图推理。
- 第三页分支 B 选择自训练模型并选图推理。

### 5.3 回归测试

- 运行 `python -m pytest tests/ -v`，确保现有测试通过。
- 验证第四页（四模型对比）不受影响。
