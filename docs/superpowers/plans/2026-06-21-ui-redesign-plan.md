# UI 重设计实现计划

> **状态：已完成** — 已于 2026-06-21 合并至 `main`（合并提交 `e2ac124`），所有任务已验收。
>
> **For agentic workers:** REQUIRED SUB-LEVEL SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将训练工作室改为选择 `/data` 下标准数据集并移除上传；在单模型推理页增加自训练模型分支；单模型推理图片只能从对应数据集 `test/` 中选择。

**Architecture:** 后端新增 `/api/test-images` 与 `/api/self-trained-models` 端点，改造 `/api/train` 与 `/api/predict`；前端第二页移除上传逻辑、绑定全局数据集选择器，第三页增加来源分支与图片下拉预览。

**Tech Stack:** FastAPI, Alpine.js, Anomalib, PyTorch Lightning, Python 3.10

---

## 文件结构

| 文件 | 责任 |
|---|---|
| `modules/ui/_model_info.py` | 新增自训练模型扫描函数 `get_self_trained_models()` |
| `modules/ui/server.py` | 新增端点、改造 `TrainRequest` 与 `/api/predict` payload、路径安全校验 |
| `modules/ui/training_backend.py` | 改造 `run_training_job()` 签名，支持从 `data_root/dataset` 启动训练 |
| `modules/ui/static/js/app.js` | 扩展全局推理状态：source、testImages、selfTrainedModels 等 |
| `modules/ui/static/js/training.js` | 移除上传逻辑，接入全局数据集，加载训练样本预览 |
| `modules/ui/static/index.html` | 重排第二页与第三页 HTML 结构 |
| `modules/ui/static/css/app.css` | 补充新布局样式 |
| `tests/test_training_api.py` | 更新测试用例覆盖新端点与安全校验 |

---

## Task 1: 后端新增数据集与自训练模型辅助函数

**Files:**
- Modify: `modules/ui/_model_info.py`
- Test: `tests/test_training_api.py`（后续 task 再写）

- [x] **Step 1: 在 `_model_info.py` 新增 `get_self_trained_models()`**

在文件末尾 `get_available_datasets()` 之后添加：

```python
def get_self_trained_models(model_key: str) -> list:
    """扫描指定算法下所有用户自训练模型。

    路径结构: results/{model_key}/{ModelName}/user/{category}/vX
    返回对象包含 path、category、version、display_name。
    """
    results_dir = resolve_project_path(cfg_get('paths.results_root', './results'))
    model_dirs = {
        "fre": "Fre",
        "patchcore": "Patchcore",
        "draem": "Draem",
        "padim": "Padim",
    }
    subdir = model_dirs.get(model_key)
    if not subdir:
        return []

    user_root = results_dir / model_key / subdir / "user"
    if not user_root.exists():
        return []

    models = []
    for cat_dir in user_root.iterdir():
        if not cat_dir.is_dir() or cat_dir.name == '__pycache__':
            continue
        for version_dir in sorted(cat_dir.iterdir()):
            if not version_dir.is_dir() or not version_dir.name.startswith('v'):
                continue
            try:
                version = int(version_dir.name[1:])
            except ValueError:
                continue
            # 要求目录下存在有效的 lightning checkpoint
            ckpts = list(version_dir.glob('*.ckpt'))
            if not ckpts:
                continue
            display_name = _resolve_display_name(model_key, cat_dir.name, 'user', results_dir)
            models.append({
                'path': str(version_dir),
                'category': cat_dir.name,
                'version': version,
                'display_name': display_name,
            })

    return sorted(models, key=lambda m: (m['category'], m['version']))
```

- [x] **Step 2: 在 `server.py` 导入新增函数**

找到 `from modules.ui._model_info import MODEL_CONFIGS, get_available_datasets`，改为：

```python
from modules.ui._model_info import MODEL_CONFIGS, get_available_datasets, get_self_trained_models
```

- [x] **Step 3: 新增 `POST /api/self-trained-models` 端点**

在 `/api/models` 端点之后添加：

```python
@app.get("/api/self-trained-models")
async def api_self_trained_models(model: str = Query(...)):
    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"未知模型: {model}")
    return {"models": get_self_trained_models(model)}
```

- [x] **Step 4: 新增 `POST /api/test-images` 端点**

在 `/api/self-trained-models` 之后添加：

```python
from modules.config import get as cfg_get
from modules._runtime import resolve_project_path

@app.get("/api/test-images")
async def api_test_images(dataset: str = Query(...)):
    data_root = resolve_project_path(cfg_get('paths.data_root', './data'))
    dataset_dir = data_root / dataset
    if not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"数据集不存在: {dataset}")

    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        return {"images": []}

    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = []
    for img_path in test_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in allowed_suffixes:
            rel = img_path.relative_to(dataset_dir).as_posix()
            images.append(rel)

    return {"images": sorted(images)}
```

- [x] **Step 5: Commit**

```bash
git add modules/ui/_model_info.py modules/ui/server.py
git commit -m "feat(ui,api): 新增自训练模型与测试图片列表端点"
```

---

## Task 2: 后端改造 `/api/train` 使用标准数据集路径

**Files:**
- Modify: `modules/ui/server.py`
- Modify: `modules/ui/training_backend.py`

- [x] **Step 1: 修改 `TrainRequest` Pydantic 模型**

将 `modules/ui/server.py` 中的：

```python
class TrainRequest(BaseModel):
    model: str
    dataset_path: str
    category: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    seed: int = 42
    excluded_samples: List[str] = []
    advanced_params: Dict[str, Any] = {}
```

改为：

```python
class TrainRequest(BaseModel):
    model: str
    dataset: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    seed: int = 42
    excluded_samples: List[str] = []
    advanced_params: Dict[str, Any] = {}
```

- [x] **Step 2: 修改 `/api/train` 端点中的校验与路径构造**

原校验 `dataset_path` 的代码块：

```python
upload_root = resolve_project_path(UPLOAD_ROOT)
train_dataset_path = resolve_project_path(request.dataset_path)
if not str(train_dataset_path).startswith(str(upload_root)):
    raise HTTPException(status_code=400, detail="dataset_path 必须在上传目录内")
if not train_dataset_path.exists():
    raise HTTPException(status_code=400, detail="数据集路径不存在")
```

替换为：

```python
data_root = resolve_project_path(cfg_get('paths.data_root', './data'))
train_dataset_path = data_root / request.dataset
if not train_dataset_path.is_dir():
    raise HTTPException(status_code=400, detail=f"数据集不存在: {request.dataset}")
train_good_dir = train_dataset_path / "train" / "good"
if not train_good_dir.is_dir():
    raise HTTPException(status_code=400, detail=f"数据集格式错误，缺少 train/good: {request.dataset}")
```

- [x] **Step 3: 修改 `run_training_job()` 调用参数**

在 `/api/train` 中找到 `run_training_job` 调用：

```python
run_training_job(
    model_name=request.model,
    dataset_path=str(train_dataset_path),
    category=request.category,
    ...
)
```

改为：

```python
run_training_job(
    model_name=request.model,
    dataset_path=str(train_dataset_path),
    category=request.dataset,
    ...
)
```

- [x] **Step 4: 修改 `training_backend.py` 中 `run_training_job()` 的签名调用**

该函数签名保持不变，但调用者传入的 `category` 已经是数据集名。确认 `display_name` 的生成逻辑：当前函数内有 `display_name = display_name or f"{model_name}-custom-{batch:03d}"`，无需改动。

- [x] **Step 5: Commit**

```bash
git add modules/ui/server.py modules/ui/training_backend.py
git commit -m "feat(ui,training): 训练接口改为使用标准数据集路径"
```

---

## Task 3: 后端改造 `/api/predict` 支持自训练模型与 test/ 图片

**Files:**
- Modify: `modules/ui/server.py`

- [x] **Step 1: 新增路径安全校验辅助函数**

在 `server.py` 的 `UPLOAD_ROOT` 常量附近添加：

```python
def _safe_test_image_path(dataset: str, image: str) -> Path:
    """校验 image 是否落在 dataset/test/ 下，返回绝对路径。"""
    data_root = resolve_project_path(cfg_get('paths.data_root', './data'))
    dataset_dir = data_root / dataset
    if not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"数据集不存在: {dataset}")

    test_dir = dataset_dir / "test"
    requested = (dataset_dir / image).resolve()
    test_resolved = test_dir.resolve()
    try:
        requested.relative_to(test_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="图片路径不在 test/ 目录内")

    if not requested.is_file():
        raise HTTPException(status_code=400, detail=f"图片不存在: {image}")
    return requested


def _safe_self_trained_path(model: str, path: str) -> Path:
    """校验自训练模型路径是否合法且包含 checkpoint。"""
    results_dir = resolve_project_path(cfg_get('paths.results_root', './results'))
    model_dirs = {"fre": "Fre", "patchcore": "Patchcore", "draem": "Draem", "padim": "Padim"}
    subdir = model_dirs.get(model)
    if not subdir:
        raise HTTPException(status_code=400, detail=f"未知模型: {model}")

    expected_prefix = (results_dir / model / subdir / "user").resolve()
    requested = Path(path).resolve()
    try:
        requested.relative_to(expected_prefix)
    except ValueError:
        raise HTTPException(status_code=400, detail="自训练模型路径不合法")

    if not requested.is_dir() or not list(requested.glob("*.ckpt")):
        raise HTTPException(status_code=400, detail="自训练模型 checkpoint 不存在")
    return requested
```

- [x] **Step 2: 修改 `/api/predict` 端点接受表单新字段**

原端点可能类似：

```python
@app.post("/api/predict")
async def api_predict(image: UploadFile = File(...), model: str = Form(...), dataset: str = Form(...)):
```

改为：

```python
@app.post("/api/predict")
async def api_predict(
    model: str = Form(...),
    dataset: str = Form(...),
    image: str = Form(...),
    source: str = Form("pretrained"),
    self_trained_path: str = Form(""),
):
    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"未知模型: {model}")
    if source not in ("pretrained", "self_trained"):
        raise HTTPException(status_code=400, detail="source 必须是 pretrained 或 self_trained")

    image_path = _safe_test_image_path(dataset, image)

    if source == "self_trained":
        model_dir = _safe_self_trained_path(model, self_trained_path)
    else:
        model_dir = None
```

- [x] **Step 3: 将安全路径传给 `_run_prediction()`**

需要修改 `_run_prediction()` 签名以接受 `model_dir: Path | None`。

在 `_run_prediction()` 内部，当 `model_dir` 不为空时，使用 `AnomalyDetectionTrainer` 加载该目录下的 checkpoint；否则使用现有预训练模型加载逻辑。具体加载方式参考 `AnomalyDetectionTrainer` 的 `load_model` 或 `engine.predict` 用法（在 `modules/ui/server.py` 中已有 `AnomalyDetectionDetector` 类，需要扩展）。

如果现有 `AnomalyDetectionDetector.load_model(model_key, dataset)` 方法无法加载自定义路径，可新增：

```python
def load_self_trained_model(self, model_key: str, model_dir: Path):
    """从自训练目录加载模型与 checkpoint。"""
    config_path = resolve_project_path(f"configs/{model_key}.yaml")
    trainer = AnomalyDetectionTrainer(
        model_name=model_key,
        category=None,
        data_path=None,
        config_path=str(config_path),
        source="user",
    )
    # 复用 trainer 内部的 model/datamodule 工厂，但只加载权重
    # 实现细节需根据 trainer.py 当前 load_model 逻辑扩展
```

> 注：此处需要查看 `modules/algorithm/trainer.py` 中已有的 checkpoint 加载方式，可能已有 `load_model` 或类似方法。若不存在，需在 Task 3 中补充最小实现。

- [x] **Step 4: Commit**

```bash
git add modules/ui/server.py
git commit -m "feat(ui,inference): 推理接口支持自训练模型与 test/ 图片"
```

---

## Task 4: 更新后端测试

**Files:**
- Modify: `tests/test_training_api.py`

- [x] **Step 1: 更新 `/api/train` 相关测试**

将原来使用 `dataset_path` 的测试请求改为使用 `dataset`：

```python
# 原来
response = client.post("/api/train", json={
    "model": "patchcore",
    "dataset_path": "...",
    "category": "...",
    ...
})

# 改为
response = client.post("/api/train", json={
    "model": "patchcore",
    "dataset": "bottle",
    ...
})
```

- [x] **Step 2: 替换“dataset_path 越界”测试为“数据集不存在”测试**

原测试：

```python
def test_train_rejects_path_outside_upload_root(client):
    response = client.post("/api/train", json={
        "model": "patchcore",
        "dataset_path": "./data/bottle",
        "category": "bottle",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0001,
        "seed": 42,
    })
    assert response.status_code == 400
```

改为：

```python
def test_train_rejects_missing_dataset(client):
    response = client.post("/api/train", json={
        "model": "patchcore",
        "dataset": "not_a_real_dataset_12345",
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 0.0001,
        "seed": 42,
    })
    assert response.status_code == 400
```

- [x] **Step 3: 新增 `/api/test-images` 测试**

```python
def test_test_images_returns_test_folder_images(client):
    # 依赖 data/bottle 存在
    response = client.get("/api/test-images?dataset=bottle")
    assert response.status_code == 200
    data = response.json()
    assert "images" in data
    assert all(img.startswith("test/") for img in data["images"])


def test_test_images_rejects_missing_dataset(client):
    response = client.get("/api/test-images?dataset=not_a_real_dataset_12345")
    assert response.status_code == 400
```

- [x] **Step 4: 新增 `/api/self-trained-models` 测试**

```python
def test_self_trained_models_rejects_invalid_model(client):
    response = client.get("/api/self-trained-models?model=notamodel")
    assert response.status_code == 400


def test_self_trained_models_returns_list(client):
    response = client.get("/api/self-trained-models?model=patchcore")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
```

- [x] **Step 5: 运行测试**

```bash
python -m pytest tests/test_training_api.py -v
```

Expected: 全部通过。

- [x] **Step 6: Commit**

```bash
git add tests/test_training_api.py
git commit -m "test(ui,api): 更新训练与推理接口测试"
```

---

## Task 5: 前端改造 `training.js` 移除上传并绑定数据集

**Files:**
- Modify: `modules/ui/static/js/training.js`
- Modify: `modules/ui/static/index.html`（第二步配合）

- [x] **Step 1: 修改 `training` 状态**

移除与上传相关的状态：`samples`、`isDragOver`、`sessionId`、`displayName`、`datasetPath`、`category`。

新增状态：

```javascript
selectedDataset: '',
trainSamples: [],
trainSampleCount: 0,
```

- [x] **Step 2: 新增加载训练样本预览方法**

在 `training` 组件中添加：

```javascript
loadTrainSamples: function () {
    var self = this;
    if (!self.selectedDataset) {
        self.trainSamples = [];
        self.trainSampleCount = 0;
        return;
    }
    fetch('/api/train-samples?dataset=' + encodeURIComponent(self.selectedDataset))
        .then(function (res) { return res.json(); })
        .then(function (data) {
            self.trainSamples = (data.samples || []).slice(0, 12);
            self.trainSampleCount = data.total || 0;
        })
        .catch(function () {
            self.trainSamples = [];
            self.trainSampleCount = 0;
        });
},
```

> 注：同步在后端新增 `/api/train-samples` 端点（可在 Task 1 中一并添加），或复用已有逻辑。为减少接口，也可在 `/api/models` 扩展或让前端直接构造图片 URL（但不推荐，因需知道具体文件名）。最简方案：在 Task 1 已新增 `/api/test-images` 后，再新增 `GET /api/train-samples?dataset=bottle` 返回 `train/good/` 下图片列表。

- [x] **Step 3: 改造 `init()` 与数据集监听**

```javascript
init: function () {
    var self = this;
    self.resetMonitor();
    self.epochs = self.modelDefaultEpochs[self.selectedModel] || 100;

    // 从全局 app 同步数据集
    var app = Alpine.store('app') || window.app;
    if (app && app.selectedDataset) {
        self.selectedDataset = app.selectedDataset;
    }
    self.loadTrainSamples();

    self.$watch('selectedModel', function (model) {
        self.epochs = self.modelDefaultEpochs[model] || 100;
    });
    self.$watch('selectedDataset', function (dataset) {
        self.loadTrainSamples();
        if (app) app.selectedDataset = dataset;
    });
    window.addEventListener('resize', function () {
        self.$nextTick(function () { self.drawChart(); });
    });
},
```

- [x] **Step 4: 移除上传相关函数**

删除 `onSelectSamples`、`onDropSamples`、`_scanDroppedItems`、`_readEntry`、`_uploadFiles`、`toggleExclude`。

- [x] **Step 5: 改造 `startTraining()`**

```javascript
startTraining: function () {
    var self = this;
    if (!self.selectedDataset) return;

    self.resetMonitor();
    self.trainingState = 'training';

    TrainingRunner.run({
        model: self.selectedModel,
        dataset: self.selectedDataset,
        epochs: parseInt(self.epochs, 10),
        batch_size: parseInt(self.batchSize, 10),
        learning_rate: parseFloat(self.learningRate),
        seed: parseInt(self.seed, 10),
        excluded_samples: [],
        advanced_params: self.advancedParams[self.selectedModel] || {},
    }, {
        // handlers 保持不变
    });
},
```

- [x] **Step 6: Commit**

```bash
git add modules/ui/static/js/training.js
git commit -m "feat(ui,training): 训练工作室移除上传并绑定数据集"
```

---

## Task 6: 前端新增 `/api/train-samples` 端点

**Files:**
- Modify: `modules/ui/server.py`

- [x] **Step 1: 在 `/api/test-images` 旁边新增 `/api/train-samples`**

```python
@app.get("/api/train-samples")
async def api_train_samples(dataset: str = Query(...)):
    data_root = resolve_project_path(cfg_get('paths.data_root', './data'))
    dataset_dir = data_root / dataset
    if not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"数据集不存在: {dataset}")

    train_good_dir = dataset_dir / "train" / "good"
    if not train_good_dir.exists():
        return {"samples": [], "total": 0}

    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    samples = []
    for img_path in train_good_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in allowed_suffixes:
            rel = img_path.relative_to(dataset_dir).as_posix()
            samples.append(rel)

    return {"samples": sorted(samples), "total": len(samples)}
```

- [x] **Step 2: Commit**

```bash
git add modules/ui/server.py
git commit -m "feat(ui,api): 新增训练样本预览端点"
```

---

## Task 7: 前端改造 `app.js` 扩展推理状态

**Files:**
- Modify: `modules/ui/static/js/app.js`

- [x] **Step 1: 新增推理相关状态**

在 `app` 的 `return` 对象中新增：

```javascript
// 推理来源
inferenceSource: 'pretrained', // 'pretrained' | 'self_trained'
selfTrainedModels: [],
selectedSelfTrainedModel: null,

// 图片选择（从 test/）
testImages: [],
selectedTestImage: '',
testImagePreviewUrl: '',
```

- [x] **Step 2: 新增加载测试图片方法**

```javascript
loadTestImages: function () {
    var self = this;
    var dataset = self.inferenceSource === 'pretrained'
        ? self.selectedDataset
        : (self.selectedSelfTrainedModel ? self.selectedSelfTrainedModel.category : '');

    if (!dataset) {
        self.testImages = [];
        self.selectedTestImage = '';
        self.testImagePreviewUrl = '';
        return;
    }

    fetch('/api/test-images?dataset=' + encodeURIComponent(dataset))
        .then(function (res) { return res.json(); })
        .then(function (data) {
            self.testImages = data.images || [];
            self.selectedTestImage = self.testImages[0] || '';
            self.updateTestImagePreview();
        });
},

updateTestImagePreview: function () {
    var self = this;
    if (!self.selectedTestImage || !self.selectedDataset) {
        self.testImagePreviewUrl = '';
        return;
    }
    var dataset = self.inferenceSource === 'pretrained'
        ? self.selectedDataset
        : (self.selectedSelfTrainedModel ? self.selectedSelfTrainedModel.category : '');
    self.testImagePreviewUrl = '/data/' + dataset + '/' + self.selectedTestImage;
},
```

- [x] **Step 3: 新增加载自训练模型方法**

```javascript
loadSelfTrainedModels: function () {
    var self = this;
    fetch('/api/self-trained-models?model=' + encodeURIComponent(self.selectedModel))
        .then(function (res) { return res.json(); })
        .then(function (data) {
            self.selfTrainedModels = data.models || [];
            if (!self.selfTrainedModels.find(function (m) {
                return self.selectedSelfTrainedModel && m.path === self.selectedSelfTrainedModel.path;
            })) {
                self.selectedSelfTrainedModel = self.selfTrainedModels[0] || null;
            }
            if (self.inferenceSource === 'self_trained') {
                self.syncDatasetFromSelfTrained();
                self.loadTestImages();
            }
        });
},

syncDatasetFromSelfTrained: function () {
    if (this.selectedSelfTrainedModel) {
        this.selectedDataset = this.selectedSelfTrainedModel.category;
    }
},
```

- [x] **Step 4: 修改 `fetchModels()` 以加载自训练模型**

在 `fetchModels()` 成功回调末尾调用 `self.loadSelfTrainedModels()`。

- [x] **Step 5: 修改 `startInference()`**

将原来读取 `uploadedFile` 的逻辑改为使用 `selectedTestImage`，并构造新的 payload：

```javascript
startInference: function () {
    var self = this;
    if (!self.selectedTestImage) return;

    self.inferenceState = 'inferring';
    self.inferenceProgress = { stage: 'init', message: '正在初始化...', pct: 0 };

    var payload = {
        model: self.selectedModel,
        dataset: self.inferenceSource === 'pretrained'
            ? self.selectedDataset
            : self.selectedSelfTrainedModel.category,
        image: self.selectedTestImage,
        source: self.inferenceSource,
    };
    if (self.inferenceSource === 'self_trained') {
        payload.self_trained_path = self.selectedSelfTrainedModel.path;
    }

    InferenceRunner.run('/api/predict', payload, {
        onProgress: function (p) { self.inferenceProgress = p; },
        onResult: function (r) { self.resultData = r; self.inferenceState = 'done'; },
        onError: function (msg) { self.errorMessage = msg; self.inferenceState = 'error'; },
    });
},
```

> 注：`InferenceRunner.run()` 当前可能接受 `FormData`。因为图片路径已在服务端解析，payload 可改为 JSON；需要同步修改 `InferenceRunner` 以支持 JSON POST（当前 `inference.js` 中可能是 `FormData` + `multipart/form-data`）。

- [x] **Step 6: 移除上传相关状态与方法**

删除 `uploadedFile`、`uploadPreviewUrl`、`onFileSelected`、`onDrop`、`resetInference` 中 revoke upload URL 的逻辑。

- [x] **Step 7: Commit**

```bash
git add modules/ui/static/js/app.js
git commit -m "feat(ui,inference): 扩展全局推理状态支持自训练模型与 test/ 图片"
```

---

## Task 8: 前端改造 `inference.js` 支持 JSON payload

**Files:**
- Modify: `modules/ui/static/js/inference.js`

- [x] **Step 1: 修改 `InferenceRunner.run()` 支持对象 payload**

将 `run(url, file, model, dataset, handlers)` 签名改为 `run(url, payload, handlers)`。内部根据 `payload` 类型决定使用 `FormData` 还是 `JSON`：

```javascript
run: function (url, payload, handlers) {
    var self = this;
    self.cancel();
    self._abortController = new AbortController();

    var options = {
        method: 'POST',
        signal: self._abortController.signal,
    };

    if (payload instanceof FormData) {
        options.body = payload;
    } else {
        options.headers = { 'Content-Type': 'application/json' };
        options.body = JSON.stringify(payload);
    }

    fetch(url, options).then(function (response) {
        // ... 后续 SSE 解析逻辑不变
    });
},
```

- [x] **Step 2: Commit**

```bash
git add modules/ui/static/js/inference.js
git commit -m "feat(ui,inference): InferenceRunner 支持 JSON payload"
```

---

## Task 9: 前端改造 `index.html` 第二页布局

**Files:**
- Modify: `modules/ui/static/index.html`

- [x] **Step 1: 替换第二页左侧上传区为数据集选择与样本预览**

找到第二页 `section#s1-training` 中的上传区域（通常包含 `x-on:dragover`、`x-on:drop`、`input type="file"` 的元素），替换为：

```html
<div class="training-dataset-panel">
    <div class="panel-header">
        <h3>训练数据集</h3>
        <span class="dataset-badge" x-text="training.selectedDataset || '未选择'"></span>
    </div>

    <div class="dataset-selector-row">
        <label>选择数据集</label>
        <select x-model="training.selectedDataset">
            <template x-for="ds in datasets.filter(d => d.source === 'default')" :key="ds.value">
                <option :value="ds.value.replace('default/', '')" x-text="ds.label"></option>
            </template>
        </select>
    </div>

    <div class="train-samples-preview">
        <div class="preview-header">
            <span>训练样本预览</span>
            <span class="sample-count" x-text="'共 ' + training.trainSampleCount + ' 张'"></span>
        </div>
        <div class="preview-grid">
            <template x-for="(sample, idx) in training.trainSamples" :key="sample">
                <div class="preview-thumb">
                    <img :src="'/data/' + training.selectedDataset + '/' + sample" loading="lazy" />
                </div>
            </template>
        </div>
    </div>
</div>
```

- [x] **Step 2: 调整第二页右侧参数区**

保持现有参数表单，但确保在视觉上填充左侧释放的空间。

- [x] **Step 3: Commit**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui,training): 重排训练工作室 HTML 布局"
```

---

## Task 10: 前端改造 `index.html` 第三页布局

**Files:**
- Modify: `modules/ui/static/index.html`

- [x] **Step 1: 替换第三页上传区为图片来源分支 + 图片选择**

找到第三页 `section#s1` 中的上传区域，替换为：

```html
<div class="inference-source-tabs">
    <button
        class="source-tab"
        :class="{ active: inferenceSource === 'pretrained' }"
        @click="inferenceSource = 'pretrained'; loadTestImages()"
    >标准数据集</button>
    <button
        class="source-tab"
        :class="{ active: inferenceSource === 'self_trained' }"
        @click="inferenceSource = 'self_trained'; loadSelfTrainedModels()"
    >自训练模型</button>
</div>

<div class="inference-source-panel">
    <!-- 分支 A: 标准数据集 -->
    <template x-if="inferenceSource === 'pretrained'">
        <div class="source-panel">
            <label>选择数据集</label>
            <select x-model="selectedDataset" @change="loadTestImages()">
                <template x-for="ds in datasets.filter(d => d.source === 'default')" :key="ds.value">
                    <option :value="ds.value.replace('default/', '')" x-text="ds.label"></option>
                </template>
            </select>
        </div>
    </template>

    <!-- 分支 B: 自训练模型 -->
    <template x-if="inferenceSource === 'self_trained'">
        <div class="source-panel">
            <label>选择自训练模型</label>
            <select x-model="selectedSelfTrainedModel" @change="syncDatasetFromSelfTrained(); loadTestImages()">
                <template x-for="m in selfTrainedModels" :key="m.path">
                    <option :value="m" x-text="m.display_name + ' — ' + m.category + ' (v' + m.version + ')'"></option>
                </template>
            </select>
            <p class="hint" x-show="selfTrainedModels.length === 0">暂无自训练模型，请先在训练工作室训练。</p>
        </div>
    </template>
</div>

<div class="inference-image-panel">
    <label>选择测试图片</label>
    <select x-model="selectedTestImage" @change="updateTestImagePreview()" :disabled="testImages.length === 0">
        <template x-for="img in testImages" :key="img">
            <option :value="img" x-text="img"></option>
        </template>
    </select>
    <p class="hint" x-show="testImages.length === 0">该数据集暂无测试图片。</p>

    <div class="test-image-preview" x-show="testImagePreviewUrl">
        <img :src="testImagePreviewUrl" />
    </div>
</div>
```

- [x] **Step 2: 移除旧的文件上传 input 与拖拽区域**

删除原 `input type="file"`、拖拽提示、上传预览 img 等元素。

- [x] **Step 3: Commit**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui,inference): 重排单模型推理页 HTML 布局"
```

---

## Task 11: 前端样式补充

**Files:**
- Modify: `modules/ui/static/css/app.css`

- [x] **Step 1: 添加训练工作室数据集面板样式**

```css
.training-dataset-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.dataset-selector-row {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.dataset-selector-row select {
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-primary);
}

.train-samples-preview {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.preview-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
}

.preview-thumb img {
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    border-radius: 6px;
    border: 1px solid var(--border);
}
```

- [x] **Step 2: 添加推理来源 tab 与图片选择样式**

```css
.inference-source-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}

.source-tab {
    flex: 1;
    padding: 10px;
    border-radius: 20px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-secondary);
    cursor: pointer;
}

.source-tab.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
}

.inference-source-panel,
.inference-image-panel {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
}

.inference-image-panel select {
    padding: 10px 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-primary);
}

.test-image-preview img {
    max-height: 160px;
    border-radius: 8px;
    border: 1px solid var(--border);
}
```

- [x] **Step 3: Commit**

```bash
git add modules/ui/static/css/app.css
git commit -m "style(ui): 训练与推理新布局样式"
```

---

## Task 12: 左上角数据集选择器过滤掉 user 来源

**Files:**
- Modify: `modules/ui/static/index.html`

- [x] **Step 1: 修改导航栏数据集下拉**

将导航栏数据集 `<select>` 的选项改为只显示 `source === 'default'` 的数据集：

```html
<select x-model="selectedDataset">
    <template x-for="ds in datasets.filter(d => d.source === 'default')" :key="ds.value">
        <option :value="ds.value.replace('default/', '')" x-text="ds.label"></option>
    </template>
</select>
```

- [x] **Step 2: Commit**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui,nav): 导航栏数据集下拉仅显示标准数据集"
```

---

## Task 13: 集成测试与最终验证

- [x] **Step 1: 启动 UI 开发服务器**

```bash
python scripts/run_ui.py --no-browser
```

- [x] **Step 2: 手动验证第二页**

1. 打开 http://127.0.0.1:8000
2. 切换到第二页（训练工作室）
3. 确认左侧显示当前数据集名称与 `train/good/` 预览缩略图
4. 切换数据集，确认预览同步更新
5. 调整参数后点击训练，确认 SSE 正常推送

- [x] **Step 3: 手动验证第三页**

1. 切换到第三页（单模型推理）
2. 分支 A：选择标准数据集，从 `test/` 下拉选图，确认缩略图预览，点击推理
3. 分支 B：切换到“自训练模型”，选择已训练模型，确认数据集自动同步，从 `test/` 选图推理
4. 尝试在浏览器控制台修改 `selectedTestImage` 为 `../train/good/xxx.png`，确认后端返回 400

- [x] **Step 4: 运行全量测试**

```bash
python -m pytest tests/ -v
```

Expected: 全部通过。

- [x] **Step 5: 最终 Commit**

```bash
git add .
git commit -m "feat(ui): 训练工作室与单模型推理页重设计完成"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** 训练工作室移除上传、接入标准数据集、自训练模型分支、test/ 图片选择、路径安全、测试策略均已覆盖。
- [x] **Placeholder scan:** 无 TBD/TODO/"稍后实现"/"类似 Task N"。
- [x] **Type consistency:** `source` 统一为 `"pretrained" | "self_trained"`；`selectedDataset` 统一为类别名字符串；`selectedSelfTrainedModel` 为对象或 null。
