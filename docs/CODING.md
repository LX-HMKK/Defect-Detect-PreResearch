# 编码规范、Git 提交、UI 架构与陷阱参考

> 本文件从 `CLAUDE.md` 抽离：编码规范细则与示例、Git 提交规范、Phase 2 UI 架构详解、全部 CSS/Alpine/IO 陷阱、UI 调试、Trainer 兼容补丁与 checkpoint 安全格式。
> 写代码或改 UI 时按需查阅。高频陷阱（cv2 导入顺序、训练串行、数据集路径）仍在 `CLAUDE.md`。

## 编码规范

### 文档语言

**所有文档、注释、提交信息必须使用中文。** 包括但不限于：README、CHANGELOG、CLAUDE.md、docstring、行内注释、Git 提交主题行。

### 导入顺序（必须遵守）

```python
# 1. 标准库（按字母顺序）
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 2. 第三方库 — cv2 必须在 anomalib 之前导入
import cv2  # ← 始终是第三方导入中的第一个
import numpy as np
import pandas as pd
from tqdm import tqdm

# 3. 框架 (anomalib)
from anomalib.data import MVTec
from anomalib.engine import Engine

# 4. 本地导入（绝对导入）
from modules.config import get_threshold
from modules.evaluation.metrics import MetricsEvaluator
```

`import cv2` **必须**在任何 anomalib 导入之前。否则 Windows 上 DLL 加载失败。

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `AnomalyDetectionTrainer`, `MetricsEvaluator` |
| 函数/变量 | snake_case | `compute_image_auroc`, `good_scores` |
| 常量 | UPPER_SNAKE_CASE | `SUPPORTED_MODELS`, `NMS_BBOX_THRESHOLD` |
| 私有方法 | `_` 前缀 | `_load_config`, `_compute_optimal_threshold` |
| 模块级私有 | `_` 前缀 | `_lightning_callback_class` |

### 类型注解

```python
# 使用类型注解提高可读性
def compute_threshold(scores: List[float], labels: List[bool]) -> float:
    threshold: float = 0.5
    return threshold

# Union 类型使用 `|`
def process_data(data: str | Path) -> Dict[str, Any]:
    ...
```

### 错误处理

```python
# 必须捕获具体异常
try:
    from anomalib.data import MVTec
except ImportError as e:
    print(f"错误：请运行 pip install anomalib>=2.0.0")
    raise

# 禁止空 except — 始终指定异常类型
try:
    result = risky_operation()
except ValueError as e:
    print(f"[WARN] 值错误: {e}")
    raise  # 或返回默认值
```

### 文档字符串

```python
def train_and_evaluate(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
    """
    完整流程：训练 + 评估。

    Args:
        max_epochs: 最大训练轮次。

    Returns:
        Dict: 评估结果（4 个核心指标）。

    Raises:
        ValueError: 必需配置键缺失时抛出。
    """
```

## Git 提交规范

Angular 协议：`<类型>(<范围>): <主题>`。

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(ui): 添加算法切换功能` |
| fix | 修复 bug | `fix(trainer): 修复阈值搜索范围` |
| docs | 文档更新 | `docs: 更新 README` |
| style | 代码格式 | `style: 格式化代码` |
| refactor | 重构 | `refactor: 重构模型配置结构` |
| perf | 性能优化 | `perf(patchcore): 启用预训练权重` |

规则：
- 主题行不超过 72 字符
- 使用命令式语气（add, fix, update）
- **禁止添加 `Co-authored-by`** 到提交信息
- **多行提交信息必须使用 `-F` 从文件读取**。禁止使用 `git commit -m @'...'@`（PowerShell here-string 语法）或直接内联多行消息——本项目中 Bash 和 PowerShell 两套 shell 并存，here-string/here-doc 语法在交叉环境下极容易误用，导致 `@` 等无关字符混入提交信息。正确做法：先将消息写入临时文件（如 `.git-msg`），然后执行 `git commit -F .git-msg`，完成后删除。

提交前自检（筛查是否误带协作者签名）：

```bash
git log --all --format="%H %B" | grep -i "Co-authored-by"
```

## Trainer 兼容性补丁

`modules/algorithm/_anomalib_compat.py` 包含针对 anomalib 2.3.0 与 PyTorch Lightning 1.9.5 的兼容性补丁（回调签名不匹配）。该文件在 `trainer.py` 中通过 `from . import _anomalib_compat` 导入时自动触发。在未验证 Lightning/anomalib 版本组合之前，不要移除这些补丁。

## 自训练模型 checkpoint 安全格式

训练完成后，`run_training_job()` 会将 Lightning checkpoint 重写为仅含 `{'state_dict': ...}` 的安全格式，并在推理时使用 `weights_only=True` 加载。因此自训练模型走 `source='user'` 路径时，`Engine.predict()` 传入 `ckpt_path=None`，状态字典由 `AnomalyDetector.load_self_trained_model()` 手动注入模型。不要直接给 `Engine.predict()` 传入原始 Lightning checkpoint 路径。

## Phase 2 UI 架构

**默认 UI**: FastAPI + Alpine.js SPA (`modules/ui/server.py` + `modules/ui/static/`)。

- 5 层动效体系：环境光呼吸 → 鼠标光晕跟随 → 滚动驱动动画 → 微交互（胶囊开关/数字跳动/弹簧按钮）→ 视图过渡
- CSS scroll-snap：**仅首页(s0) 吸附**（`.snap-container` `scroll-snap-type: y proximity` + `.snap-page--home { snap-align: start; snap-stop: always }`），s1/s2/s3 为 `snap-align: none` 自由滚动、可页内自由停留；四页顺序：算法介绍(s0) → 训练工作室(s1) → 单模型推理(s2) → 四模型对比(s3)
- 每页 100dvh 全视口，滚动吸附到整页，无半页停留
- 右侧进度环导航点（SVG 圆环 + 页码"1/4"），实时反映滚动位置
- **进出动画**：统一由 JS WAAPI 驱动（`snapPageEnter` / `snapPageExit`），方向向下推送（与滚动方向一致），CSS 退出动画已删除以避免双动画竞争
- **S1（训练工作室）布局**：左侧样本上传与参数配置，右侧实时监控曲线与指标面板；上传后生成可排除样本的画廊，训练完成触发全局模型列表刷新
- **S2（单模型推理）布局**：三列流水线（上传→选择→推理）→ 完成后收缩为 `.pipeline-summary` 步骤摘要 → `.result-dashboard` 左图右信息分栏（`.result-dashboard-grid`：左列对比滑块主视觉 `.result-dashboard-compare` + 右列判决/元信息 `.result-dashboard-aside`）
- **S3（四模型对比）布局**：共享原图画廊居中（`.compare-shared-image` max-width 560px / img max-height 180px）→ 单行摘要栏 `.compare-summary-row` → 四列对比网格（顶部横线色标 `.compare-slot-accent`，仅热力图 + 得分/置信度）
- 亮/暗双模式：胶囊开关，localStorage 持久化，`prefers-color-scheme` 系统跟随
- 设计规范以代码现状为准；历史 UI 迭代已归档至 `CHANGELOG.md`。

**回退**: `python scripts/run_ui.py --gradio` 启动原有 Gradio UI（`modules/ui/demo.py`），功能完整保留。

## 陷阱

### CSS 陷阱：`.pipeline` Grid 子元素数量

`.pipeline` 使用 `grid-template-columns: 1fr 1fr 1fr` 精确三列布局。**禁止在 `.pipeline` 内添加除 `.pipeline-step` 外的任何子元素**——即使是非 `pipeline-step` 的 div 也会被 grid auto-placement 占据列位，将后续步骤推至第二行。步骤间连接线必须使用 `::after` 伪元素，不能添加 DOM 节点。

### Alpine 陷阱：`x-data` 子作用域访问父属性

`section#s3` 带有 `x-data="compare"`（子作用域），其内的 Alpine 表达式无法直接访问 `app` 作用域的属性（如 `resultData`）。`x-show` / `:src` 等绑定必须使用当前作用域内的属性（如 `compareDone`、`compareSlots`）。

### CSS 陷阱：`.compare-heatmap` 选择器泄漏

`app.css` 中 `.compare-heatmap { position: absolute; }` 为单模型对比滑块设计（热力图叠加在原图之上），该选择器会泄漏到四模型对比槽位。`.compare-slot .compare-heatmap` 覆盖规则若未显式设置 `position: relative`，热力图将脱离文档流，导致父容器 `.compare-heatmap-wrap` 高度塌陷至 0px，配合 `overflow: hidden` 将热力图完全裁剪。**任何对 `.compare-heatmap` 的修改必须验证两种用法**：(1) `.compare-container .compare-heatmap` 滑块叠层，(2) `.compare-slot .compare-heatmap` 对比槽位。

### CSS 陷阱：Snap 进出动画双驱动竞争

进出动画**仅由 JS WAAPI 驱动**（`Anim.snapPageEnter` / `Anim.snapPageExit`）。旧的 CSS `@keyframes pageContentExit` 和 `.snap-page--exiting .snap-page-inner > *` 规则已删除。若重新添加 CSS animation 到 `.snap-page--exiting` 选择器，会与 JS WAAPI 形成双动画竞争，导致元素同时执行两套动画（闪烁/跳变）。`@supports (animation-timeline: view())` 块已注释禁用，待 Chrome 原生 scroll-driven animations 成熟后再评估迁移。

### CSS 陷阱：内联「关键布局 CSS」覆盖外链 app.css

`index.html` `<head>` 内有一个内联 `<style>` 块（「关键布局 CSS」，FOUC 用），**复制了** `.snap-container` / `.snap-page` / `.snap-page--home` 等规则。它在文档序中位于外链 app.css/flowchart.css/apple-redesign.css **之后**，对同特异性单类选择器（`.snap-page--home` 与 `.snap-page` 均为 0,1,0）**后者胜** → inline 的 `.snap-page { padding: max(120px,12vh) }` 会覆盖 app.css 的 `.snap-page--home { padding: 0 }`。改 `.snap-page` / `.snap-page--home` / `.hero-title` 基础规则时，若只改 app.css 会"改了不生效"。**务必同步改 inline `<style>` 块与 apple-redesign.css**（后者覆盖 `.hero-title` 的 font-weight/font-size）；判定来源用 CSSOM 遍历 `document.styleSheets`（含嵌套 `@media` 的 `cssRules`）查计算值真正生效的 sheet。

### IntersectionObserver 陷阱：`root` 参数缺失

`.snap-container` 为 `overflow-y: auto` 的滚动容器，section 在其内部滚动。若 `IntersectionObserver` 不指定 `root` 参数，默认以 viewport 为根进行观察——而 `.snap-container` 填满 100dvh 视口，其内部所有 section 对 viewport 均 100% 可见，导致 Observer **永远检测不到 section 切换**。务必在 options 中传入 `root: container`（container 指向 `.snap-container` 元素）。

### CSS 陷阱：`border-image` + `border-radius` 互斥

`border-image` 会完全替代 `border-radius` 的渲染——设置 `border-image` 后圆角静默失效，显示为直角。需要渐变边框+圆角共存时，必须用 `::before` 伪元素 + `mask-composite: exclude` 模拟，而非 `border-image`。

示例：
```css
/* ❌ 错误：border-image 会覆盖 border-radius */
.summary {
    border-radius: 12px;
    border-image: linear-gradient(135deg, gold, orange) 1;
}

/* ✅ 正确：::before + mask-composite */
.summary {
    border-radius: 12px;
    position: relative;
}
.summary::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: linear-gradient(135deg, gold, orange);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
}
```

## UI 调试 (Phase 2)

```bash
# 启动 FastAPI 开发服务器
python scripts/run_ui.py
# → http://127.0.0.1:8000

# 健康检查
curl http://127.0.0.1:8000/api/health

# 模型列表
curl http://127.0.0.1:8000/api/models

# 浏览器控制台验证 snap 状态
document.querySelector('.snap-container').style.scrollSnapType  // "y mandatory"
document.querySelector('.snap-dot-label').textContent           // "1 / 4"
```

**Gradio 6 CSS 作用域问题（仅影响 legacy Gradio UI）**：`gr.Blocks(css=...)` 传入的 CSS 会被 Gradio 6 做选择器作用域处理——在所有选择器前加 `.gradio-container.xxx .contain`。这会导致 `@media` 查询内的 `:root` 选择器失效（变成 `.contain :root`，无法匹配文档根）。**解决方案**：需要通过 CSS `@media` 动态切换的变量（如亮色模式色板），必须通过 `gr.HTML("<style>…</style>")` 注入，绕过 Gradio 的 CSS 处理器。顶层 `:root` 块（暗色默认值）不受影响。

**亮/暗双模式：** 系统通过 `prefers-color-scheme` 自动检测，并支持手动切换。手动选择存储在 `localStorage.theme`，优先级高于系统设定。CSS 通过 `html[data-theme="light"]` 选择器覆盖变量。切换逻辑在 `modules/ui/static/theme.js`。
