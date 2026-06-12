# 贡献指南

## 开发环境

- **操作系统:** Windows 11（开发主要在 Windows 上进行）
- **Python:** 3.10
- **GPU:** NVIDIA RTX 4060 Laptop GPU (8GB VRAM)
- **框架:** anomalib 2.3.0 + PyTorch 2.x + PyTorch Lightning 1.9.5

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest tests/ -v

# 验证数据完整性
python tools/validate_data.py
```

## 开发约定

详见 [CLAUDE.md](CLAUDE.md)，以下是关键规则摘要：

### 提交信息

- 必须使用**中文**
- 格式遵循 Angular 约定：`<类型>(<范围>): <主题>`
- **禁止**添加 `Co-Authored-By` 到提交信息

示例：
```
feat(tools): 添加混淆矩阵生成脚本
fix(trainer): 修复 FRE 模型像素级指标缺失问题
docs: 更新数据集注册表
```

### 代码风格

- `import cv2` 必须在任何 anomalib 导入之前
- Windows 上 `num_workers` 必须设为 `0`
- pycache 通过 `modules/_runtime.py` 重定向到 `temp/pycache/`
- UTF-8 stdout 包装用于 Windows 终端兼容

### 文档

- 所有文档、注释、提交信息必须使用**中文**

## 添加新数据集

1. 数据按 MVTec AD 布局组织：`train/good/`、`test/<defect>/`、`ground_truth/<defect>/`
2. 将数据集放在 `data/<dataset_name>/` 下
3. 更新 `data/README.md` 数据集清单
4. 运行 `python tools/validate_data.py` 检查数据完整性
5. 如需要，在 `configs/config.yaml` 的 `threshold.dataset_defaults` 中添加默认阈值

## 添加新模型

1. 在 `modules/algorithm/trainer.py` 的 `SUPPORTED_MODELS` 和 `MODEL_INFO` 中注册
2. 在 `get_model_from_config()` 中添加模型创建分支
3. 创建 `configs/<model_name>.yaml` 配置文件
4. 更新 `configs/config.yaml` 的 `models` 部分
5. 更新 README 的算法对比表

## 测试

- 测试文件位于 `tests/`
- 不需要 GPU 或 anomalib 的测试可以在任何环境运行
- 需要 GPU 的集成测试目前尚未添加（待 CI 环境支持）
