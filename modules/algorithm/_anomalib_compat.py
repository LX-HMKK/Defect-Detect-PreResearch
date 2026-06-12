"""
anomalib 2.3.0 与 PyTorch Lightning 1.9.5 兼容层

本模块在导入时自动应用 monkey-patch，修复以下几个不兼容问题：

1. TimerCallback.on_before_optimizer_step() 缺少 opt_idx 参数
2. on_validation_batch_start/end() dataloader_idx 参数问题
3. on_test_batch_start/end() dataloader_idx 参数问题
4. on_predict_batch_start/end() dataloader_idx 参数问题
5. on_predict_epoch_end() outputs 参数签名不匹配

升级 anomalib 或 pytorch-lightning 前需先确认这些补丁是否仍然需要。
"""

from anomalib.callbacks import TimerCallback
import pytorch_lightning.callbacks
import lightning.pytorch.callbacks

# ================================================================================
# 修复 1: TimerCallback.on_before_optimizer_step() 缺少 opt_idx 参数
# ================================================================================

_anomalib_callback_class = TimerCallback.__mro__[1]
_original_anomalib_callback_method = _anomalib_callback_class.on_before_optimizer_step

_lightning_callback_class = pytorch_lightning.callbacks.Callback
_original_lightning_callback_method = _lightning_callback_class.on_before_optimizer_step


def _patched_on_before_optimizer_step(self, trainer, pl_module, optimizer, **kwargs):
    return _original_anomalib_callback_method(self, trainer, pl_module, optimizer)


def _patched_lightning_on_before_optimizer_step(self, trainer, pl_module, optimizer, **kwargs):
    opt_idx = kwargs.get('opt_idx', 0)
    return _original_lightning_callback_method(self, trainer, pl_module, optimizer, opt_idx)


_anomalib_callback_class.on_before_optimizer_step = _patched_on_before_optimizer_step
_lightning_callback_class.on_before_optimizer_step = _patched_lightning_on_before_optimizer_step

# ================================================================================
# 修复 2: on_validation_batch_start() dataloader_idx 参数
# ================================================================================

_original_lightning_val_batch_start = _lightning_callback_class.on_validation_batch_start


def _patched_on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
    dataloader_idx = kwargs.get('dataloader_idx', 0)
    return _original_lightning_val_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx)


_lightning_callback_class.on_validation_batch_start = _patched_on_validation_batch_start

# ================================================================================
# 修复 3-7: 所有 batch 回调的 dataloader_idx 参数
# ================================================================================

_original_val_batch_end = _lightning_callback_class.on_validation_batch_end


def _patched_on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
    dataloader_idx = kwargs.get('dataloader_idx', 0)
    return _original_val_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


_lightning_callback_class.on_validation_batch_end = _patched_on_validation_batch_end

_original_test_batch_start = _lightning_callback_class.on_test_batch_start


def _patched_on_test_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
    dataloader_idx = kwargs.get('dataloader_idx', 0)
    return _original_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx)


_lightning_callback_class.on_test_batch_start = _patched_on_test_batch_start

_original_test_batch_end = _lightning_callback_class.on_test_batch_end


def _patched_on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
    dataloader_idx = kwargs.get('dataloader_idx', 0)
    return _original_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


_lightning_callback_class.on_test_batch_end = _patched_on_test_batch_end

_original_predict_batch_start = _lightning_callback_class.on_predict_batch_start


def _patched_on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, **kwargs):
    dataloader_idx = kwargs.get('dataloader_idx', 0)
    return _original_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx)


_lightning_callback_class.on_predict_batch_start = _patched_on_predict_batch_start

_original_predict_batch_end = _lightning_callback_class.on_predict_batch_end


def _patched_on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, **kwargs):
    dataloader_idx = kwargs.get('dataloader_idx', 0)
    return _original_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)


_lightning_callback_class.on_predict_batch_end = _patched_on_predict_batch_end

# ================================================================================
# 修复 8: on_predict_epoch_end() outputs 参数签名不匹配
# ================================================================================

_lt_callback_class = lightning.pytorch.callbacks.Callback
_original_lt_predict_epoch_end = _lt_callback_class.on_predict_epoch_end


def _patched_lt_on_predict_epoch_end(self, trainer, pl_module, outputs=None):
    return _original_lt_predict_epoch_end(self, trainer, pl_module)


_lt_callback_class.on_predict_epoch_end = _patched_lt_on_predict_epoch_end

_original_pl_predict_epoch_end = _lightning_callback_class.on_predict_epoch_end


def _patched_pl_on_predict_epoch_end(self, trainer, pl_module, *args, **kwargs):
    if args:
        outputs = args[0]
    elif 'outputs' in kwargs:
        outputs = kwargs['outputs']
    else:
        outputs = None
    return _original_pl_predict_epoch_end(self, trainer, pl_module, outputs)


_lightning_callback_class.on_predict_epoch_end = _patched_pl_on_predict_epoch_end
