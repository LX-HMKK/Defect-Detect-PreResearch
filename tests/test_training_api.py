import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_image_bytes():
    """生成一张 64x64 的灰度 PNG 图片字节。"""
    img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture
def client():
    """共享的 TestClient，导入 server 模块时会触发 heavy import。"""
    from modules.ui.server import app
    with TestClient(app) as c:
        yield c


def test_upload_samples_accepts_images(client):
    """上传单张图片应返回 200 并生成 MVTec AD 临时结构。"""
    data = _make_image_bytes()
    response = client.post(
        '/api/upload-samples',
        files={'files': ('sample.png', io.BytesIO(data), 'image/png')},
    )
    assert response.status_code == 200
    body = response.json()
    assert body['total'] == 1
    assert body['max_allowed'] == 150
    assert 'session_id' in body
    assert 'dataset_path' in body
    assert 'samples' in body
    assert Path(body['dataset_path']).exists()


def test_upload_samples_rejects_non_image(client):
    """上传非图片文件应返回 400。"""
    response = client.post(
        '/api/upload-samples',
        files={'files': ('readme.txt', io.BytesIO(b'hello'), 'text/plain')},
    )
    assert response.status_code == 400


def test_upload_samples_rejects_empty_request(client):
    """空文件列表应返回 400。"""
    response = client.post('/api/upload-samples', files={})
    assert response.status_code == 422


def test_train_status_initially_idle(client):
    """初始状态训练锁应为空闲。"""
    response = client.get('/api/train-status')
    assert response.status_code == 200
    assert response.json()['running'] is False


def test_train_rejects_invalid_model(client):
    """不支持的模型应在启动 SSE 前返回 400。"""
    response = client.post(
        '/api/train',
        json={
            'model': 'notamodel',
            'dataset_path': '.cache/uploads/training_test',
            'category': 'training_test',
            'epochs': 1,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'seed': 42,
        },
    )
    assert response.status_code == 400


def test_train_rejects_invalid_category(client):
    """category 含非法字符应返回 400。"""
    response = client.post(
        '/api/train',
        json={
            'model': 'patchcore',
            'dataset_path': '.cache/uploads/training_test',
            'category': 'training/test',
            'epochs': 1,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'seed': 42,
        },
    )
    assert response.status_code == 400


def test_train_rejects_out_of_bounds_epochs(client):
    """epochs 越界应返回 400。"""
    response = client.post(
        '/api/train',
        json={
            'model': 'patchcore',
            'dataset_path': '.cache/uploads/training_test',
            'category': 'training_test',
            'epochs': 0,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'seed': 42,
        },
    )
    assert response.status_code == 400


def test_train_rejects_out_of_bounds_batch_size(client):
    """batch_size 越界应返回 400。"""
    response = client.post(
        '/api/train',
        json={
            'model': 'patchcore',
            'dataset_path': '.cache/uploads/training_test',
            'category': 'training_test',
            'epochs': 1,
            'batch_size': 0,
            'learning_rate': 0.0001,
            'seed': 42,
        },
    )
    assert response.status_code == 400


def test_train_rejects_invalid_learning_rate(client):
    """learning_rate 不在 (0, 1) 区间应返回 400。"""
    response = client.post(
        '/api/train',
        json={
            'model': 'patchcore',
            'dataset_path': '.cache/uploads/training_test',
            'category': 'training_test',
            'epochs': 1,
            'batch_size': 32,
            'learning_rate': 1.0,
            'seed': 42,
        },
    )
    assert response.status_code == 400


def test_train_rejects_path_outside_upload_root(client):
    """dataset_path 不在上传目录下应返回 400。"""
    response = client.post(
        '/api/train',
        json={
            'model': 'patchcore',
            'dataset_path': './data/bottle',
            'category': 'training_test',
            'epochs': 1,
            'batch_size': 32,
            'learning_rate': 0.0001,
            'seed': 42,
        },
    )
    assert response.status_code == 400
