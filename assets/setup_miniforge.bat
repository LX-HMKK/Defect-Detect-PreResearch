@echo off
chcp 65001 >nul
echo ================================================================================
echo  工业图像异常检测系统 - Miniforge 环境配置脚本
echo ================================================================================
echo.

set CONDA_PATH=D:\ProgramData\miniforge3\Scripts\conda.exe
set ENV_NAME=anomalib
set PYTHON_PATH=C:\Users\lx_hm\.conda\envs\%ENV_NAME%\python.exe

echo [INFO] Conda 路径: %CONDA_PATH%
echo [INFO] 环境名称: %ENV_NAME%
echo.

REM 检查环境是否存在
echo [STEP 1/6] 检查虚拟环境...
"%CONDA_PATH%" env list | findstr "^%ENV_NAME% " >nul
if errorlevel 1 (
    echo [INFO] 环境不存在，创建新环境...
    "%CONDA_PATH%" create -n %ENV_NAME% python=3.10 -y
    if errorlevel 1 (
        echo [ERROR] 环境创建失败
        exit /b 1
    )
) else (
    echo [OK] 环境已存在
)
echo.

echo [STEP 2/6] 安装 PyTorch (CUDA 11.8)...
"%CONDA_PATH%" run -n %ENV_NAME% conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 -y
if errorlevel 1 (
    echo [WARNING] CUDA 11.8 安装失败，尝试 CPU 版本...
    "%CONDA_PATH%" run -n %ENV_NAME% conda install -c pytorch pytorch torchvision -y
)
echo.

echo [STEP 3/6] 安装科学计算依赖...
"%CONDA_PATH%" run -n %ENV_NAME% conda install -c conda-forge numpy pandas scipy scikit-learn opencv tqdm pyyaml pillow -y
echo.

echo [STEP 4/6] 安装 anomalib 2.x 及其他 pip 依赖...
"%CONDA_PATH%" run -n %ENV_NAME% pip install "anomalib>=2.0.0" --upgrade
if errorlevel 1 (
    echo [ERROR] anomalib 安装失败
    exit /b 1
)
echo [INFO] 升级 timm (解决兼容性问题)...
"%CONDA_PATH%" run -n %ENV_NAME% pip install timm --upgrade
echo [INFO] 重新安装 opencv-python (解决 DLL 问题)...
"%CONDA_PATH%" run -n %ENV_NAME% pip install opencv-python==4.8.1.78 --force-reinstall --no-deps
echo.

echo [STEP 5/6] 验证安装...
"%PYTHON_PATH%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
"%PYTHON_PATH%" -c "import anomalib; print(f'Anomalib: {anomalib.__version__}')"
"%PYTHON_PATH%" -c "from anomalib.data import MVTec, Folder; from anomalib.engine import Engine; from anomalib.models import Patchcore, Draem, EfficientAd; print('Anomalib 2.x API OK')"
echo.

echo [STEP 6/6] 验证项目模块...
cd /d "%~dp0"
"%PYTHON_PATH%" -c "from modules.data_processing.dataset_formatter import MVTecFormatter; print('✓ MVTecFormatter 导入成功')"
"%PYTHON_PATH%" -c "from modules.algorithm.trainer import AnomalyDetectionTrainer; print('✓ AnomalyDetectionTrainer 导入成功')"
"%PYTHON_PATH%" -c "from modules.evaluation.metrics import MetricsEvaluator; print('✓ MetricsEvaluator 导入成功')"
"%PYTHON_PATH%" -c "from modules.ui.demo import AnomalyDetector; print('✓ AnomalyDetector 导入成功')"
echo.

echo ================================================================================
echo  Setup completed!
echo ================================================================================
echo.
echo Run commands:
echo.
echo   Train PatchCore:
echo   %PYTHON_PATH% run_training.py --model patchcore --category bottle --data_path ./data --device cuda
echo.
echo   Train all models:
echo   %PYTHON_PATH% run_training.py --model all --category bottle --data_path ./data --device cuda
echo.
echo   Start UI:
echo   %PYTHON_PATH% run_ui.py
echo.
echo Note: First run requires downloading pretrained weights from HuggingFace.
echo.
pause
