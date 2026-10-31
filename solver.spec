# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

import torch

if torch.version.cuda is not None or getattr(torch.version, 'hip', None) is not None:
    raise SystemExit('Release builds require CPU-only PyTorch. Run python create_release.py --force.')

excluded_packages = [
    # Other applications, development tools and local Transformers backends.
    'demo', 'create_release', 'prepare_dataset', 'add_to_dataset', 'edit_boxes',
    'train_yolo', 'yolo_test', 'simple_boxes', 'gradio',
    'transformers', 'diffusers', 'accelerate', 'bitsandbytes',
    'tensorflow', 'keras', 'jax', 'jaxlib', 'torchaudio',
    'tensorboard', 'torch.utils.tensorboard', 'wandb', 'mlflow', 'comet_ml',
    'clearml', 'ray', 'triton',
    # Optional export/inference backends; this app uses the PyTorch YOLO model.
    'onnx', 'onnxruntime', 'openvino', 'coremltools', 'tensorrt', 'paddle',
    'ultralytics.trackers', 'ultralytics.solutions', 'lap',
    # The app serves HTML; native GUI and notebook backends are unnecessary.
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6', 'tkinter', '_tkinter',
    'IPython', 'notebook', 'jupyterlab', 'pytest',
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('templates', 'templates'), ('static', 'static'), ('fonts', 'fonts')],
    hiddenimports=['main', 'uvicorn.logging', 'uvicorn.loops.asyncio', 'uvicorn.protocols.http.h11_impl', 'uvicorn.lifespan.on'],
    hookspath=[],
    hooksconfig={'matplotlib': {'backends': ['Agg']}},
    runtime_hooks=[],
    excludes=excluded_packages,
    noarchive=False,
    optimize=0,
)
# Fail rather than silently ship GPU DLLs pulled in by an optional dependency.
gpu_libraries = ('cublas', 'cudnn', 'cudart', 'cufft', 'curand', 'cusolver',
                 'cusparse', 'nvrtc', 'torch_cuda', 'c10_cuda', 'hiprtc', 'rocblas')
unexpected_gpu_files = [name for name, source, kind in a.binaries
                        if any(token in Path(source).name.lower() for token in gpu_libraries)]
if unexpected_gpu_files:
    raise SystemExit('Unexpected GPU libraries in CPU release: ' + ', '.join(unexpected_gpu_files))
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='solver',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
