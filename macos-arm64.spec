# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['model/main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('./.venv/lib/python3.11/site-packages/mediapipe/modules/', 'mediapipe/modules/'),
        ('./model/keypoint_classifier/keypoint_classifier_label.csv', 'keypoint_classifier/'),
        ('./model/keypoint_classifier/keypoint.csv', 'keypoint_classifier/'),
        ('./model/keypoint_classifier/pytorch_mlp_model.pth', 'keypoint_classifier/'),
        ('./model/keypoint_classifier/scaler.save', 'keypoint_classifier/')
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
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
