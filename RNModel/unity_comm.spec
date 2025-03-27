# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['unity_comm.py'],
    pathex=[],
    binaries=[],
    datas=[ ('age_model_resnet.h5', '.'),
    ('gender_model_resnet.h5', '.'),
    ('affectnet_model.h5', '.'),
	('shape_predictor_68_face_landmarks.dat','./face_recognition_models/models'),('shape_predictor_5_face_landmarks.dat','./face_recognition_models/models'),('mmod_human_face_detector.dat','./face_recognition_models/models'),('dlib_face_recognition_resnet_model_v1.dat','./face_recognition_models/models')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='unity_comm',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='unity_comm',
)
