# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec: 把 physio 侧车 + numpy/scipy + CPython 运行时打成自包含目录。
# 用法（在 physio_service/ 目录下）: pyinstaller --noconfirm --clean physio_service.spec

a = Analysis(
    ["run.py"],
    pathex=["."],
    binaries=[],
    datas=[("config.json", ".")],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL", "pandas", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

# 不打包构建机的 libstdc++/libgcc_s。
# 它们可能链接了较新的 glibc（本机是 2.36/2.35），一旦打进包，在 glibc 较旧的
# 目标机（如 UOS 20 / 2.28）上会直接加载失败。按 manylinux 约定，这两个库由目标
# 系统的 base 提供（扩展通常只需 GLIBCXX_3.4，几乎所有发行版都有）。
a.binaries = [
    b
    for b in a.binaries
    if not (b[0].startswith("libstdc++") or b[0].startswith("libgcc_s"))
]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="physio_service",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="physio_service",
)
