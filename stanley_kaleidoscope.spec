# -*- mode: python ; coding: utf-8 -*-

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

# Collect PyQt6 platform and imageformat plugins
datas = collect_data_files("PyQt6")

a = Analysis(
    ["src/main.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "PIL.JpegImagePlugin",
        "PIL.PngImagePlugin",
        "PIL.TiffImagePlugin",
        "PIL.WebPImagePlugin",
        "PIL.BmpImagePlugin",
        "PIL.GifImagePlugin",
        "PIL.IcoImagePlugin",
        "PIL.Image",
        "PIL.ImageOps",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Stanley Kaleidoscope",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # MUST be False on macOS — UPX corrupts Qt binaries
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,  # prevents startup hang on macOS with PyQt6
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="assets/icon.icns",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Stanley Kaleidoscope",
)

app = BUNDLE(
    coll,
    name="Stanley Kaleidoscope.app",
    icon="assets/icon.icns",
    bundle_identifier="com.stanley.kaleidoscope",
    version="1.0.0",
    info_plist={
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        "NSAppleScriptEnabled": False,
        "CFBundleShortVersionString": "1.0.0",
        "CFBundleVersion": "1.0.0",
        "LSMinimumSystemVersion": "10.15",
        "NSHumanReadableCopyright": "© 2024 Stanley",
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "Image",
                "CFBundleTypeRole": "Viewer",
                "LSItemContentTypes": [
                    "public.png",
                    "public.jpeg",
                    "public.tiff",
                    "com.compuserve.gif",
                    "public.bmp",
                ],
            }
        ],
    },
)
