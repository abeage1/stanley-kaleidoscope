#!/usr/bin/env bash
# build_app.sh — One-command build for Stanley Kaleidoscope.app
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  Stanley Kaleidoscope — macOS App Builder"
echo "=================================================="

# ── 1. Ensure Python 3.11 ────────────────────────────────────────────────
echo ""
echo "→ Installing Python 3.11 via uv…"
uv python install 3.11

# ── 2. Create / refresh venv ─────────────────────────────────────────────
echo ""
echo "→ Creating virtual environment…"
uv venv --python 3.11 .venv

# shellcheck disable=SC1091
source .venv/bin/activate

# ── 3. Install dependencies ───────────────────────────────────────────────
echo ""
echo "→ Installing dependencies…"
uv pip install "PyQt6>=6.6.0" "numpy>=1.26.0" "Pillow>=10.0.0" "pyinstaller>=6.3.0"

# ── 4. Generate icon ──────────────────────────────────────────────────────
echo ""
echo "→ Generating app icon…"
mkdir -p assets

if [ ! -f "assets/icon.icns" ]; then
    ICONSET_DIR="assets/icon.iconset"
    mkdir -p "$ICONSET_DIR"

    python3 - <<'PYEOF'
from PIL import Image, ImageDraw, ImageFilter
import os, math

def make_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background circle
    margin = size // 16
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(30, 30, 35, 255),
    )

    # Draw kaleidoscope-like segments
    cx, cy = size / 2, size / 2
    r = size * 0.38
    n = 8
    for i in range(n):
        angle1 = math.radians(i * 360 / n)
        angle2 = math.radians((i + 0.5) * 360 / n)
        hue = i / n
        # Simple HSV to RGB
        h = hue * 6
        sector = int(h)
        f = h - sector
        colors = [
            (int(255 * (1 - f)), 0, int(255 * f)),
            (0, int(255 * f), 255),
            (0, 255, int(255 * (1 - f))),
            (int(255 * f), 255, 0),
            (255, int(255 * (1 - f)), 0),
            (255, 0, int(255 * f)),
        ]
        base_color = colors[sector % 6]
        # Lighten
        color = tuple(min(255, c + 40) for c in base_color) + (220,)
        x1 = cx + r * math.cos(angle1)
        y1 = cy + r * math.sin(angle1)
        x2 = cx + r * math.cos(angle2)
        y2 = cy + r * math.sin(angle2)
        draw.polygon(
            [(cx, cy), (x1, y1), (x2, y2)],
            fill=color,
        )

    # Center dot
    dot_r = size * 0.06
    draw.ellipse(
        [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
        fill=(255, 255, 255, 200),
    )
    return img

sizes = [16, 32, 64, 128, 256, 512, 1024]
iconset = "assets/icon.iconset"
for s in sizes:
    icon = make_icon(s)
    icon.save(f"{iconset}/icon_{s}x{s}.png")
    if s <= 512:
        icon2 = make_icon(s * 2)
        icon2.save(f"{iconset}/icon_{s}x{s}@2x.png")

print("Icon PNG files generated.")
PYEOF

    iconutil -c icns "$ICONSET_DIR" -o "assets/icon.icns"
    echo "   assets/icon.icns created."
else
    echo "   assets/icon.icns already exists, skipping."
fi

# ── 5. Clean previous build artifacts ────────────────────────────────────
echo ""
echo "→ Cleaning previous build…"
rm -rf build/ dist/

# ── 6. Run PyInstaller ────────────────────────────────────────────────────
echo ""
echo "→ Running PyInstaller…"
pyinstaller stanley_kaleidoscope.spec --noconfirm --clean

# ── 7. Done ───────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Build complete!"
echo "=================================================="
echo ""
echo "App location:  dist/Stanley Kaleidoscope.app"
echo ""
echo "To run:"
echo "  open \"dist/Stanley Kaleidoscope.app\""
echo ""
echo "To distribute:"
echo "  1. Drag \"dist/Stanley Kaleidoscope.app\" to /Applications"
echo "  2. First launch: right-click → Open (bypasses Gatekeeper)"
echo ""
