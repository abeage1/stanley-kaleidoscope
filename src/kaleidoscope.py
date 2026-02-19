"""
Pure NumPy kaleidoscope transformation engine.
No loops — fully vectorized polar/Cartesian coordinate transforms.

Modes
-----
radial          Classic N-segment wedge (original)
rectangle       Mirror-fold in X and Y independently
triangle_45     45-45-90 right triangle (tiles the square)
triangle_60     60-60-60 equilateral triangle (tiles with hexagonal lattice)
triangle_30_60  30-60-90 right triangle (tiles the rectangle 1:√3)
"""

import numpy as np


# ── Shared helpers ───────────────────────────────────────────────────────────

def _build_grid(out_w: int, out_h: int):
    """Return meshgrid of output pixel coordinates (xg, yg)."""
    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    return np.meshgrid(xs, ys)


def _rotate(dx: np.ndarray, dy: np.ndarray, angle_deg: float):
    """Rotate (dx, dy) by angle_deg degrees."""
    if angle_deg == 0.0:
        return dx, dy
    rad = np.deg2rad(angle_deg)
    c, s = np.cos(rad), np.sin(rad)
    return dx * c - dy * s, dx * s + dy * c


def _mirror_fold(v: np.ndarray, period: float) -> np.ndarray:
    """
    Mirror-tile fold: maps any v to [0, period/2].
    Creates a triangle-wave with period `period`.
    """
    v_mod = v % period
    return np.where(v_mod <= period * 0.5, v_mod, period - v_mod)


def _bilinear_sample(
    source: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    src_w: int,
    src_h: int,
) -> np.ndarray:
    """Vectorized bilinear interpolation."""
    sx_c = np.clip(sx, 0.0, src_w - 1.001)
    sy_c = np.clip(sy, 0.0, src_h - 1.001)

    x0 = np.floor(sx_c).astype(np.int32)
    y0 = np.floor(sy_c).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)

    fx = (sx_c - x0).astype(np.float32)[..., np.newaxis]
    fy = (sy_c - y0).astype(np.float32)[..., np.newaxis]

    c00 = source[y0, x0].astype(np.float32)
    c10 = source[y0, x1].astype(np.float32)
    c01 = source[y1, x0].astype(np.float32)
    c11 = source[y1, x1].astype(np.float32)

    result = (
        c00 * (1 - fx) * (1 - fy)
        + c10 * fx * (1 - fy)
        + c01 * (1 - fx) * fy
        + c11 * fx * fy
    )
    return result.astype(np.uint8)


def _common_setup(source, rotation_deg, zoom, center_x_pct, center_y_pct, output_size):
    """
    Build centered, rotated, zoomed offset grids (dx, dy) and source center
    (cx_src, cy_src).  Returns (dx, dy, cx_src, cy_src, src_w, src_h, out_w, out_h).
    """
    src_h, src_w = source.shape[:2]
    if output_size is not None:
        out_w, out_h = output_size
    else:
        out_w, out_h = src_w, src_h

    cx_out = out_w / 2.0
    cy_out = out_h / 2.0
    cx_src = center_x_pct / 100.0 * src_w
    cy_src = center_y_pct / 100.0 * src_h

    xg, yg = _build_grid(out_w, out_h)
    dx = (xg - cx_out) / zoom
    dy = (yg - cy_out) / zoom
    dx, dy = _rotate(dx, dy, rotation_deg)

    return dx, dy, cx_src, cy_src, src_w, src_h, out_w, out_h


# ── Mode 1: Radial (original) ─────────────────────────────────────────────

def apply_kaleidoscope(
    source: np.ndarray,
    num_segments: int,
    rotation_deg: float,
    zoom: float,
    center_x_pct: float,
    center_y_pct: float,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Classic radial N-segment mirror-wedge kaleidoscope.
    """
    src_h, src_w = source.shape[:2]
    if output_size is not None:
        out_w, out_h = output_size
    else:
        out_w, out_h = src_w, src_h

    cx_out = out_w / 2.0
    cy_out = out_h / 2.0
    cx_src = center_x_pct / 100.0 * src_w
    cy_src = center_y_pct / 100.0 * src_h

    xg, yg = _build_grid(out_w, out_h)
    dx = xg - cx_out
    dy = yg - cy_out

    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    rotation_rad = np.deg2rad(rotation_deg)
    theta = (theta + rotation_rad) % (2.0 * np.pi)

    segment_angle = 2.0 * np.pi / num_segments
    theta_norm = theta % segment_angle
    half_segment = segment_angle / 2.0
    mirror_mask = theta_norm > half_segment
    theta_norm[mirror_mask] = segment_angle - theta_norm[mirror_mask]

    r_zoomed = r / zoom
    sx = r_zoomed * np.cos(theta_norm) + cx_src
    sy = r_zoomed * np.sin(theta_norm) + cy_src

    return _bilinear_sample(source, sx, sy, src_w, src_h)


# ── Mode 2: Rectangle tile ────────────────────────────────────────────────

def apply_rectangle(
    source: np.ndarray,
    tile_size_pct: float,
    tile_aspect: float,
    rotation_deg: float,
    zoom: float,
    center_x_pct: float,
    center_y_pct: float,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Mirror-fold in X and Y independently, creating a rectangular tile pattern.

    tile_size_pct : tile width as % of min(src_w, src_h)
    tile_aspect   : height / width ratio (1.0 = square)
    """
    dx, dy, cx_src, cy_src, src_w, src_h, *_ = _common_setup(
        source, rotation_deg, zoom, center_x_pct, center_y_pct, output_size
    )

    min_dim = min(src_w, src_h)
    tile_w = tile_size_pct / 100.0 * min_dim
    tile_h = tile_w * tile_aspect

    tx = _mirror_fold(dx, tile_w)
    ty = _mirror_fold(dy, tile_h)

    sx = cx_src + tx
    sy = cy_src + ty
    return _bilinear_sample(source, sx, sy, src_w, src_h)


# ── Mode 3: 45-45-90 right triangle ──────────────────────────────────────

def apply_triangle_45(
    source: np.ndarray,
    tile_size_pct: float,
    rotation_deg: float,
    zoom: float,
    center_x_pct: float,
    center_y_pct: float,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    45-45-90 right-triangle tile (8-fold dihedral symmetry of the square).

    The plane is tiled by mirror reflections of a 45-45-90 triangle.
    Steps:
      1. Mirror-fold in X and Y to reduce to [0, tile_s/2]²
      2. Fold along the diagonal (swap if tx > ty) to reduce to the
         lower-left triangle where ty ≤ tx.
    """
    dx, dy, cx_src, cy_src, src_w, src_h, *_ = _common_setup(
        source, rotation_deg, zoom, center_x_pct, center_y_pct, output_size
    )

    tile_s = tile_size_pct / 100.0 * min(src_w, src_h)

    tx = _mirror_fold(dx, tile_s)   # [0, tile_s/2]
    ty = _mirror_fold(dy, tile_s)   # [0, tile_s/2]

    # Reflect across the diagonal y = x  →  keep region where tx >= ty
    need_swap = tx < ty
    tx_f = np.where(need_swap, ty, tx)
    ty_f = np.where(need_swap, tx, ty)

    sx = cx_src + tx_f
    sy = cy_src + ty_f
    return _bilinear_sample(source, sx, sy, src_w, src_h)


# ── Mode 4: Equilateral triangle (60-60-60) ───────────────────────────────

def apply_triangle_60(
    source: np.ndarray,
    tile_size_pct: float,
    rotation_deg: float,
    zoom: float,
    center_x_pct: float,
    center_y_pct: float,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Equilateral-triangle tile (60-60-60) using the triangular lattice.

    Algorithm:
      1. Express (dx, dy) in the triangular lattice basis
           a1 = (s, 0),  a2 = (s/2, s√3/2)
         giving lattice coordinates (u, v).
      2. Reduce to the unit parallelogram: u_frac, v_frac ∈ [0, 1).
      3. Mirror-fold the upper triangle (u+v ≥ 1) to the lower triangle
         via the true reflection across the shared edge:
           (u, v)  →  (1-v, 1-u)
      4. Convert (u_fold, v_fold) back to Cartesian source offsets.
    """
    dx, dy, cx_src, cy_src, src_w, src_h, *_ = _common_setup(
        source, rotation_deg, zoom, center_x_pct, center_y_pct, output_size
    )

    s = tile_size_pct / 100.0 * min(src_w, src_h)
    sqrt3 = np.sqrt(3.0)

    # Lattice coordinates
    v_lat = dy * 2.0 / (s * sqrt3)
    u_lat = dx / s - v_lat * 0.5

    # Fractional part → unit parallelogram [0, 1)²
    u_frac = u_lat - np.floor(u_lat)
    v_frac = v_lat - np.floor(v_lat)

    # True mirror reflection across shared edge (u + v = 1):
    #   upper (u+v ≥ 1)  →  (1-v, 1-u)
    upper = (u_frac + v_frac) >= 1.0
    u_fold = np.where(upper, 1.0 - v_frac, u_frac)
    v_fold = np.where(upper, 1.0 - u_frac, v_frac)

    # Back to Cartesian
    tx = u_fold * s + v_fold * s * 0.5
    ty = v_fold * s * sqrt3 * 0.5

    sx = cx_src + tx
    sy = cy_src + ty
    return _bilinear_sample(source, sx, sy, src_w, src_h)


# ── Mode 5: 30-60-90 right triangle ──────────────────────────────────────

def apply_triangle_30_60(
    source: np.ndarray,
    tile_size_pct: float,
    rotation_deg: float,
    zoom: float,
    center_x_pct: float,
    center_y_pct: float,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    30-60-90 right-triangle tile.

    Two such triangles form a rectangle with sides 1 : √3.
    Steps:
      1. Mirror-fold in X (period = tile_w) and Y (period = tile_w × √3)
         to land in [0, tile_w/2] × [0, tile_w√3/2].
      2. Reflect any point above the hypotenuse across it using the
         standard point-reflection formula for the line:
           tx / (tile_w/2) + ty / (tile_h/2) = 1
    """
    dx, dy, cx_src, cy_src, src_w, src_h, *_ = _common_setup(
        source, rotation_deg, zoom, center_x_pct, center_y_pct, output_size
    )

    tile_w = tile_size_pct / 100.0 * min(src_w, src_h)
    tile_h = tile_w * np.sqrt(3.0)

    tx = _mirror_fold(dx, tile_w)   # [0, tile_w/2]
    ty = _mirror_fold(dy, tile_h)   # [0, tile_h/2]

    # Hypotenuse: (tx / hw) + (ty / hh) = 1
    # where hw = tile_w/2, hh = tile_h/2
    # Equivalent line equation: A*tx + B*ty = 1
    #   A = 2/tile_w,  B = 2/tile_h
    A = 2.0 / tile_w
    B = 2.0 / tile_h
    AB2 = A * A + B * B  # = 4/tile_w² + 4/tile_h² = 4(1 + 1/3)/tile_w² = 16/(3 tile_w²)

    # Points above hypotenuse: A*tx + B*ty > 1
    d = A * tx + B * ty - 1.0   # signed distance (unnormalised)
    above = d > 0.0

    # Reflect across hypotenuse: p' = p - 2*(A,B)*d / AB2
    tx_r = tx - 2.0 * A * d / AB2
    ty_r = ty - 2.0 * B * d / AB2

    tx_f = np.where(above, tx_r, tx)
    ty_f = np.where(above, ty_r, ty)

    sx = cx_src + tx_f
    sy = cy_src + ty_f
    return _bilinear_sample(source, sx, sy, src_w, src_h)


# ── Top-level dispatcher ──────────────────────────────────────────────────

MODES = {
    "radial":         "Radial",
    "rectangle":      "Rectangle",
    "triangle_45":    "Triangle 45-45-90",
    "triangle_60":    "Triangle 60-60-60",
    "triangle_30_60": "Triangle 30-60-90",
}


def apply_effect(mode: str, source: np.ndarray, params: dict,
                 output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Dispatch to the correct transformation function."""
    r = params.get("rotation_deg", 0.0)
    z = params.get("zoom", 1.0)
    cx = params.get("center_x_pct", 50.0)
    cy = params.get("center_y_pct", 50.0)

    if mode == "radial":
        return apply_kaleidoscope(
            source,
            num_segments=params.get("num_segments", 8),
            rotation_deg=r, zoom=z,
            center_x_pct=cx, center_y_pct=cy,
            output_size=output_size,
        )
    ts = params.get("tile_size_pct", 50.0)
    if mode == "rectangle":
        return apply_rectangle(
            source, ts, params.get("tile_aspect", 1.0),
            r, z, cx, cy, output_size,
        )
    if mode == "triangle_45":
        return apply_triangle_45(source, ts, r, z, cx, cy, output_size)
    if mode == "triangle_60":
        return apply_triangle_60(source, ts, r, z, cx, cy, output_size)
    if mode == "triangle_30_60":
        return apply_triangle_30_60(source, ts, r, z, cx, cy, output_size)
    raise ValueError(f"Unknown mode: {mode!r}")
