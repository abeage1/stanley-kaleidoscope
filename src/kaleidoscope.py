"""
Pure NumPy kaleidoscope transformation engine.
No loops — fully vectorized polar coordinate transform.
"""

import numpy as np


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
    Apply a radial kaleidoscope effect to a source image.

    Parameters
    ----------
    source        : H×W×C uint8 NumPy array (RGB or RGBA)
    num_segments  : number of mirror segments (2–24)
    rotation_deg  : rotation offset in degrees
    zoom          : zoom factor (1.0 = no zoom)
    center_x_pct  : sample center X as percent of image width (0–100)
    center_y_pct  : sample center Y as percent of image height (0–100)
    output_size   : (width, height) of output; defaults to source size

    Returns
    -------
    uint8 NumPy array of same channel count as source
    """
    src_h, src_w = source.shape[:2]
    channels = source.shape[2] if source.ndim == 3 else 1

    if output_size is not None:
        out_w, out_h = output_size
    else:
        out_w, out_h = src_w, src_h

    # Center of the output canvas
    cx_out = out_w / 2.0
    cy_out = out_h / 2.0

    # Center of sampling in source image
    cx_src = (center_x_pct / 100.0) * src_w
    cy_src = (center_y_pct / 100.0) * src_h

    # Build meshgrid of output pixel coordinates
    xs = np.arange(out_w, dtype=np.float64)
    ys = np.arange(out_h, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys)  # shape: (out_h, out_w)

    # Shift to output center
    dx = xg - cx_out
    dy = yg - cy_out

    # Convert to polar
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)

    # Add rotation offset
    rotation_rad = np.deg2rad(rotation_deg)
    theta = theta + rotation_rad

    # Normalize theta to [0, 2π)
    two_pi = 2.0 * np.pi
    theta = theta % two_pi

    # Fold into one segment: normalize to [0, 2π/N)
    segment_angle = two_pi / num_segments
    theta_norm = theta % segment_angle

    # Mirror alternate half: fold at π/N
    half_segment = segment_angle / 2.0
    mirror_mask = theta_norm > half_segment
    theta_norm[mirror_mask] = segment_angle - theta_norm[mirror_mask]

    # Convert back to Cartesian, applying zoom
    r_zoomed = r / zoom
    sx = r_zoomed * np.cos(theta_norm) + cx_src
    sy = r_zoomed * np.sin(theta_norm) + cy_src

    # Bilinear interpolation
    result = _bilinear_sample(source, sx, sy, src_w, src_h, channels)
    return result.astype(np.uint8)


def _bilinear_sample(
    source: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    src_w: int,
    src_h: int,
    channels: int,
) -> np.ndarray:
    """Vectorized bilinear interpolation."""
    # Clamp coordinates to valid range
    sx_clamped = np.clip(sx, 0, src_w - 1.001)
    sy_clamped = np.clip(sy, 0, src_h - 1.001)

    x0 = np.floor(sx_clamped).astype(np.int32)
    y0 = np.floor(sy_clamped).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)

    # Fractional parts
    fx = (sx_clamped - x0).astype(np.float32)[..., np.newaxis]
    fy = (sy_clamped - y0).astype(np.float32)[..., np.newaxis]

    # Sample four corners
    c00 = source[y0, x0].astype(np.float32)
    c10 = source[y0, x1].astype(np.float32)
    c01 = source[y1, x0].astype(np.float32)
    c11 = source[y1, x1].astype(np.float32)

    # Bilinear blend
    result = (
        c00 * (1 - fx) * (1 - fy)
        + c10 * fx * (1 - fy)
        + c01 * (1 - fx) * fy
        + c11 * fx * fy
    )
    return result
