"""Minimal PNG writer using only Python stdlib — no GUI/display required."""

import struct
import zlib

import numpy as np


_VIRIDIS = [
    ( 68,   1,  84), ( 72,  36, 117), ( 64,  67, 135), ( 52,  94, 141),
    ( 41, 120, 142), ( 32, 144, 140), ( 34, 167, 132), ( 68, 190, 112),
    (121, 209,  81), (189, 222,  38), (253, 231,  37),
]


def write_png(filename, rgb):
    """Write a uint8 (H, W, 3) numpy array as a PNG file."""
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + bytes(rgb[y].astype(np.uint8)) for y in range(h))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag, data):
        payload = tag + data
        return (
            struct.pack(">I", len(data))
            + payload
            + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        )

    with open(filename, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))


def matrix_to_rgb(matrix, vmin=None, vmax=None, cmap=None):
    """Map a 2D float matrix to a uint8 RGB array using viridis-like colormap."""
    if cmap is None:
        cmap = _VIRIDIS
    if vmin is None:
        vmin = float(matrix.min())
    if vmax is None:
        vmax = float(matrix.max())
    span = vmax - vmin if vmax > vmin else 1.0
    normed = np.clip((matrix.astype(float) - vmin) / span, 0.0, 1.0)

    n = len(cmap) - 1
    idx = normed * n
    lo = np.floor(idx).astype(int).clip(0, n - 1)
    hi = (lo + 1).clip(0, n)
    frac = (idx - lo)[..., np.newaxis]

    c0 = np.array([cmap[i] for i in lo.ravel()], dtype=float).reshape(*lo.shape, 3)
    c1 = np.array([cmap[i] for i in hi.ravel()], dtype=float).reshape(*hi.shape, 3)
    return (c0 + frac * (c1 - c0)).clip(0, 255).astype(np.uint8)


def tile_images(images, ncols):
    """Tile a list of equal-sized RGB arrays into a grid."""
    h, w = images[0].shape[:2]
    nrows = (len(images) + ncols - 1) // ncols
    canvas = np.full((nrows * h, ncols * w, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, ncols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img
    return canvas
