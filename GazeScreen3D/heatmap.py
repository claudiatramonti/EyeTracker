"""
Heatmap buffer for GazeScreen3D (accumulate hits on the monitor plane).
"""

from __future__ import annotations

import cv2
import numpy as np


class GazeHeatmap:
    def __init__(self, width, height, decay=0.997, blob_sigma=28.0):
        self.width = width
        self.height = height
        self.decay = decay
        self.blob_sigma = blob_sigma
        self.accum = np.zeros((height, width), dtype=np.float32)
        self.last_hit = None
        self.on_screen = False

    def clear(self):
        self.accum.fill(0.0)
        self.last_hit = None
        self.on_screen = False

    def add_hit(self, u, v, on_screen):
        self.accum *= self.decay
        self.on_screen = bool(on_screen)
        if not on_screen or u is None or v is None:
            self.last_hit = None
            return

        x = int(round(u))
        y = int(round(v))
        if not (0 <= x < self.width and 0 <= y < self.height):
            self.last_hit = None
            self.on_screen = False
            return

        self.last_hit = (x, y)
        radius = max(3, int(self.blob_sigma * 2.5))
        y0 = max(0, y - radius)
        y1 = min(self.height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(self.width, x + radius + 1)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        blob = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * self.blob_sigma**2))
        self.accum[y0:y1, x0:x1] += blob.astype(np.float32)

    def render_bgr(self):
        peak = float(self.accum.max()) if self.accum.size else 0.0
        if peak < 1e-6:
            base = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            norm = np.clip(self.accum / peak, 0.0, 1.0)
            heat_u8 = (norm * 255.0).astype(np.uint8)
            base = cv2.applyColorMap(heat_u8, cv2.COLORMAP_INFERNO)

        if self.last_hit is not None:
            color = (0, 255, 0) if self.on_screen else (0, 0, 255)
            cv2.circle(base, self.last_hit, 14, color, 2, cv2.LINE_AA)
            cv2.circle(base, self.last_hit, 3, color, -1, cv2.LINE_AA)
        return base
