"""Picture-in-picture camera previews composited onto the heatmap."""

from __future__ import annotations

import cv2
import numpy as np


class CameraPanelOverlay:
    """Embed IR / front camera feeds in a centered row on the heatmap."""

    SLOT_ORDER = ("left", "right", "front")
    SLOT_LABELS = {"left": "Left IR", "right": "Right IR", "front": "Front"}

    def __init__(self, screen_width, screen_height, panel_height=None, margin=16, gap=14):
        self.screen_width = screen_width
        self.screen_height = screen_height
        if panel_height is None:
            panel_height = max(280, min(380, int(screen_height * 0.28)))
        self.panel_height = panel_height
        self.margin = margin
        self.gap = gap
        self.visible = True
        self._frames = {slot: None for slot in self.SLOT_ORDER}
        self._active_slots = []
        self._layout = {}
        self._cluster_bounds = None

    def set_active_slots(self, slot_ids):
        self._active_slots = [slot for slot in self.SLOT_ORDER if slot in slot_ids]
        self._rebuild_layout()

    def set_frame(self, slot_id, frame):
        if slot_id in self._frames:
            self._frames[slot_id] = None if frame is None else frame.copy()

    def toggle(self):
        self.visible = not self.visible
        return self.visible

    def status_line(self):
        state = "ON" if self.visible else "OFF"
        return f"Camera previews: {state} (V toggle)"

    def _rebuild_layout(self):
        self._layout = {}
        self._cluster_bounds = None
        count = len(self._active_slots)
        if count == 0:
            return

        max_cluster_w = int(self.screen_width * 0.75)
        usable_w = max_cluster_w - self.gap * (count - 1)
        panel_w = max(200, usable_w // count)
        total_w = count * panel_w + (count - 1) * self.gap

        x_start = (self.screen_width - total_w) // 2
        y0 = (self.screen_height - self.panel_height) // 2

        for index, slot_id in enumerate(self._active_slots):
            x0 = x_start + index * (panel_w + self.gap)
            self._layout[slot_id] = (x0, y0, panel_w, self.panel_height)

        pad = 10
        self._cluster_bounds = (
            x_start - pad,
            y0 - pad,
            x_start + total_w + pad,
            y0 + self.panel_height + pad,
        )

    def _fit_frame(self, frame, width, height):
        fh, fw = frame.shape[:2]
        scale = min(width / fw, height / fh)
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        offset_x = (width - new_w) // 2
        offset_y = (height - new_h) // 2
        canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
        return canvas, offset_x, offset_y, new_w, new_h

    def overlay_on(self, frame):
        if not self.visible or not self._active_slots:
            return frame

        out = frame.copy()

        if self._cluster_bounds is not None:
            x1, y1, x2, y2 = self._cluster_bounds
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), -1)

        for slot_id in self._active_slots:
            source = self._frames.get(slot_id)
            if source is None:
                continue

            x0, y0, w, h = self._layout[slot_id]
            preview, _, _, _, _ = self._fit_frame(source, w, h)
            out[y0 : y0 + h, x0 : x0 + w] = preview

            label = self.SLOT_LABELS.get(slot_id, slot_id)
            cv2.putText(out, label, (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(out, label, (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.rectangle(out, (x0, y0), (x0 + w, y0 + h), (180, 180, 180), 1)

        return out

    def hit_test(self, x, y):
        """Return slot id and mapped frame coordinates for mouse clicks."""
        if not self.visible:
            return None

        for slot_id in self._active_slots:
            if slot_id not in self._layout:
                continue
            x0, y0, w, h = self._layout[slot_id]
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue

            source = self._frames.get(slot_id)
            if source is None:
                return None

            _, offset_x, offset_y, fitted_w, fitted_h = self._fit_frame(source, w, h)
            local_x = x - x0 - offset_x
            local_y = y - y0 - offset_y
            if not (0 <= local_x < fitted_w and 0 <= local_y < fitted_h):
                return None

            fh, fw = source.shape[:2]
            frame_x = int(local_x * fw / fitted_w)
            frame_y = int(local_y * fh / fitted_h)
            return {"slot": slot_id, "frame_x": frame_x, "frame_y": frame_y}

        return None
