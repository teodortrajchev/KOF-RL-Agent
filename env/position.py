# env/position.py
#
#   Determine whether the enemy is currently to the player's left or right,
#   and how far away (normalized). This is needed because block direction
#   and several movement decisions are relative to enemy position, not
#   absolute screen position.

#   - Assumes the two fighters are the largest moving foreground blobs in
#     the play-field band. Hit sparks / projectiles / UI flashes are noise
#     and can occasionally cause a bad frame — that's why `valid` exists.

#   - Identity ("which blob is me")

import numpy as np
import cv2


class PositionTracker:
    def __init__(
        self,
        playfield_band: tuple[float, float] = (0.35, 0.85),
        assume_self_starts_left: bool = True,
        min_contour_area: int = 150,
        history: int = 300,
        var_threshold: int = 32,

    ):
        self.playfield_band = playfield_band
        self.assume_self_starts_left = assume_self_starts_left
        self.min_contour_area = min_contour_area
        self._history = history
        self._var_threshold = var_threshold

        self._bg = self._new_bg_subtractor()
        self._self_x = None
        self._enemy_x = None
        self._initialized = False

        self._stale_frames = 0
        self.max_stale_frames = 5

    def _new_bg_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=False,
        )

    def reset(self):
        self._bg = self._new_bg_subtractor()
        self._self_x = None
        self._enemy_x = None
        self._initialized = False
        self._stale_frames = 0

    def update(self, frame_bgr: np.ndarray) -> dict:

        h, w = frame_bgr.shape[:2]
        y0 = int(h * self.playfield_band[0])
        y1 = int(h * self.playfield_band[1])
        band = frame_bgr[y0:y1]

        fg_mask = self._bg.apply(band)
        fg_mask = cv2.medianBlur(fg_mask, 5)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]
        contours.sort(key=cv2.contourArea, reverse=True)

        if len(contours) < 2:
            self._stale_frames += 1
            return self._result(w)

        centroids = []
        for c in contours[:2]:
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            centroids.append(m["m10"] / m["m00"])

        if len(centroids) < 2:
            self._stale_frames += 1
            return self._result(w)

        centroids.sort()
        left_x, right_x = centroids[0], centroids[1]

        self._stale_frames = 0
        self._bg.apply(band, learningRate=0.002)
        if not self._initialized:
            if self.assume_self_starts_left:
                self._self_x, self._enemy_x = left_x, right_x
            else:
                self._self_x, self._enemy_x = right_x, left_x
            self._initialized = True
        else:
            prev_self, prev_enemy = self._self_x, self._enemy_x
            cost_keep = abs(left_x - prev_self) + abs(right_x - prev_enemy)
            cost_swap = abs(right_x - prev_self) + abs(left_x - prev_enemy)
            if cost_keep <= cost_swap:
                self._self_x, self._enemy_x = left_x, right_x
            else:
                self._self_x, self._enemy_x = right_x, left_x

        return self._result(w)

    def _result(self, width: int) -> dict:
        if self._self_x is None or self._enemy_x is None:
            # default to "enemy to the right" rather than silently blocking the wrong way every time.
            return {"enemy_is_right": True, "dx_norm": 0.0, "valid": False}

        dx = self._enemy_x - self._self_x
        valid = self._stale_frames <= self.max_stale_frames
        return {
            "enemy_is_right": dx >= 0,
            "dx_norm": float(np.clip(dx / width, -1.0, 1.0)) if valid else 0.0,
            "valid": valid,
        }
1