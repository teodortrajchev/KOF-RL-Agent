# env/position.py
#
#   Determine whether the enemy is currently to the player's LEFT or RIGHT,
#   and how far away (normalized). This is needed because block direction
#   and several movement decisions are relative to enemy position, not
#   absolute screen position.
#
#   Background-subtraction blob tracking, not template matching. This means
#   it works without a per-character sprite template, but it DOES need
#   tuning against real footage — the defaults here are starting points.
#
#   - Assumes the two fighters are the largest moving foreground blobs in
#     the play-field band. Hit sparks / projectiles / UI flashes are noise
#     and can occasionally cause a bad frame — that's why `valid` exists.
#   - Needs a handful of frames after reset for the background model to
#     settle (MOG2 warm-up). Don't trust results in the very first frames
#     after `reset()`.
#   - Identity ("which blob is me") is tracked frame-to-frame by nearest-
#     neighbour continuity, seeded by `assume_self_starts_left`. If your
#     character actually spawns on the right in KOF Wing, flip that flag.
#   - This will drift/misfire during clinches or when characters overlap.
#     Treat `dx_norm` as a noisy signal, not ground truth — smoothing
#     (e.g. an EMA) is worth adding once you can see it against real frames.

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

    def _new_bg_subtractor(self):
        return cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=False,
        )

    def reset(self):
        """Call this from env.reset() — the background model must not
        carry state across episodes (round-start screens etc. would
        get baked in as 'background')."""
        self._bg = self._new_bg_subtractor()
        self._self_x = None
        self._enemy_x = None
        self._initialized = False

    def update(self, frame_bgr: np.ndarray) -> dict:
        """
        frame_bgr: raw BGR frame (same one used for HP detection — pass the
        SAME captured frame here rather than grabbing a new one, to avoid
        temporal desync between position, HP, and pixel obs).
        """
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
            return self._result(w)

        centroids = []
        for c in contours[:2]:
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            centroids.append(m["m10"] / m["m00"])

        if len(centroids) < 2:
            return self._result(w)

        centroids.sort()
        left_x, right_x = centroids[0], centroids[1]

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
            # No reading yet — default to "enemy to the right" rather than
            # silently blocking the wrong way every time.
            return {"enemy_is_right": True, "dx_norm": 0.0, "valid": False}

        dx = self._enemy_x - self._self_x
        return {
            "enemy_is_right": dx >= 0,
            "dx_norm": float(np.clip(dx / width, -1.0, 1.0)),
            "valid": True,
        }
