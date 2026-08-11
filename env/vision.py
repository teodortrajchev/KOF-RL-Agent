# env/vision.py

import numpy as np
import cv2
import mss
import mss.tools


OBS_SIZE: int = 84

DEFAULT_GAME_REGION: dict = {
    "top": 0,
    "left": 0,
    "width": 954,
    "height": 600,
}

# HP BAR REGIONS

PLAYER_HP_REGION = {
    "x": 110,
    "y": 80,
    "w": 205,
    "h": 17,
}

ENEMY_HP_REGION = {
    "x": 450,
    "y": 80,
    "w": 205,
    "h": 17,
}
# Full HP calibration values
PLAYER_HP_MAX = 0.89
ENEMY_HP_MAX = 0.88


class ScreenCapture:

    def __init__(
        self,
        game_region: dict | None = None,
        obs_size: int = OBS_SIZE,
    ):
        self.game_region = game_region or DEFAULT_GAME_REGION
        self.obs_size = obs_size
        self._sct = mss.mss()

    # Unified single-grab capture (USE THIS FROM THE ENV)
    # The old pattern — grab_frame() for pixels, then get_player_hp() and
    # get_enemy_hp() each calling grab_raw_bgr() again — issued THREE
    # separate mss.grab() calls per env.step(). Each grab happens at a
    # slightly different wall-clock moment, so the pixel observation and
    # the HP values used for reward were never actually looking at the same
    # instant, and you were paying 3x the capture cost for no reason.
    # This method grabs once and derives everything from that one frame.

    def get_observation_bundle(self) -> tuple[np.ndarray, float, float, np.ndarray]:
        """
        Returns (obs_frame, player_hp, enemy_hp, raw_bgr).

        obs_frame : (84, 84, 1) grayscale, for the CNN.
        player_hp / enemy_hp : normalized to [0, 1].
        raw_bgr : the original capture, handed back so callers (e.g. the
                  position tracker) can reuse it instead of re-grabbing.
        """
        raw_bgr = self.grab_raw_bgr()

        frame_grey = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        obs_frame = cv2.resize(
            frame_grey,
            (self.obs_size, self.obs_size),
            interpolation=cv2.INTER_AREA,
        )[:, :, np.newaxis]

        player_hp = self._get_hp_percentage(raw_bgr, PLAYER_HP_REGION, reverse=False)
        player_hp = min(player_hp / PLAYER_HP_MAX, 1.0)

        enemy_hp = self._get_hp_percentage(raw_bgr, ENEMY_HP_REGION, reverse=True)
        enemy_hp = min(enemy_hp / ENEMY_HP_MAX, 1.0)

        return obs_frame, player_hp, enemy_hp, raw_bgr

    # Frame Capture (kept for standalone use / backward compatibility)

    def grab_frame(self) -> np.ndarray:
        raw = self._sct.grab(self.game_region)

        frame_bgr = np.array(raw)[..., :3]
        frame_grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        frame_resized = cv2.resize(
            frame_grey,
            (self.obs_size, self.obs_size),
            interpolation=cv2.INTER_AREA,
        )

        return frame_resized[:, :, np.newaxis]

    def grab_raw_bgr(self) -> np.ndarray:
        raw = self._sct.grab(self.game_region)
        # Make a contiguous copy so OpenCV can draw on it
        return np.ascontiguousarray(np.array(raw)[..., :3])

    # HP Detection

    def _get_hp_percentage(
        self,
        frame: np.ndarray,
        region: dict,
        reverse: bool = False,
    ) -> float:

        x = region["x"]
        y = region["y"]
        w = region["w"]
        h = region["h"]

        hp_img = frame[y:y+h, x:x+w]

        hsv = cv2.cvtColor(hp_img, cv2.COLOR_BGR2HSV)

        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([40, 255, 255])

        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        if reverse:
            mask = np.fliplr(mask)

        cols = np.any(mask > 0, axis=0)

        filled_width = np.sum(cols)

        raw_hp = filled_width / w

        # Clamp raw values to [0, 1]
        return min(max(raw_hp, 0.0), 1.0)

    def get_player_hp(self) -> float:
        """Standalone accessor — issues its own grab. Prefer
        get_observation_bundle() from the env to avoid redundant captures."""
        frame = self.grab_raw_bgr()
        raw_hp = self._get_hp_percentage(frame, PLAYER_HP_REGION, reverse=False)
        return min(raw_hp / PLAYER_HP_MAX, 1.0)

    def get_enemy_hp(self) -> float:
        """Standalone accessor — issues its own grab. Prefer
        get_observation_bundle() from the env to avoid redundant captures."""
        frame = self.grab_raw_bgr()
        raw_hp = self._get_hp_percentage(frame, ENEMY_HP_REGION, reverse=True)
        return min(raw_hp / ENEMY_HP_MAX, 1.0)

    # Cleanup

    def close(self) -> None:
        self._sct.close()

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.obs_size, self.obs_size, 1)


# Standalone Test

if __name__ == "__main__":

    print("vision.py test")
    print(f"Capturing region: {DEFAULT_GAME_REGION}")
    print("Press 'q' to quit")

    cap = ScreenCapture()

    while True:

        obs, player_hp, enemy_hp, raw_bgr = cap.get_observation_bundle()

        display = cv2.resize(
            obs[:, :, 0],
            (420, 420),
            interpolation=cv2.INTER_NEAREST,
        )

        cv2.imshow("KOF RL - Observation Preview", display)

        print(
            f"\rPlayer HP: {player_hp:.2f} | Enemy HP: {enemy_hp:.2f}",
            end=""
        )

        cv2.rectangle(
            raw_bgr,
            (PLAYER_HP_REGION["x"], PLAYER_HP_REGION["y"]),
            (
                PLAYER_HP_REGION["x"] + PLAYER_HP_REGION["w"],
                PLAYER_HP_REGION["y"] + PLAYER_HP_REGION["h"],
            ),
            (0, 255, 0),
            2,
        )

        cv2.rectangle(
            raw_bgr,
            (ENEMY_HP_REGION["x"], ENEMY_HP_REGION["y"]),
            (
                ENEMY_HP_REGION["x"] + ENEMY_HP_REGION["w"],
                ENEMY_HP_REGION["y"] + ENEMY_HP_REGION["h"],
            ),
            (0, 0, 255),
            2,
        )

        cv2.imshow("KOF RL - Raw Capture", raw_bgr)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

    cap.close()
    cv2.destroyAllWindows()