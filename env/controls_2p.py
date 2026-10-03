import time
import ctypes
import pydirectinput

pydirectinput.PAUSE = 0.0


# ============================================================
# WINDOWS NUMPAD VIRTUAL-KEY CODES
# ============================================================

VK_NUMPAD1 = 0x61
VK_NUMPAD2 = 0x62
VK_NUMPAD3 = 0x63
VK_NUMPAD4 = 0x64
VK_NUMPAD5 = 0x65
VK_NUMPAD6 = 0x66


# ============================================================
# PER-PLAYER KEYMAPS
# ============================================================

P1_KEYMAP = {
    "left": "a",
    "right": "d",
    "up": "w",
    "down": "s",

    "punch_lt": "u",
    "kick_lt": "i",
    "buff": "o",

    "punch_hv": "j",
    "kick_hv": "k",
    "extra": "l",
}


P2_KEYMAP = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",

    # ACTUAL NUMPAD KEYS
    "punch_lt": VK_NUMPAD4,
    "kick_lt": VK_NUMPAD5,
    "buff": VK_NUMPAD6,

    "punch_hv": VK_NUMPAD1,
    "kick_hv": VK_NUMPAD2,
    "extra": VK_NUMPAD3,
}


class GameController:
    """
    PPO-friendly controller.

    ACTION SPACE (17 actions):

      0  idle
      1  move left
      2  move right
      3  jump-forward attack
      4  crouch
      5  light punch
      6  heavy punch
      7  light kick
      8  heavy kick
      9  anti-air combo
      10 block
      11 buff
      12 crouching light kick
      13 crouching heavy punch
      14 backdash
      15 forward rush + heavy punch
      16 extra button tap

    P1 uses normal keyboard keys.

    P2 uses arrow keys for movement and physical NUMPAD keys
    for attacks.
    """

    def __init__(
        self,
        hold_duration: float = 0.06,
        keymap: dict | None = None
    ):
        self.hold_duration = hold_duration
        self.buff_state = 0
        self.keymap = keymap or P1_KEYMAP

    # ========================================================
    # LOW-LEVEL KEYBOARD HELPERS
    # ========================================================

    @staticmethod
    def _is_numpad_key(key):
        return isinstance(key, int)

    @staticmethod
    def _numpad_down(vk):
        """
        Press an actual Windows NUMPAD key.

        keybd_event receives the Windows virtual-key code.
        """

        ctypes.windll.user32.keybd_event(
            vk,
            0,
            0,
            0
        )

    @staticmethod
    def _numpad_up(vk):
        """
        Release an actual Windows NUMPAD key.
        """

        ctypes.windll.user32.keybd_event(
            vk,
            0,
            2,
            0
        )

    def _key_down(self, key):
        if self._is_numpad_key(key):
            self._numpad_down(key)
        else:
            pydirectinput.keyDown(key)

    def _key_up(self, key):
        if self._is_numpad_key(key):
            self._numpad_up(key)
        else:
            pydirectinput.keyUp(key)

    # ========================================================
    # ACTION DISPATCH
    # ========================================================

    def prepare_action(self, action: int, enemy_is_right: bool = True):
        """Returns (keys_to_press, hold_seconds) without pressing anything."""
        km, h = self.keymap, self.hold_duration
        toward = km["right"] if enemy_is_right else km["left"]
        away = km["left"] if enemy_is_right else km["right"]

        if action == 11:
            self.buff_state = (self.buff_state % 3) + 1

        table = {
            0: ([], h),
            1: ([km["left"]], h),
            2: ([km["right"]], h),
            3: ([km["up"], toward], h),
            4: ([km["down"]], 2 * h),
            5: ([km["punch_lt"]], h),
            6: ([km["punch_hv"]], h),
            7: ([km["kick_lt"]], h),
            8: ([km["kick_hv"]], h),
            9: ([km["up"], km["punch_hv"]], h),
            10: ([away], 2 * h),  # BLOCK = hold AWAY (verify in-game)
            11: ([km["buff"]], h),
            12: ([km["down"], km["kick_lt"]], h),
            13: ([km["down"], km["punch_hv"]], h),
            14: ([away], h),
            15: ([toward, km["punch_hv"]], h),
            16: ([km["extra"]], h),
        }
        return table[action]

    def execute_action(self, action: int, enemy_is_right: bool = True):
        keys, hold = self.prepare_action(action, enemy_is_right)
        for k in keys:
            self._key_down(k)
        time.sleep(hold)
        for k in keys:
            self._key_up(k)

    @staticmethod
    def action_count():
        return 17

    # ========================================================
    # BUFF SYSTEM
    # ========================================================

    def tap_key(self, key):
        """
        Public single-key tap.

        Useful for menu confirmation and other
        non-gameplay actions.
        """

        self._tap(key)

    def _use_buff(self):

        self.buff_state = (
            self.buff_state % 3
        ) + 1

        self._tap(
            self.keymap["buff"]
        )

    # ========================================================
    # PRIMITIVES
    # ========================================================

    def _tap(self, key):

        self._key_down(key)

        time.sleep(
            self.hold_duration
        )

        self._key_up(key)

    def _hold(self, key):

        self._key_down(key)

        time.sleep(
            self.hold_duration * 2
        )

        self._key_up(key)

    def _combo(self, keys):

        # Press all keys
        for key in keys:
            self._key_down(key)

        time.sleep(
            self.hold_duration
        )

        # Release all keys
        for key in keys:
            self._key_up(key)

    # ========================================================
    # RELEASE EVERYTHING
    # ========================================================

    def release_all(self):

        for key in self.keymap.values():

            try:
                self._key_up(key)

            except Exception:
                pass

    # ========================================================
    # ACTION COUNT
    # ========================================================

