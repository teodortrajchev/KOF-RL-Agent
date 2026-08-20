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

    def execute_action(
        self,
        action: int,
        enemy_is_right: bool = True
    ):
        km = self.keymap

        # ----------------------------------------------------
        # IDLE
        # ----------------------------------------------------

        if action == 0:
            time.sleep(self.hold_duration)
            return

        # ----------------------------------------------------
        # MOVEMENT
        # ----------------------------------------------------

        if action == 1:
            self._tap(km["left"])

        elif action == 2:
            self._tap(km["right"])

        # ----------------------------------------------------
        # JUMP FORWARD ATTACK
        # ----------------------------------------------------

        elif action == 3:
            toward_key = (
                km["right"]
                if enemy_is_right
                else km["left"]
            )

            self._combo([
                km["up"],
                toward_key
            ])

        # ----------------------------------------------------
        # CROUCH
        # ----------------------------------------------------

        elif action == 4:
            self._hold(km["down"])

        # ====================================================
        # ATTACKS
        # ====================================================

        # Light punch
        elif action == 5:
            self._tap(km["punch_lt"])

        # Heavy punch
        elif action == 6:
            self._tap(km["punch_hv"])

        # Light kick
        elif action == 7:
            self._tap(km["kick_lt"])

        # Heavy kick
        elif action == 8:
            self._tap(km["kick_hv"])

        # ----------------------------------------------------
        # ANTI-AIR COMBO
        # ----------------------------------------------------

        elif action == 9:
            self._combo([
                km["up"],
                km["punch_hv"]
            ])

        # ----------------------------------------------------
        # BLOCK
        # ----------------------------------------------------

        elif action == 10:
            self._directional_hold(enemy_is_right)

        # ----------------------------------------------------
        # BUFF
        # ----------------------------------------------------

        elif action == 11:
            self._use_buff()

        # ----------------------------------------------------
        # CROUCHING LIGHT KICK
        # ----------------------------------------------------

        elif action == 12:
            self._combo([
                km["down"],
                km["kick_lt"]
            ])

        # ----------------------------------------------------
        # CROUCHING HEAVY PUNCH
        # ----------------------------------------------------

        elif action == 13:
            self._combo([
                km["down"],
                km["punch_hv"]
            ])

        # ----------------------------------------------------
        # BACKDASH
        # ----------------------------------------------------

        elif action == 14:
            away_key = (
                km["left"]
                if enemy_is_right
                else km["right"]
            )

            self._tap(away_key)

        # ----------------------------------------------------
        # FORWARD RUSH + HEAVY PUNCH
        # ----------------------------------------------------

        elif action == 15:
            toward_key = (
                km["right"]
                if enemy_is_right
                else km["left"]
            )

            self._combo([
                toward_key,
                km["punch_hv"]
            ])

        # ----------------------------------------------------
        # EXTRA
        # ----------------------------------------------------

        elif action == 16:
            self._tap(km["extra"])

    # ========================================================
    # BLOCKING
    # ========================================================

    def _directional_hold(self, enemy_is_right: bool):

        direction = (
            self.keymap["right"]
            if enemy_is_right
            else self.keymap["left"]
        )

        try:
            self._key_down(direction)

            time.sleep(
                self.hold_duration * 2
            )

            self._key_up(direction)

        except Exception:
            pass

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

    @staticmethod
    def action_count():
        return 171