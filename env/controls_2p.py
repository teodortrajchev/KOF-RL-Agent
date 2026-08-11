import time
import pydirectinput

pydirectinput.PAUSE = 0.0


# PER-PLAYER KEYMAPS

# Every action in GameController is expressed in terms of these logical
# roles (left / right / up / down / punch_lt / kick_lt / buff / punch_hv /
# kick_hv / extra). Swapping the keymap is all that's needed to drive P1
# vs P2 — none of the action-dispatch logic changes.

P1_KEYMAP = {
    "left": "a",
    "right": "d",
    "up": "w",
    "down": "s",
    "punch_lt": "u",   # light punch
    "kick_lt": "i",    # light kick
    "buff": "o",
    "punch_hv": "j",   # heavy punch
    "kick_hv": "k",    # heavy kick
    "extra": "l",
}

P2_KEYMAP = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "punch_lt": "num4",
    "kick_lt": "num5",
    "buff": "num6",
    "punch_hv": "num1",
    "kick_hv": "num2",
    "extra": "num3",
}


class GameController:
    """
    PPO-friendly controller.

    ACTION SPACE (17 actions):
      0  idle
      1  move left
      2  move right
      3  jump-forward attack   (direction-aware — see execute_action)
      4  crouch
      5  light punch
      6  heavy punch
      7  light kick
      8  heavy kick
      9  anti-air combo
      10 block                (direction-aware — see execute_action)
      11 buff (buff button)
      12 crouching light kick (low poke)
      13 crouching heavy punch (anti-poke)
      14 backdash              (direction-aware — retreats from enemy)
      15 forward rush + heavy punch (direction-aware — closes distance)
      16 extra button tap

    Actions 3, 10, 14, 15 need to know where the enemy is, since "block" or
    "retreat" or "advance" only mean something relative to the enemy's
    position. Pass `enemy_is_right` into execute_action for these.

    For 2-player (agent vs agent) use, construct one GameController per
    player with the matching keymap:
        p1 = GameController(keymap=P1_KEYMAP)
        p2 = GameController(keymap=P2_KEYMAP)
    Each instance only ever presses its own player's keys, so both can be
    driven independently within the same game tick without conflicts.
    """

    def __init__(self, hold_duration: float = 0.06, keymap: dict | None = None):
        self.hold_duration = hold_duration
        self.buff_state = 0
        self.keymap = keymap or P1_KEYMAP

    # ACTION DISPATCH
    def execute_action(self, action: int, enemy_is_right: bool = True):
        km = self.keymap

        # idle
        if action == 0:
            time.sleep(self.hold_duration)
            return

        # movement
        if action == 1:
            self._tap(km["left"])

        elif action == 2:
            self._tap(km["right"])

        # jump forward attack — direction-aware
        elif action == 3:
            toward_key = km["right"] if enemy_is_right else km["left"]
            self._combo([km["up"], toward_key])

        # crouch
        elif action == 4:
            self._hold(km["down"])


        # ATTACKS

        # light punch
        elif action == 5:
            self._tap(km["punch_lt"])

        # heavy punch
        elif action == 6:
            self._tap(km["punch_hv"])

        # light kick
        elif action == 7:
            self._tap(km["kick_lt"])

        # heavy kick
        elif action == 8:
            self._tap(km["kick_hv"])

        # anti-air combo
        elif action == 9:
            self._combo([km["up"], km["punch_hv"]])

        # block — hold TOWARD the enemy
        elif action == 10:
            self._directional_hold(enemy_is_right)

        # buff system
        elif action == 11:
            self._use_buff()

        # crouching light kick (low poke)
        elif action == 12:
            self._combo([km["down"], km["kick_lt"]])

        # crouching heavy punch (anti-poke, more commitment)
        elif action == 13:
            self._combo([km["down"], km["punch_hv"]])

        # backdash — move AWAY from the enemy
        elif action == 14:
            away_key = km["left"] if enemy_is_right else km["right"]
            self._tap(away_key)

        # forward rush + heavy punch — move TOWARD the enemy and swing
        elif action == 15:
            toward_key = km["right"] if enemy_is_right else km["left"]
            self._combo([toward_key, km["punch_hv"]])

        elif action == 16:
            self._tap(km["extra"])

    # BLOCKING (direction-aware)
    def _directional_hold(self, enemy_is_right: bool):
        direction = self.keymap["right"] if enemy_is_right else self.keymap["left"]
        try:
            pydirectinput.keyDown(direction)
            time.sleep(self.hold_duration * 2)
            pydirectinput.keyUp(direction)
        except Exception:
            pass

    # BUFF SYSTEM
    def tap_key(self, key: str):
        """Public single-key tap, for non-gameplay actions like menu
        confirmation (round-end / continue screens in match mode)."""
        self._tap(key)

    def _use_buff(self):
        self.buff_state = (self.buff_state % 3) + 1
        self._tap(self.keymap["buff"])

    # PRIMITIVES
    def _tap(self, key: str):
        pydirectinput.keyDown(key)
        time.sleep(self.hold_duration)
        pydirectinput.keyUp(key)

    def _hold(self, key: str):
        pydirectinput.keyDown(key)
        time.sleep(self.hold_duration * 2)
        pydirectinput.keyUp(key)

    def _combo(self, keys: list[str]):
        for k in keys:
            pydirectinput.keyDown(k)

        time.sleep(self.hold_duration)

        for k in keys:
            pydirectinput.keyUp(k)

    def release_all(self):
        for k in self.keymap.values():
            try:
                pydirectinput.keyUp(k)
            except Exception:
                pass

    @staticmethod
    def action_count():
        return 17