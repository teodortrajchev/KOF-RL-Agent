import time
import pydirectinput

pydirectinput.PAUSE = 0.0


class GameController:
    """
    PPO-friendly controller.

    ACTION SPACE (16 actions):
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
      10 block                (direction-aware — see execute_action)
      11 buff (O button)
      12 crouching light kick (low poke)
      13 crouching heavy punch (anti-poke)
      14 backdash              (direction-aware — retreats from enemy)
      15 forward rush + heavy punch (direction-aware — closes distance)

    Actions 10, 14, 15 need to know where the enemy is, since "block" or
    "retreat" or "advance" only mean something relative to the enemy's
    position. Pass `enemy_is_right` into execute_action for these — this
    replaces the old random.choice() block direction, which was only
    correct about half the time.
    """

    def __init__(self, hold_duration: float = 0.06):
        self.hold_duration = hold_duration
        self.buff_state = 0

    # ACTION DISPATCH
    def execute_action(self, action: int, enemy_is_right: bool = True):

        # idle
        if action == 0:
            time.sleep(self.hold_duration)
            return

        # movement
        if action == 1:
            self._tap("a")

        elif action == 2:
            self._tap("d")

        # jump forward attack
        elif action == 3:
            self._combo(["w", "d"])

        # crouch
        elif action == 4:
            self._hold("s")

        # light punch
        elif action == 5:
            self._tap("u")

        # heavy punch
        elif action == 6:
            self._tap("j")

        # light kick
        elif action == 7:
            self._tap("i")

        # heavy kick
        elif action == 8:
            self._tap("k")

        # anti-air combo
        elif action == 9:
            self._combo(["w", "j"])

        # block — hold TOWARD the enemy (this is how blocking works in
        # most 2D fighters: you hold "back", i.e. the direction the enemy
        # is on, not away from them)
        elif action == 10:
            self._directional_hold(enemy_is_right)

        # buff system
        elif action == 11:
            self._use_buff()

        # crouching light kick (low poke)
        elif action == 12:
            self._combo(["s", "i"])

        # crouching heavy punch (anti-poke, more commitment)
        elif action == 13:
            self._combo(["s", "j"])

        # backdash — move AWAY from the enemy
        elif action == 14:
            away_key = "a" if enemy_is_right else "d"
            self._tap(away_key)

        # forward rush + heavy punch — move TOWARD the enemy and swing
        elif action == 15:
            toward_key = "d" if enemy_is_right else "a"
            self._combo([toward_key, "j"])
        elif action == 16:
            self._tap("l")

    # BLOCKING
    def _directional_hold(self, enemy_is_right: bool):
        direction = "d" if enemy_is_right else "a"
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
        # The three "modes" are chosen by the game, not by which tap we
        # send — every branch pressed the same key anyway, so this is
        # just a single tap. buff_state is still tracked so the env can
        # expose "how long since last buff press" if useful, and so a
        # future version can read the actual active buff off-screen.
        self.buff_state = (self.buff_state % 3) + 1
        self._tap("o")

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
        for k in ["a", "d", "w", "s", "u", "j", "i", "k", "o", "l"]:
            try:
                pydirectinput.keyUp(k)
            except Exception:
                pass

    @staticmethod
    def action_count():
        return 17