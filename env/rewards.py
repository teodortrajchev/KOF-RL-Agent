# env/rewards.py
#
#   Computes the scalar reward signal for each timestep.
#
# VERSION 1 — placeholder rewards only:
#   • +0.1  every step (survival bonus — discourages dying fast)
#   • Hooks for win/loss bonuses are present but return 0 until

#   • Parse HP bar pixel values from vision.py's raw BGR frame
#   • Return  Δ(enemy_hp) − Δ(player_hp)  as dense reward signal
#   • Add a large bonus/penalty on round win/loss

from dataclasses import dataclass


@dataclass
class GameState:
    """
    Snapshot of the game at a single timestep.

    All HP values are floats in [0.0, 1.0] (fraction of full health).
    In v1 these are all placeholder defaults — v2 will populate them
    from real pixel-based HP bar detection.
    """
    player_hp: float = 1.0        # Current player health (0 = dead)
    enemy_hp: float  = 1.0        # Current enemy health  (0 = dead)
    is_round_over: bool = False   # True when a round-end screen is detected
    player_won: bool    = False   # True if player won the last round


class RewardCalculator:
    """
    Computes the per-step reward given the previous and current GameState.

    Parameters
    ----------
    survival_bonus : float
        Small positive reward added every step the player is alive.
        Keeps the agent from standing still and getting killed instantly.
    win_reward : float
        Reward given when the player wins a round.
    loss_penalty : float
        Penalty (negative reward) when the player loses a round.
    hp_delta_scale : float
        Multiplier for the HP-difference reward (used in v2+).
    """

    def __init__(
        self,
        survival_bonus: float = 0.1,
        win_reward:     float = 5.0,
        loss_penalty:   float = -5.0,
        hp_delta_scale: float = 10.0,
    ):
        self.survival_bonus  = survival_bonus
        self.win_reward      = win_reward
        self.loss_penalty    = loss_penalty
        self.hp_delta_scale  = hp_delta_scale

    #Public API

    def compute(
        self,
        prev_state: GameState,
        curr_state: GameState,
    ) -> float:
        """
        Return the scalar reward for one environment step.

        Parameters
        ----------
        prev_state : GameState  — state at the START of the step
        curr_state : GameState  — state at the END of the step

        Returns
        -------
        float — the reward signal passed to the RL agent
        """
        reward = 0.0

        #Survival bonus
        # Given every step.  Motivates the agent to stay alive longer.
        reward += self.survival_bonus

        # HP delta reward (v2 — currently returns 0)
        # Will be:  reward += hp_delta_scale * (Δenemy_hp − Δplayer_hp)
        # Positive when agent damages the enemy more than it takes damage.
        reward += self._hp_delta_reward(prev_state, curr_state)

        # Round outcome bonus
        if curr_state.is_round_over:
            if curr_state.player_won:
                reward += self.win_reward
            else:
                reward += self.loss_penalty

        return reward


    def _hp_delta_reward(
        self,
        prev: GameState,
        curr: GameState,
    ) -> float:
        """
        Dense reward based on HP changes this step.

        v1: returns 0.0 (HP values are placeholders).
        v2: uncomment the formula below once HP detection is working.
        """
        # ── v2 formula (uncomment when HP detection is ready) ──
        # enemy_hp_lost  = prev.enemy_hp  - curr.enemy_hp    # positive = good
        # player_hp_lost = prev.player_hp - curr.player_hp   # positive = bad
        # return self.hp_delta_scale * (enemy_hp_lost - player_hp_lost)

        # v1 placeholder
        return 0.0


if __name__ == "__main__":
    calc = RewardCalculator()

    # Scenario A: normal step, no HP change (v1 behaviour)
    s0 = GameState(player_hp=1.0, enemy_hp=1.0)
    s1 = GameState(player_hp=1.0, enemy_hp=1.0)
    r  = calc.compute(s0, s1)
    print(f"Scenario A (no HP change):  reward = {r:.2f}  (expected 0.10)")

    # Scenario B: player wins round
    s2 = GameState(player_hp=0.8, enemy_hp=0.0, is_round_over=True, player_won=True)
    r  = calc.compute(s1, s2)
    print(f"Scenario B (player wins):   reward = {r:.2f}  (expected 5.10)")

    # Scenario C: player loses round
    s3 = GameState(player_hp=0.0, enemy_hp=0.5, is_round_over=True, player_won=False)
    r  = calc.compute(s1, s3)
    print(f"Scenario C (player loses):  reward = {r:.2f}  (expected -4.90)")
