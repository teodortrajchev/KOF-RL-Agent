
#   Computes the scalar reward signal for each timestep.


from dataclasses import dataclass


@dataclass
class GameState:
    """
    Snapshot of the game at a single timestep.
    All HP values are floats in [0.0, 1.0]
    """
    player_hp: float = 1.0        # Current player health (0 = dead)
    enemy_hp: float  = 1.0        # Current enemy health  (0 = dead)
    is_round_over: bool = False   # True when a round-end screen is detected
    player_won: bool    = False   # True if player won the last round


class RewardCalculator:

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

        #Return the scalar reward for one environment step.

        reward = 0.0

        #Survival bonus
        # Given every step.  Motivates the agent to stay alive longer.
        reward += self.survival_bonus


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

        #Dense reward based on HP changes this step.

        # v2 formula

        enemy_hp_lost  = prev.enemy_hp  - curr.enemy_hp    # positive = good
        player_hp_lost = prev.player_hp - curr.player_hp   # positive = bad

        return self.hp_delta_scale * (enemy_hp_lost - player_hp_lost)



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
