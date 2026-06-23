from pettingzoo.utils.wrappers import BaseParallelWrapper

SEARCH_ACTION = 8  # DSSE "Search Cell" action id

class SparseProbSearchWrapper(BaseParallelWrapper):
    def __init__(
        self,
        env,
        *,
        step_penalty: float = -0.001,
        search_cost: float = 0.0,
        prob_threshold: float = 0.02,
        search_bonus: float = 0.02,
    ):
        super().__init__(env)
        self.step_penalty = float(step_penalty)
        self.search_cost = float(search_cost)
        self.prob_threshold = float(prob_threshold)
        self.search_bonus = float(search_bonus)

    @staticmethod
    def _agent_idx(a: str) -> int:
        try:
            return int(a.split("_")[-1])
        except Exception:
            return 0

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        any_obs = next(iter(obs.values()))
        posvec, mat = any_obs
        positions = [(int(posvec[i]), int(posvec[i + 1])) for i in range(0, len(posvec), 2)]

        new_rewards = dict(rewards)  # keep DSSE base rewards (found etc.)

        for a in new_rewards:
            new_rewards[a] += self.step_penalty

            idx = self._agent_idx(a)
            x, y = positions[idx] if idx < len(positions) else positions[0]

            if actions.get(a, SEARCH_ACTION) == SEARCH_ACTION:
                new_rewards[a] += self.search_cost

                p = float(mat[y, x])  # numpy index [row, col]
                if p >= self.prob_threshold:
                    new_rewards[a] += self.search_bonus

        return obs, new_rewards, terminations, truncations, infos
