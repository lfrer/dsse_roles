from ray.tune.stopper import Stopper

class PlateauStopper(Stopper):
    def __init__(
        self,
        metric: str = "evaluation/env_runners/episode_return_mean",
        mode: str = "max",
        patience: int = 20,
        min_iter: int = 30,
        min_delta: float = 1e-3,
        max_iters: int | None = None,
        max_env_steps: int = 200_000_000,
    ):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_iter = min_iter
        self.min_delta = min_delta
        self.max_iters = max_iters
        self.max_env_steps = max_env_steps

        self.best = None
        self.bad_count = 0
        print("Init Plateaustopper with "+str(self.metric) )

    def __call__(self, trial_id, result):
        val = result.get(self.metric)
        it = int(result.get("training_iteration", 0))

        if self.max_iters is not None and it >= self.max_iters:
            return True
        if self.max_env_steps is not None:
            steps = result.get("num_env_steps_sampled_lifetime") or result.get("timesteps_total")
            if steps is not None and int(steps) >= self.max_env_steps:
                return True

        if it < self.min_iter:
            return False

        val = result.get(self.metric)
        if val is None:
            return False

        val = float(val)

        if self.best is None:
            self.best = val
            self.bad_count = 0
            return False

        improved = False
        if self.mode == "max":
            improved = val > self.best + self.min_delta
        else:
            improved = val < self.best - self.min_delta

        if improved:
            self.best = val
            self.bad_count = 0
        else:
            self.bad_count += 1

        return self.bad_count >= self.patience

    def stop_all(self):
        return False
