import numpy as np
from typing import Any, Tuple
from DSSE import DroneSwarmSearch

class RandomVectorAndStartDroneSwarmSearch(DroneSwarmSearch):
    def __init__(
        self,
        *args: Any,
        vector_min: Tuple[float, float] = (-0.5, -0.5),
        vector_max: Tuple[float, float] = (0.5, 0.5),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._vector_min = np.array(vector_min, dtype=float)
        self._vector_max = np.array(vector_max, dtype=float)

    def _sample_start(self, rng: np.random.Generator) -> Tuple[int, int]:
        x = int(rng.integers(0, self.grid_size))
        y = int(rng.integers(0, self.grid_size))
        return (x, y)

    def _sample_vector(self, rng: np.random.Generator) -> Tuple[float, float]:
        v = rng.uniform(self._vector_min, self._vector_max)
        return float(v[0]), float(v[1])

    def reset(self, *, seed=None, options=None):
        rng = np.random.default_rng(seed)

        vx, vy = self._sample_vector(rng)
        opt = dict(options or {})
        opt["vector"] = (vx, vy)

        start = self._sample_start(rng)
#        print("Setting start to: ", start)
        self.person_initial_position = start
        self.disaster_position = start  # <-- this is the key for centering the probability matrix

        return super().reset(seed=seed, options=opt)
