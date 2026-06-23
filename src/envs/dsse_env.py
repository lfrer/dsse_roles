
from typing import Dict, Any
from DSSE import DroneSwarmSearch
from pprint import pp

def make_search_env(config: Dict[str, Any]) -> DroneSwarmSearch:

    env = DroneSwarmSearch(
        grid_size=config.get("grid_size", 40),
        render_mode=config.get("render_mode", "ansi"),
        render_grid=config.get("render_grid", False),
        render_gradient=config.get("render_gradient", False),
        vector=tuple(config.get("vector", (1, 1))),
        timestep_limit=config.get("timestep_limit", 300),
        person_amount=config.get("person_amount", 1),
        dispersion_inc=config.get("dispersion_inc", 0.05),
        person_initial_position=tuple(config.get("person_initial_position", (15, 15))),
        drone_amount=config.get("drone_amount", 2),
        drone_speed=config.get("drone_speed", 10),
        probability_of_detection=config.get("probability_of_detection", 0.9),
        pre_render_time=config.get("pre_render_time", 0),
    )

    env_reset_options = {
        "drones_positions": [tuple(p) for p in config.get("drones_positions", [])],
        "person_pod_multipliers": config.get("person_pod_multipliers", [1.0]),
        "vector": tuple(config.get("vector", (1, 1))),
    }

    return env, env_reset_options
