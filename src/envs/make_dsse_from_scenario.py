from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import AllPositionsWrapper, RetainDronePosWrapper

from src.envs.RewardScheme import SparseProbSearchWrapper
from src.envs.randomized_vec_env import RandomVectorAndStartDroneSwarmSearch
from typing import Dict, Any


def make_dsse_env_from_scenario(scenario_config: Dict[str, Any]):
    randomize = scenario_config.get("random_vector", False)
    timelimit = scenario_config.get("timestep_limit", 300)

    if randomize:
#        print("Randomizing vector and start pos")
        env = RandomVectorAndStartDroneSwarmSearch(
            grid_size=scenario_config.get("grid_size", 40),
            render_mode=scenario_config.get("render_mode", "ansi"),
            render_grid=scenario_config.get("render_grid", False),
            render_gradient=scenario_config.get("render_gradient", False),
            vector=tuple(scenario_config.get("vector", (0.3, 0.3))),
            vector_min=scenario_config.get("vector_min", (-0.5, -0.5)),
            vector_max=scenario_config.get("vector_max", (0.5, 0.5)),
            timestep_limit=timelimit,
            person_amount=scenario_config.get("person_amount", 1),
            dispersion_start=scenario_config.get("dispersion_start", 0.01),
            dispersion_inc=scenario_config.get("dispersion_inc", 0.02),
            person_initial_position=tuple(scenario_config.get("person_initial_position", (20, 20))),
            drone_amount=scenario_config.get("drone_amount", 2),
            drone_speed=scenario_config.get("drone_speed", 10),
            probability_of_detection=scenario_config.get("probability_of_detection", 0.9),
            pre_render_time=scenario_config.get("pre_render_time", 0),
        )
#        print("start pos:", env.person_initial_position)
    else:
        env = DroneSwarmSearch(
            grid_size=scenario_config.get("grid_size", 40),
            render_mode=scenario_config.get("render_mode", "ansi"),
            render_grid=scenario_config.get("render_grid", False),
            render_gradient=scenario_config.get("render_gradient", False),
            vector=tuple(scenario_config.get("vector", (0.3, 0.3))),
            timestep_limit=timelimit,
            person_amount=scenario_config.get("person_amount", 1),
            dispersion_start=scenario_config.get("dispersion_start", 0.01),
            dispersion_inc=scenario_config.get("dispersion_inc", 0.02),
            person_initial_position=tuple(scenario_config.get("person_initial_position", (20, 20))),
            drone_amount=scenario_config.get("drone_amount", 2),
            drone_speed=scenario_config.get("drone_speed", 10),
            probability_of_detection=scenario_config.get("probability_of_detection", 0.9),
            pre_render_time=scenario_config.get("pre_render_time", 0),
        )

    env = AllPositionsWrapper(env)

    if scenario_config.get("sparse_prob_wrapper", False):
        env = SparseProbSearchWrapper(
            env,
            step_penalty=scenario_config.get("step_penalty", -0.001),
            search_cost=scenario_config.get("search_cost", 0.0),
            prob_threshold=scenario_config.get("prob_threshold", 0.02),
            search_bonus=scenario_config.get("search_bonus", 0.02),
        )

    drones_positions = scenario_config.get("drones_positions", None)
    if drones_positions:
        env = RetainDronePosWrapper(env, [tuple(p) for p in drones_positions])

    return env
