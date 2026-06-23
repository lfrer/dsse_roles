# src/make_pics.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pygame

from src.envs.make_dsse_from_scenario import make_dsse_env_from_scenario


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def capture_pygame_surface() -> np.ndarray:
    surface = pygame.display.get_surface()
    if surface is None:
        raise RuntimeError(
            "No pygame display surface found. "
            "If you are on a headless server, try xvfb-run."
        )

    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2))  # (W,H,C) -> (H,W,C)


def save_frame(frame: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 6), dpi=300)
    plt.imshow(frame)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def step_env_random(env, steps: int) -> None:

    for _ in range(steps):
        if not getattr(env, "agents", None):
            break

        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_space(agent).sample()

        _, _, terminations, truncations, _ = env.step(actions)

        if all(terminations.values()) or all(truncations.values()):
            break


def render_scenario(
    config_path: Path,
    output_dir: Path,
    seed: int = 42,
    steps: int = 10,
) -> Path:
    cfg = load_json(config_path)

    cfg["render_mode"] = "human"
    cfg["render_grid"] = True
    cfg["render_gradient"] = True
    cfg["pre_render_time"] = 10

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config_path.stem}_t{steps}_inittest.png"

    env = make_dsse_env_from_scenario(cfg)

    print("Starting position:", env.person_initial_position)

    try:
        try:
            env.reset(seed=seed)
        except TypeError:
            env.reset()

        if steps > 0:
            step_env_random(env, steps)

        env.render()
        pygame.event.pump()
        pygame.display.flip()

        frame = capture_pygame_surface()
        save_frame(frame, out_path)

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--outdir", type=str, default="pics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.outdir)

    for cfg_str in args.configs:
        cfg_path = Path(cfg_str)
        out_path = render_scenario(
            cfg_path,
            output_dir,
            seed=args.seed,
            steps=args.steps,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
