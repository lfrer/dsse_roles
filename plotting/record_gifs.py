import argparse
import re
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from ray.rllib.models import ModelCatalog

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",  type=str, required=True)
parser.add_argument("--scenario",    type=str, required=True)
parser.add_argument(
    "--model",
    type=str,
    choices=["cnn", "cnn_roles", "lstm", "lstm_roles"],
    default="cnn_roles",
)
parser.add_argument("--n",           type=int, default=10,   help="Number of GIFs")
parser.add_argument("--fps",         type=int, default=5,    help="Frames per second")
parser.add_argument("--output-dir",  type=str, default="gifs")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

if args.model == "cnn":
    from src.models.CNNModel import CNNModel
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

elif args.model == "cnn_roles":
    from src.models.CNNModelR3DM import CNNModelR3DM
    from src.policies.r3dm_cnn_policy import R3DMRolePPOTorchPolicyCNN  # noqa: F401
    ModelCatalog.register_custom_model("CNNModelR3DM", CNNModelR3DM)

elif args.model == "lstm":
    from src.models.CNNLSTMModel import CNNLSTMModel
    ModelCatalog.register_custom_model("CNNLSTMModel", CNNLSTMModel)

elif args.model == "lstm_roles":
    from src.models.CNNLSTMModelR3DM import CNNLSTMModelR3DM
    from src.policies.r3dm_lstm_policy import R3DMRolePPOTorchPolicyLSTM  # noqa: F401
    ModelCatalog.register_custom_model("CNNLSTMModelR3DM", CNNLSTMModelR3DM)

from src.tests.test_helpers import (
    load_scenario_cfg,
    resolve_checkpoint_path,
    build_env,
    init_ray_once,
    load_agent,
    register_dsse_env,
)
from src.recorder import PygameRecord

cfg = load_scenario_cfg(args.scenario)
grid_size = cfg.get("grid_size", 40)


cfg["render_mode"] = "human" 

env = build_env(cfg)
if args.model == "cnn_roles":
    register_dsse_env("DSSE_PPO_CNN_R3DM")
    policy_name = "shared_policy"
elif args.model == "cnn":
    register_dsse_env("DSSE_PPO_Baseline")
    policy_name = "default_policy"
init_ray_once()

ckpt = resolve_checkpoint_path(args.checkpoint)
agent = load_agent(ckpt)

is_recurrent = args.model in ("lstm", "lstm_roles")
has_roles = args.model in ("cnn_roles", "lstm_roles")

if is_recurrent:
    policy = agent.get_policy(policy_name)
    init_state = policy.get_initial_state()
else:
    policy = agent.get_policy(policy_name)


ROLE_COLORS = [
    (230, 230,  50),  
    ( 50, 130, 220),   
]
ROLE_NAMES = ["Role 0", "Role 1"]

def agent_index(aid: str) -> int:
    m = re.search(r"\d+$", aid)
    return int(m.group()) if m else 0


def get_drone_positions(env) -> dict[str, tuple[int, int]]:

    for attr in ("agents_positions", "drones_positions", "_agents_positions"):
        raw = getattr(env, attr, None)
        if raw is not None:
            if isinstance(raw, dict):
                return {k: tuple(v) for k, v in raw.items()}
            if isinstance(raw, (list, np.ndarray)):
                agents = list(env.agents) if hasattr(env, "agents") else []
                if len(agents) == len(raw):
                    return {a: tuple(raw[i]) for i, a in enumerate(agents)}
                return {f"drone{i}": tuple(raw[i]) for i in range(len(raw))}
    return {}


try:
    _font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
    _font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 11)
except Exception:
    _font = ImageFont.load_default()
    _font_sm = _font


def draw_role_rings(img: Image.Image,
                    drone_positions: dict,
                    role_ids: dict,
                    grid_size: int) -> Image.Image:
    w, h = img.size
    cell_w = w / grid_size
    cell_h = h / grid_size
    draw = ImageDraw.Draw(img)

    for aid, pos in drone_positions.items():
        col, row = pos 
        role = role_ids.get(aid, 0)
        color = ROLE_COLORS[role % len(ROLE_COLORS)]

        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        r = min(cell_w, cell_h) * 0.42

        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=color + (230,),   
            width=3,
        )
        draw.text(
            (cx - 4, cy - 7),
            str(role),
            fill=color + (255,),
            font=_font_sm,
        )
    return img


def draw_hud(img: Image.Image,
             step: int,
             ep_reward: float,
             found: bool,
             role_counts: dict,  
             has_roles: bool) -> Image.Image:
    draw = ImageDraw.Draw(img, "RGBA")

    lines = [
        f"Step   : {step:>4}",
        f"Reward : {ep_reward:>6.3f}",
        f"Found  : {'YES ✓' if found else 'no'}",
    ]

    if has_roles and role_counts:
        total = sum(role_counts.values()) or 1
        for rid, cnt in sorted(role_counts.items()):
            name = ROLE_NAMES[rid % len(ROLE_NAMES)]
            pct = cnt / total * 100
            color_hex = "#{:02X}{:02X}{:02X}".format(*ROLE_COLORS[rid % len(ROLE_COLORS)])
            lines.append(f"{name[:8]}: {pct:4.0f}%")

    pad = 6
    line_h = 17
    box_w = 160
    box_h = pad * 2 + line_h * len(lines)

    draw.rectangle([4, 4, 4 + box_w, 4 + box_h], fill=(0, 0, 0, 160))

    for i, line in enumerate(lines):
        y = 4 + pad + i * line_h
        if has_roles and i >= 3:
            rid = i - 3
            fill = ROLE_COLORS[rid % len(ROLE_COLORS)] + (255,)
        else:
            fill = (220, 220, 220, 255)
        draw.text((10, y), line, fill=fill, font=_font)

    return img


def annotate_frame(frame_img: Image.Image,
                   step: int,
                   ep_reward: float,
                   found: bool,
                   drone_positions: dict,
                   role_ids: dict,
                   role_counts: dict,
                   grid_size: int,
                   has_roles: bool) -> Image.Image:
    img = frame_img.convert("RGBA")

    if has_roles and drone_positions:
        img = draw_role_rings(img, drone_positions, role_ids, grid_size)

    img = draw_hud(img, step, ep_reward, found, role_counts, has_roles)

    return img.convert("RGB")


import pygame

class AnnotatedRecorder(PygameRecord):

    def __init__(self, filename: str, fps: int, grid_size: int, has_roles: bool):
        super().__init__(filename, fps)
        self.grid_size = grid_size
        self.has_roles = has_roles
        self.step = 0
        self.ep_reward = 0.0
        self.found = False
        self.drone_positions: dict = {}
        self.role_ids: dict = {}
        self.role_counts: dict = {}

    def add_frame(self):
        surf = pygame.display.get_surface()
        arr = pygame.surfarray.array3d(surf)
        arr = np.moveaxis(arr, 0, 1)
        img = Image.fromarray(np.uint8(arr))

        img = annotate_frame(
            img,
            step=self.step,
            ep_reward=self.ep_reward,
            found=self.found,
            drone_positions=self.drone_positions,
            role_ids=self.role_ids,
            role_counts=self.role_counts,
            grid_size=self.grid_size,
            has_roles=self.has_roles,
        )
        self.frames.append(img)


def get_role_id(model) -> int | None:
    rp = getattr(model, "last_role_probs", None)
    if rp is None:
        return None
    try:
        arr = rp.detach().cpu().numpy()
        arr = arr if arr.ndim == 1 else arr[0]
        return int(np.argmax(arr))
    except Exception:
        return None


def run_episode(gif_path: Path):
    rec = AnnotatedRecorder(str(gif_path), args.fps, grid_size, has_roles)

    with rec:
        obs, _ = env.reset()
        states = (
            {aid: [np.array(s, copy=True) for s in init_state] for aid in obs}
            if is_recurrent else None
        )
        ep_reward = 0.0
        found = False
        step = 0
        model = getattr(policy, "model", None)

        while env.agents:
            actions, new_states, cur_role_ids = {}, {}, {}

            for aid, o in obs.items():
                if is_recurrent:
                    a, s_out, _ = agent.compute_single_action(
                        o, state=states[aid], explore=False, policy_id= policy_name,
                    )
                    new_states[aid] = s_out
                else:
                    a = agent.compute_single_action(o, policy_id=policy_name, explore=False)

                actions[aid] = a

                if has_roles and model is not None:
                    rid = get_role_id(model)
                    if rid is not None:
                        cur_role_ids[aid] = rid

            obs, rws, terms, truncs, info = env.step(actions)

            ep_reward += float(sum(rws.values()))
            step += 1

            if not found:
                found = any(
                    v for d in [terms, truncs] for v in d.values()
                ) and ep_reward > 0

            for aid, rid in cur_role_ids.items():
                rec.role_counts[rid] = rec.role_counts.get(rid, 0) + 1

            positions = get_drone_positions(env)

            rec.step = step
            rec.ep_reward = ep_reward
            rec.found = found
            rec.drone_positions = positions
            rec.role_ids = cur_role_ids

            if is_recurrent:
                states = new_states

            rec.add_frame()


scenario_stem = Path(args.scenario).stem
print(f"Recording {args.n} GIFs  →  {output_dir}/")
print(f"Model: {args.model}  |  Roles overlay: {has_roles}")

for i in range(args.n):
    gif_path = output_dir / f"{args.model}_{scenario_stem}_ep{i:03d}.gif"
    print(f"  [{i+1}/{args.n}] {gif_path.name}")
    run_episode(gif_path)

env.close()
print("Done.")
