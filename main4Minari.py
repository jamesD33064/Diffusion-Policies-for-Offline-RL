from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import minari
import numpy as np
import torch
import json
from tqdm import tqdm

from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo
from agents.ql_diffusion import Diffusion_QL as Agent

# --------------------------------------------------------
# Globals ────────────── hyper-parameters per dataset
# --------------------------------------------------------
hyperparameters: Dict[str, Dict] = {
    'halfcheetah-medium-v2':         {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 1},
    'hopper-medium-v2':              {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 9.0,  'top_k': 2},
    'walker2d-medium-v0':            {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'Walker2d-v5':                   {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 1.0,  'top_k': 1},
    'halfcheetah-medium-replay-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 2.0,  'top_k': 0},
    'hopper-medium-replay-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 2},
    'walker2d-medium-replay-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 4.0,  'top_k': 1},
    'halfcheetah-medium-expert-v2':  {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 7.0,  'top_k': 0},
    'hopper-medium-expert-v2':       {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 2},
    'walker2d-medium-expert-v2':     {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'Walker2d-v5':                   {'lr': 3e-4, 'eta': 1.0,   'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 1, 'num_epochs': 2000, 'gn': 5.0,  'top_k': 1},
    'antmaze-umaze-v0':              {'lr': 3e-4, 'eta': 0.5,   'max_q_backup': False,  'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 2},
    'antmaze-umaze-diverse-v0':      {'lr': 3e-4, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 3.0,  'top_k': 2},
    'antmaze-medium-play-v0':        {'lr': 1e-3, 'eta': 2.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 2.0,  'top_k': 1},
    'antmaze-medium-diverse-v0':     {'lr': 3e-4, 'eta': 3.0,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 1.0,  'top_k': 1},
    'antmaze-large-play-v0':         {'lr': 3e-4, 'eta': 4.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'antmaze-large-diverse-v0':      {'lr': 3e-4, 'eta': 3.5,   'max_q_backup': True,   'reward_tune': 'cql_antmaze', 'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 1},
    'pen-human-v1':                  {'lr': 3e-5, 'eta': 0.15,  'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 7.0,  'top_k': 2},
    'pen-cloned-v1':                 {'lr': 3e-5, 'eta': 0.1,   'max_q_backup': False,  'reward_tune': 'normalize',   'eval_freq': 50, 'num_epochs': 1000, 'gn': 8.0,  'top_k': 2},
    'kitchen-complete-v2':           {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 250 , 'gn': 9.0,  'top_k': 2},
    'kitchen-partial-v0':            {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 2},
    'kitchen-mixed-v0':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
    'FrankaKitchen-v1':              {'lr': 3e-4, 'eta': 0.005, 'max_q_backup': False,  'reward_tune': 'no',          'eval_freq': 50, 'num_epochs': 1000, 'gn': 10.0, 'top_k': 0},
}

# --------------------------------------------------------
# Helper ─────────────── Windows-safe path
# --------------------------------------------------------
_illegal = re.compile(r"[<>:\\|/?*]")

def sanitize(path: str | Path) -> str:
    """Replace Windows-illegal characters with '_'"""
    return _illegal.sub("_", str(path))

# --------------------------------------------------------
# Dataset loading via Minari
# --------------------------------------------------------

def load_dataset_from_minari(env_name: str) -> Dict[str, np.ndarray]:
    """Convert Minari dataset to the dict expected by *Data_Sampler*."""
    ds = minari.load_dataset("mujoco/walker2d/medium-v0")
    _ = ds.recover_environment()

    obs_list, next_obs_list, actions_list, rewards_list, dones_list = [], [], [], [], []

    for epi in ds.iterate_episodes():
        o = epi.observations          # shape (T+1, obs_dim)
        a = epi.actions               # shape (T, action_dim)
        r = epi.rewards               # shape (T,)
        d = epi.terminations | epi.truncations  # shape (T,)

        obs_list.append(o[:-1])
        next_obs_list.append(o[1:])
        actions_list.append(a)
        rewards_list.append(r)
        dones_list.append(d)

    # Concatenate all episodes
    observations      = np.concatenate(obs_list, axis=0)
    next_observations = np.concatenate(next_obs_list, axis=0)
    actions           = np.concatenate(actions_list, axis=0)
    rewards           = np.concatenate(rewards_list, axis=0)
    terminals         = np.concatenate(dones_list, axis=0).astype(bool)

    # observations (999613, 17)
    # next_observations (999613, 17)
    # actions (999613, 6)
    # rewards (999613,)
    # terminals (999613,)

    return {
        "observations":      observations,
        "next_observations": next_observations,
        "actions":           actions,
        "rewards":           rewards,
        "terminals":         terminals,
    }

# --------------------------------------------------------
# Main training loop (only diff: dataset loader & eval normalized score)
# --------------------------------------------------------

def train_agent(env: gym.Env, state_dim: int, action_dim: int, max_action: float,
                device: str, output_dir: str, args: argparse.Namespace) -> None:

    # 1) -------- Load offline buffer ---------------------------------------
    dataset = load_dataset_from_minari(args.env_name)
    # for i in dataset.keys():
    #     print(i, dataset[i].shape)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner("Loaded Minari buffer")

    # 2) -------- Build agent -----------------------------------------------
    from agents.ql_diffusion import Diffusion_QL as Agent
    if args.algo == "ql":
        agent = Agent(state_dim, action_dim, max_action, device,
                      discount=args.discount, tau=args.tau,
                      max_q_backup=args.max_q_backup, beta_schedule=args.beta_schedule,
                      n_timesteps=args.T, eta=args.eta, lr=args.lr,
                      lr_decay=args.lr_decay, lr_maxt=args.num_epochs,
                      grad_norm=args.gn)
    else:
        from agents.bc_diffusion import Diffusion_BC as Agent
        agent = Agent(state_dim, action_dim, max_action, device,
                      discount=args.discount, tau=args.tau,
                      beta_schedule=args.beta_schedule, n_timesteps=args.T, lr=args.lr)

    # 3) -------- Training + Eval -------------------------------------------
    max_steps = args.num_epochs * args.num_steps_per_epoch
    iterations = int(args.eval_freq * args.num_steps_per_epoch)
    evaluations = []
    writer = SummaryWriter(log_dir=output_dir+"/runs/experiment")
    
    metric = 1e8
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.) if args.early_stop else None

    while agent.step < max_steps:
        # train -------------------------------------------------------------
        loss = agent.train(data_sampler, iterations, args.batch_size, writer)

        # log ---------------------------------------------------------------
        curr_epoch = agent.step // args.num_steps_per_epoch
        logger.record_tabular("Epoch", curr_epoch)
        logger.record_tabular("BC Loss", np.mean(loss["bc_loss"]))
        logger.dump_tabular()

        # eval --------------------------------------------------------------
        res = eval_policy(agent, args.env_name, args.seed, args.eval_episodes)
        evaluations.append([*res, curr_epoch])
        np.save(Path(output_dir) / "eval.npy", evaluations)
        if writer is not None:
            writer.add_scalar("Eval/Normalized Return", res[2], agent.step)
            writer.add_scalar("Eval/Raw Return", res[0], agent.step)

        # early stop / model selection -------------------------------------
        metric = np.mean(loss["bc_loss"])
        if stop_check and stop_check(metric, metric):
            break

        # save model -------------------------------------
        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline
    scores = np.array(evaluations)
    if args.ms == 'online':
        best_id = np.argmax(scores[:, 2])
        best_res = {'model selection': args.ms, 'epoch': scores[best_id, -1],
                    'best normalized score avg': scores[best_id, 2],
                    'best normalized score std': scores[best_id, 3],
                    'best raw score avg': scores[best_id, 0],
                    'best raw score std': scores[best_id, 1]}
        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))
    elif args.ms == 'offline':
        bc_loss = scores[:, 4]
        top_k = min(len(bc_loss) - 1, args.top_k)
        where_k = np.argsort(bc_loss) == top_k
        best_res = {'model selection': args.ms, 'epoch': scores[where_k][0][-1],
                    'best normalized score avg': scores[where_k][0][2],
                    'best normalized score std': scores[where_k][0][3],
                    'best raw score avg': scores[where_k][0][0],
                    'best raw score std': scores[where_k][0][1]}

        with open(os.path.join(output_dir, f"best_score_{args.ms}.txt"), 'w') as f:
            f.write(json.dumps(best_res))

# ---------------------------------------------------------------------------
# Evaluation util (Gymnasium env + simple normalization table) --------------
# ---------------------------------------------------------------------------
_normalization: Dict[str, Tuple[float, float]] = {
    # raw-score → normalized = 100 * (raw - min) / (max - min)
    "hopper-v4": (-20.0, 3234.3),
    "walker2d-v4": (0.0, 5000.0),
    "halfcheetah-v4": (0.0, 12000.0),
}

def get_normalized_score(env: gym.Env, raw_return: float) -> float:
    key = env.spec.id.split("-")[0] + "-v4"
    if key in _normalization:
        min_r, max_r = _normalization[key]
        return 100 * (raw_return - min_r) / (max_r - min_r)
    return raw_return  # fallback


def eval_policy(policy, env_name: str, seed: int, eval_episodes: int = 10):
    env = gym.make(env_name)
    env.reset(seed=seed + 42)

    raw_returns = []
    norm_returns = []
    for _ in range(eval_episodes):
        done, obs, ep_return = False, env.reset()[0], 0.0
        while not done:
            action = policy.sample_action(np.asarray(obs))
            obs, reward, done, _, _ = env.step(action)
            ep_return += reward
        raw_returns.append(ep_return)
        norm_returns.append(get_normalized_score(env, ep_return))

    return (np.mean(raw_returns), np.std(raw_returns),
            np.mean(norm_returns), np.std(norm_returns))


def record_policy(policy, env_name: str, save_dir: str = "./video", seed: int = 0):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=save_dir, episode_trigger=lambda e: True)
    env.reset(seed=seed)
    obs, _ = env.reset()
    done = False
    from tqdm import tqdm
    
    pbar = tqdm(desc="Env Steps", unit="step")
    while not done:
        action = policy.sample_action(np.asarray(obs))
        obs, reward, done, _, _ = env.step(action)
        pbar.update(1)
    pbar.close()

    env.close()
    print(f"影片已儲存到：{save_dir}")

# ---------------------------------------------------------------------------
# Argument parsing & entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # ----- experiment settings -------------------------------------------
    p.add_argument("--exp", default="exp_1")
    p.add_argument("--device", default=0, type=int)
    p.add_argument("--env_name", default="walker2d-medium-expert-v2")
    p.add_argument("--dir", default="results")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_steps_per_epoch", default=1_000, type=int)
    p.add_argument('--save_best_model', default=True, action='store_true')

    # ----- optimisation ---------------------------------------------------
    p.add_argument("--batch_size", default=256, type=int)
    p.add_argument("--lr_decay", action="store_true")
    p.add_argument("--early_stop", action="store_true")

    # ----- RL -------------------------------------------------------------
    p.add_argument("--discount", default=0.99, type=float)
    p.add_argument("--tau", default=0.005, type=float)

    # ----- diffusion ------------------------------------------------------
    p.add_argument("--T", default=5, type=int)
    p.add_argument("--beta_schedule", default="vp")

    # ----- algo -----------------------------------------------------------
    p.add_argument("--algo", default="ql", choices=["ql", "bc"])
    p.add_argument("--ms", default="offline", choices=["online", "offline"])
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of episodes for policy evaluation")

    args = p.parse_args()

    # ---------- preset from table ----------------------------------------
    preset = hyperparameters[args.env_name]
    for k, v in preset.items():
        setattr(args, k, v)

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # ---------- logging dir ----------------------------------------------
    fname = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay:
        fname += "|lr_decay"
    fname += f"|ms-{args.ms}|{args.seed}"

    results_dir = Path(args.dir) / sanitize(fname)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------- set random seeds -----------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- build env & log ------------------------------------------
    env = gym.make(args.env_name)
    env.reset(seed=args.seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    max_a = float(env.action_space.high[0])

    setup_logger(results_dir.name, variant=vars(args), log_dir=str(results_dir))
    utils.print_banner(f"Env: {args.env_name}, s_dim: {s_dim}, a_dim: {a_dim}")

    # ---------- train -----------------------------------------------------
    train_agent(env, s_dim, a_dim, max_a, args.device, str(results_dir), args)

    # ---------- Record Video -----------------------------------------------------
    # env_name = "Walker2d-v5"
    # env = gym.make(env_name)
    # s_dim = env.observation_space.shape[0]
    # a_dim = env.action_space.shape[0]
    # max_a = float(env.action_space.high[0])

    # agent = Agent(
    #     state_dim=s_dim,
    #     action_dim=a_dim,
    #     max_action=max_a,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    #     discount=0.99,
    #     tau=0.005,
    #     max_q_backup=False,
    #     eta=1.0,
    #     beta_schedule="vp",
    #     n_timesteps=5,
    #     lr=3e-4,
    #     lr_decay=False,
    #     lr_maxt=2000,
    #     grad_norm=5.0,
    # )
    # agent.load_model("results/Walker2d-v5_exp_1_diffusion-ql_T-5_lr_decay_ms-offline_0", id=113)
    # record_policy(agent, env_name, save_dir="video/walker2d_run")
