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
dataset_path = 'DiffusionQL_MoveTo_V3.zarr'
hyperparameters: Dict[str, Dict] = {
    'CMO_MoveTo':   {'s_dim': 3, 'a_dim': 4, 'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'normalize', 'eval_freq': 1, 'num_epochs': 2000, 'gn': 5.0, 'top_k': 1}
}

# --------------------------------------------------------
# Helper ─────────────── Windows-safe path
# --------------------------------------------------------
_illegal = re.compile(r"[<>:\\|/?*]")

def sanitize(path: str | Path) -> str:
    """Replace Windows-illegal characters with '_'"""
    return _illegal.sub("_", str(path))

# --------------------------------------------------------
# Dataset loading via zarr
# --------------------------------------------------------

def load_dataset_from_zarr(dataset_path: str, action_dim: int = None) -> Dict[str, np.ndarray]:
    import zarr
    root = zarr.open(dataset_path, mode='r')
    data_grp = root['data']
    obs_all   = data_grp['state'][:]          # (T+N_epis, obs_dim)
    act_all   = data_grp['action'][:]         # (T+N_epis, act_dim)
    rew_all   = data_grp['reward'][:]         # (T+N_epis,)
    done_all  = data_grp['done'][:]           # (T+N_epis,)
    ends      = root['meta']['episode_ends'][:]

    # action 轉成 one hot
    num_actions = (
        action_dim
        if action_dim is not None
        else int(act.max()) + 1
    )
    act_all = np.eye(num_actions, dtype=np.float32)[act_all]

    # 製作回傳格式
    obs_list, next_obs_list, act_list, rew_list, done_list = [], [], [], [], []

    prev = 0
    for end in tqdm(ends):
        obs   = obs_all[prev:end]
        nexto = obs_all[prev+1:end+1]
        act   = act_all[prev:end]
        rew   = rew_all[prev:end]
        done  = done_all[prev:end]

        if len(nexto) < len(obs):
            obs, act, rew, done = obs[:-1], act[:-1], rew[:-1], done[:-1]
            nexto = nexto[:-1]

        obs_list.append(obs)
        next_obs_list.append(nexto)
        act_list.append(act)
        rew_list.append(rew)
        done_list.append(done)

        prev = end

    observations      = np.concatenate(obs_list, axis=0)
    next_observations = np.concatenate(next_obs_list, axis=0)
    actions           = np.concatenate(act_list, axis=0)
    rewards           = np.concatenate(rew_list, axis=0)
    terminals         = np.concatenate(done_list, axis=0).astype(bool)

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

def train_agent(state_dim: int, action_dim: int, max_action: float,
                device: str, output_dir: str, args: argparse.Namespace) -> None:

    # 1) -------- Load offline buffer ---------------------------------------
    dataset = load_dataset_from_zarr(dataset_path, action_dim=action_dim)
    # for i in dataset.keys():
    #     print(i, dataset[i].shape)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune)
    utils.print_banner("Loaded zarr buffer")

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

        # early stop / model selection -------------------------------------
        metric = np.mean(loss["bc_loss"])
        if stop_check and stop_check(metric, metric):
            break

        # save model -------------------------------------
        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

# ---------------------------------------------------------------------------
# Argument parsing & entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # ----- experiment settings -------------------------------------------
    p.add_argument("--exp", default="1")
    p.add_argument("--device", default=0, type=int)
    p.add_argument("--env_name", default="walker2d-medium-expert-v2")
    p.add_argument("--dir", default="results")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_steps_per_epoch", default=1000, type=int)
    p.add_argument('--save_best_model', default=True, action='store_true')

    # ----- optimisation ---------------------------------------------------
    p.add_argument("--batch_size", default=32, type=int)
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

    args = p.parse_args()

    # ---------- preset from table ----------------------------------------
    preset = hyperparameters[args.env_name]
    for k, v in preset.items():
        setattr(args, k, v)

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # ---------- logging dir ----------------------------------------------
    fname = f"{args.env_name}|exp_{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay:
        fname += "|lr_decay"

    results_dir = Path(args.dir) / sanitize(fname)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------- set random seeds -----------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- build env & log ------------------------------------------
    setup_logger(results_dir.name, variant=vars(args), log_dir=str(results_dir))
    utils.print_banner(f"Env: {args.env_name}, s_dim: {args.s_dim}, a_dim: {args.a_dim}")

    # ---------- train -----------------------------------------------------
    train_agent(
        state_dim=args.s_dim,
        action_dim=args.a_dim,
        max_action=1.0, # 因為 one hot 所以是 1
        device=args.device,
        output_dir=str(results_dir),
        args=args
    )
