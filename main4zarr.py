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
    'CMO_MoveTo':   {'dataset_name':'DiffusionQL_MoveTo_V3.zarr', 'n_agent': 1, 's_dim': 3, 'a_dim': 4, 'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'normalize', 'eval_freq': 1, 'num_epochs': 2000, 'gn': 5.0, 'top_k': 1},
    'CMO_1V1':      {'dataset_name':'DiffusionQL_1v1_V2.zarr', 'n_agent': 1, 's_dim': 8, 'a_dim': 4, 'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'normalize', 'eval_freq': 1, 'num_epochs': 100, 'gn': 5.0, 'top_k': 1},
    'CMO_3V3':      {'dataset_name':'DiffusionQL_3v3_V2.zarr', 'n_agent': 3, 'state_l_dim': 29, 'state_g_dim': 27, 'a_dim': 4, 'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'normalize', 'eval_freq': 1, 'num_epochs': 100, 'gn': 5.0, 'top_k': 1},
    'CMO_7V7':      {'dataset_name':'DiffusionQL_7v7_V1.zarr', 'n_agent': 7, 'state_l_dim': 65, 'state_g_dim': 63, 'a_dim': 4, 'lr': 3e-4, 'eta': 1.0, 'max_q_backup': False, 'reward_tune': 'normalize', 'eval_freq': 1, 'num_epochs': 100, 'gn': 5.0, 'top_k': 1}
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
from typing import Dict
import numpy as np
import argparse
import zarr
from tqdm import tqdm

def load_dataset_from_zarr(
    dataset_path: str,
    action_dim: int,
    args: argparse.Namespace
) -> Dict[str, np.ndarray]:
    root = zarr.open(dataset_path, mode='r')
    data_grp = root["data"]

    if args.n_agent > 1:
        # === Read Multi Agent ===
        local_all  = data_grp["local_state"][:]          # (T_tot , n_agent, obs_l)
        global_all = data_grp["global_state"][:]         # (T_tot , obs_g)
        act_raw    = data_grp["action"][:]               # (T_tot , n_agent, 1)
        rew_all    = data_grp["reward"][:]               # (T_tot , n_agent, 1)
    else:
        # === Read Single Agent ===
        obs_all = data_grp["state"][:]                   # (T_tot , obs_dim)
        act_raw = data_grp["action"][:]                  # (T_tot ,)
        rew_all = data_grp["reward"][:]                  # (T_tot ,)
    done_all   = data_grp["done"][:]
    ends       = root["meta"]["episode_ends"][:]

    # -------- 動作 one-hot --------
    num_actions = action_dim if action_dim else int(act_raw.max()) + 1
    # 對任意 shape 的 act_raw one-hot
    act_flat = act_raw.reshape(-1)
    act_oh   = np.eye(num_actions, dtype=np.float32)[act_flat]
    act_all  = act_oh.reshape(*act_raw.shape, num_actions)
    if act_raw.shape[-1] == 1: act_all = act_all.squeeze(2) # TODO 這裡之後要改成應對 n agent是1 的狀況

    # -------- episode 逐段切 --------
    segs = []
    prev = 0
    for end in ends:
        segs.append(slice(prev, end))
        prev = end

    # 預先收集 list
    out = dict(
        actions=[], rewards=[], terminals=[]
    )
    if args.n_agent > 1:
        out.update(dict(
            local_state=[], next_local_state=[],
            global_state=[], next_global_state=[]
        ))
    else:
        out.update(dict(
            observations=[], next_observations=[]
        ))

    for sl in tqdm(segs, desc="slice episodes"):
        nxt = slice(sl.start + 1, sl.stop + 1)          # 對應下一步
        if nxt.stop > act_all.shape[0]:                 # 最後一步沒有 next
            break

        done_seg = done_all[sl]
        term_seg = done_seg.astype(bool)

        if args.n_agent > 1:
            loc = local_all[sl]
            glo = global_all[sl]
            loc_next = local_all[nxt]
            glo_next = global_all[nxt]
            act_seg = act_all[sl]
            rew_seg = rew_all[sl]

            # append
            out["local_state"].append(loc)
            out["next_local_state"].append(loc_next)
            out["global_state"].append(glo)
            out["next_global_state"].append(glo_next)
            out["actions"].append(act_seg)
            out["rewards"].append(rew_seg)
            out["terminals"].append(term_seg[:, None])   # (L,1)
        else:
            obs  = obs_all[sl]
            obs2 = obs_all[nxt]
            out["observations"].append(obs)
            out["next_observations"].append(obs2)
            out["actions"].append(act_all[sl])
            out["rewards"].append(rew_all[sl])
            out["terminals"].append(term_seg)

    # -------- concat 回傳 --------
    for k, v in out.items():
        out[k] = np.concatenate(v, axis=0)

    return out

# --------------------------------------------------------
# Main training loop (only diff: dataset loader & eval normalized score)
# --------------------------------------------------------
def train_agent(dataset_name: str, max_action: float,
                device: str, output_dir: str, args: argparse.Namespace) -> None:

    # 1) -------- Load offline buffer ---------------------------------------
    dataset = load_dataset_from_zarr(dataset_name, action_dim=args.a_dim, args=args)
    data_sampler = Data_Sampler(dataset, device, args.reward_tune, args.n_agent > 1)
    utils.print_banner("Loaded zarr buffer")

    # 2) -------- Build agent -----------------------------------------------
    agent = None
    if args.n_agent > 1:
        from agents.ma_ql_diffusion import Diffusion_QL as Agent
        agent = Agent(args.state_l_dim, args.state_g_dim, args.a_dim, args.n_agent, max_action, device,
                discount=args.discount, tau=args.tau,
                max_q_backup=args.max_q_backup, beta_schedule=args.beta_schedule,
                n_timesteps=args.T, eta=args.eta, lr=args.lr,
                lr_decay=args.lr_decay, lr_maxt=args.num_epochs,
                grad_norm=args.gn)
    else:
        from agents.ql_diffusion import Diffusion_QL as Agent
        agent = Agent(args.s_dim, args.a_dim, max_action, device,
                        discount=args.discount, tau=args.tau,
                        max_q_backup=args.max_q_backup, beta_schedule=args.beta_schedule,
                        n_timesteps=args.T, eta=args.eta, lr=args.lr,
                        lr_decay=args.lr_decay, lr_maxt=args.num_epochs,
                        grad_norm=args.gn)

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
        if args.save_model_per_iter is not None and curr_epoch % args.save_model_per_iter == 0:
            agent.save_model(output_dir, curr_epoch)

# ---------------------------------------------------------------------------
# Argument parsing & entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # ----- experiment settings -------------------------------------------
    p.add_argument("--exp", default="1")
    p.add_argument("--device", default=0, type=int)
    p.add_argument("--env_name")
    p.add_argument("--dir", default="results")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--num_steps_per_epoch", default=1000, type=int)
    p.add_argument('--save_model_per_iter', default=10)

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

    args = p.parse_args()

    # ---------- preset from table ----------------------------------------
    preset = hyperparameters[args.env_name]
    for k, v in preset.items():
        setattr(args, k, v)

    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    # ---------- logging dir ----------------------------------------------
    fname = f"{args.env_name}|exp_{args.exp}|diffusion-ql|T-{args.T}"
    if args.lr_decay:
        fname += "|lr_decay"

    results_dir = Path(args.dir) / sanitize(fname)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---------- set random seeds -----------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------- build env & log ------------------------------------------
    setup_logger(results_dir.name, variant=vars(args), log_dir=str(results_dir))
    if args.n_agent > 1:
        utils.print_banner(f"Env: {args.env_name}, Local State Dim: {args.state_l_dim}, Global State Dim: {args.state_g_dim}, Action Dim: {args.a_dim}")
    else:
        utils.print_banner(f"Env: {args.env_name}, State Dim: {args.s_dim}, Action Dim: {args.a_dim}")

    # ---------- train -----------------------------------------------------
    train_agent(
        dataset_name=args.dataset_name,
        max_action=1.0, # 因為 one hot 所以是 1
        device=args.device,
        output_dir=str(results_dir),
        args=args
    )
