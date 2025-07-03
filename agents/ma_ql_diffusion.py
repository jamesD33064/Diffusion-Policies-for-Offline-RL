import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from agents.diffusion import Diffusion
from agents.model import MLP
from tqdm import tqdm
import copy, itertools
from itertools import chain

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class Diffusion_QL(object):
    def __init__(self,
                 state_l_dim,
                 state_g_dim,
                 action_dim,
                 n_agent,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.n_agent = n_agent

        # -------- Actors & EMA actors ----------
        self.actors      = nn.ModuleList()
        self.ema_actors  = nn.ModuleList()
        for _ in range(n_agent):
            noise_pred_model = MLP(state_dim=state_l_dim, action_dim=action_dim, device=device)
            actor = Diffusion(state_dim=state_l_dim,
                              action_dim=action_dim,
                              model=noise_pred_model,
                              max_action=max_action,
                              beta_schedule=beta_schedule,
                              n_timesteps=n_timesteps).to(device)
            self.actors.append(actor)
            self.ema_actors.append(copy.deepcopy(actor).eval())  # EMA 初始值

        # -------- EMA actors setting ----------
        self.step = 0
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.ema_decay = ema_decay
        # self.register_buffer("step", torch.zeros(()))
            
        # -------- Centralized Critic ----------
        self.critic        = Critic(state_g_dim, n_agent * action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # -------- Optimizers ----------
        self.actor_opt  = torch.optim.Adam(itertools.chain(*(a.parameters() for a in self.actors)), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr*0.3)

        # -------- Schedulers ----------
        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_opt, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_opt, T_max=lr_maxt, eta_min=0.)

    @torch.no_grad()
    # -------- 更新EMA ----------
    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        for ema, actor in zip(self.ema_actors, self.actors):
            for p_ema, p in zip(ema.parameters(), actor.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}

        for _ in tqdm(range(iterations), desc="Training iterations"):
            # Sample replay buffer / batch
            state_l, state_g, action, next_state_l, next_state_g, reward, not_done = replay_buffer.sample(batch_size)
            reward_mean = reward.mean(dim=1, keepdim=True).squeeze(-1)  # 把 n_agent 的 reward 平均變成只有一個reward

            """ Q (Critic) Training """
            joint_a = action.reshape(action.size(0), -1)
            current_q1, current_q2 = self.critic(state_g, joint_a)

            # Double Q-Learning
            a_next_list = []    # 收集各 actor 的 next action (a0t+1)
            for i, ema_actor in enumerate(self.ema_actors):
                obs_i_next = next_state_l[:, i, :]
                a_i_next  = ema_actor(obs_i_next)
                a_next_list.append(a_i_next)
            next_joint_a = torch.cat(a_next_list, dim=-1)   # joint a0t+1

            target_q1, target_q2 = self.critic_target(next_state_g, next_joint_a)
            target_q = torch.min(target_q1, target_q2)

            target_q = (reward_mean + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_opt.step()

            """ Policy (Actor) Training """
            bc_loss_total = 0.0
            new_action_list = []

            for i, actor_i in enumerate(self.actors):
                obs_i   = state_l[:, i, :]        # (B, obs_dim_l)
                act_i   = action[:, i, :]         # (B, action_dim)
                # BC loss
                bc_i = actor_i.loss(act_i, obs_i)
                bc_loss_total += bc_i
                # 產生 new_action
                a_i0 = actor_i(obs_i)
                new_action_list.append(a_i0)

            new_joint_a = torch.cat(new_action_list, dim=-1)  

            q1_new_action, q2_new_action = self.critic(state_g, new_joint_a)
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
            actor_loss = bc_loss_total + self.eta * q_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0:
                actor_grad_norms = nn.utils.clip_grad_norm_(chain(*(a.parameters() for a in self.actors)), self.grad_norm)
            self.actor_opt.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Gradient/Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Gradient/Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Loss/Actor Loss', actor_loss.item(), self.step)
                log_writer.add_scalar('Loss/BC Loss', bc_loss_total.item(), self.step)
                log_writer.add_scalar('Loss/QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Loss/Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss_total.item())
            metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    @torch.no_grad()
    def sample_action(self, state_l, state_g, k_candidate=50):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        # state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        # with torch.no_grad():
        #     action = self.actor.sample(state_rpt)
        #     q_value = self.critic_target.q_min(state_rpt, action).flatten()
        #     idx = torch.multinomial(F.softmax(q_value), 1)
        # return action[idx].cpu().data.numpy().flatten()
    
        # ----- 將 numpy 轉 tensor -----
        obs_l = torch.tensor(state_l, dtype=torch.float32, device=self.device)           # (N, O_l)
        if state_g is not None:
            s_g  = torch.tensor(state_g,  dtype=torch.float32, device=self.device)       # (S_g,)
            s_g  = s_g.unsqueeze(0).repeat(k_candidate, 1)                               # (K, S_g)

        # ----- 為每個 agent 重複觀測 K 次並抽樣 K 個動作 -----
        cand_actions = []                              # list[(K, act_dim)]
        for i, actor_i in enumerate(self.actors):
            obs_i_rep = obs_l[i].unsqueeze(0).repeat(k_candidate, 1)   # (K, O_l)
            a_i_k     = actor_i.sample(obs_i_rep)                      # (K, act_dim)
            cand_actions.append(a_i_k)

        # ----- 拼成 K 組 joint-action 向量 -----
        joint_a = torch.cat(cand_actions, dim=-1)      # (K, N*act_dim)

        # ----- 用 critic 打分 -----
        if state_g is None:
            raise ValueError("central critic 需要 state_g 輸入")
        q_val  = self.critic_target.q_min(s_g, joint_a).squeeze(-1)    # (K,)

        # ----- 依 softmax 機率抽 1 組，或直接 argmax -----
        # idx = torch.multinomial(torch.softmax(q_val, dim=0), 1).item()   # 隨機（探勘）
        idx = torch.argmax(q_val).item()                                 # 貪婪（利用）

        # ----- 拆回 per-agent 動作 -----
        act_dim = cand_actions[0].shape[1]
        action_out = []
        for j in range(self.n_agent):
            a_j = joint_a[idx, j*act_dim:(j+1)*act_dim]                  # (act_dim,)
            action_out.append(a_j.cpu().numpy())

        return np.stack(action_out, axis=0)        # (n_agent, act_dim)

    def save_model(self, dir, epoch=None):
        os.makedirs(dir, exist_ok=True)
        # --------- 儲存所有 actors ----------
        for idx, actor in enumerate(self.actors):
            fname = f"actor{idx}.pth" if epoch is None else f"actor{idx}_{epoch}.pth"
            torch.save(actor.state_dict(), os.path.join(dir, fname))
        # --------- 儲存 centralized critic ----------
        critic_name = "critic.pth" if epoch is None else f"critic_{epoch}.pth"
        torch.save(self.critic.state_dict(), os.path.join(dir, critic_name))

        print(f"[Checkpoint] saved to: {dir} (epoch={epoch})")

    def load_model(self, dir, epoch=None, map_location=None):    
        map_location = map_location or self.device
        # ---------- 讀取每個 actor ----------
        for idx, actor in enumerate(self.actors):
            fname = f"actor{idx}.pth" if epoch is None else f"actor{idx}_{epoch}.pth"
            path  = os.path.join(dir, fname)
            actor.load_state_dict(torch.load(path, map_location=map_location))

        # ---------- 讀取 centralized critic ----------
        critic_name = "critic.pth" if epoch is None else f"critic_{epoch}.pth"
        critic_path = os.path.join(dir, critic_name)
        self.critic.load_state_dict(torch.load(critic_path, map_location=map_location))

        print(f"[Checkpoint] loaded from: {dir} (tag={epoch})")
