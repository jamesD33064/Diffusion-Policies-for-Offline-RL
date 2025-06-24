# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
from typing import Dict

class Data_Sampler(object):
	def __init__(self,
					data: Dict[str, np.ndarray],
					device: torch.device,
					reward_tune: str = 'no',
					is_MA: bool = False):

		self.device  = device
		self.is_MA   = is_MA

		if self.is_MA:     # ---------- Multi Agent ----------
			self.local_state        = torch.from_numpy(data["local_state"]).float()         # (T , N , obs_l)
			self.next_local_state   = torch.from_numpy(data["next_local_state"]).float()    # (T , N , obs_l)
			self.global_state       = torch.from_numpy(data["global_state"]).float()        # (T , obs_g)
			self.next_global_state  = torch.from_numpy(data["next_global_state"]).float()   # (T , obs_g)

			self.action   = torch.from_numpy(data["actions"]).float()       # (T , N , A)  或 (T , N)
			self.reward   = torch.from_numpy(data["rewards"]).float()       # (T , N)     → 後面會 view(-1,1)
			self.not_done = 1. - torch.from_numpy(data["terminals"]).float()# (T , N)

			T, N, _ = self.local_state.shape
			self.size  = T
			self.N  = N
		else:     # ---------- Single Agent ----------
			self.state      = torch.from_numpy(data['observations']).float()
			self.next_state = torch.from_numpy(data['next_observations']).float()
			self.action     = torch.from_numpy(data['actions']).float()
			self.reward     = torch.from_numpy(data['rewards']).view(-1, 1).float()
			self.not_done   = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()

			self.size       = self.state.shape[0]
			self.state_dim  = self.state.shape[1]
			self.action_dim = self.action.shape[1]

		# ---------- 獎勵調整 ----------
		if reward_tune == 'normalize':
			self.reward = (self.reward - self.reward.mean()) / (self.reward.std() + 1e-8)
		elif reward_tune == 'iql_antmaze':
			self.reward = self.reward - 1.0
		elif reward_tune == 'antmaze':
			self.reward = (self.reward - 0.25) * 2.0

    # -------------------------------------------------
	def sample(self, batch_size: int):
		"""回傳 Batch"""
		idx = torch.randint(0, self.size, (batch_size,))

		if self.is_MA:          # -------- 多智能體 --------
			return (
				self.local_state[idx].to(self.device),        # s_local
				self.global_state[idx].to(self.device),       # s_global
				self.action[idx].to(self.device),             # a
				self.next_local_state[idx].to(self.device),   # s'_local
				self.next_global_state[idx].to(self.device),  # s'_global
				self.reward[idx].to(self.device),             # r
				self.not_done[idx].to(self.device)            # 1-done
			)
		else:                   # -------- 單智能體 --------
			return (
				self.state[idx].to(self.device),              # s
				self.action[idx].to(self.device),             # a
				self.next_state[idx].to(self.device),         # s'
				self.reward[idx].to(self.device),             # r
				self.not_done[idx].to(self.device)            # 1-done
			)

def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward
