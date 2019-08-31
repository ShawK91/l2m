# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from abstract.agent import Agent
import torch

class Learner(Agent):
	"""Learner object encapsulating a local learner

		Parameters:
		algo_name (str): Algorithm Identifier
		state_dim (int): State size
		action_dim (int): Action size
		actor_lr (float): Actor learning rate
		critic_lr (float): Critic learning rate
		gamma (float): DIscount rate
		tau (float): Target network sync generate
		init_w (bool): Use kaimling normal to initialize?
		**td3args (**kwargs): arguments for TD3 algo


	"""

	def __init__(self, id, algo_name, state_dim, goal_dim, action_dim, actor_lr, critic_lr, gamma, tau):
		super().__init__(id, algo_name)

		if algo_name == 'cerl_td3':
			from algos.td3.td3 import TD3
			self.algo = TD3(wwid=id, state_dim=state_dim, goal_dim=goal_dim, action_dim=action_dim, actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau, polciy_noise=0.1, policy_noise_clip=0.2, policy_ups_freq=2)



