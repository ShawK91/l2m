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

import torch

class Learner():
	"""Abstract Class specifying an object
	"""

	def __init__(self, model_constructor, args, algo, gamma, **kwargs):

		if algo == 'td3':
			from algos.td3.td3 import TD3
			self.algo = TD3(model_constructor, actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=gamma, tau=args.tau, polciy_noise=0.1, policy_noise_clip=0.2, policy_ups_freq=2)

		elif algo == 'ddqn':
			from algos.ddqn.ddqn import DDQN
			self.algo = DDQN(args, model_constructor, gamma)

		elif algo == 'sac':
			from algos.sac.sac import SAC
			self.algo = SAC(args, model_constructor, gamma, **kwargs)

		else:
			Exception('Unknown algo in learner.py')

		#Agent Stats
		self.fitnesses = []
		self.ep_lens = []
		self.value = None
		self.visit_count = 0


	def update_parameters(self, replay_buffer, batch_size, iterations):
		for _ in range(iterations):
			s, ns, a, r, done = replay_buffer.sample(batch_size)

			if torch.cuda.is_available():
				s =s.cuda()
				ns = ns.cuda()
				a=a.cuda()
				r=r.cuda()
				done=done.cuda()
			self.algo.update_parameters(s, ns, a, r, done)


	def update_stats(self, fitness, ep_len, gamma=0.2):
		self.visit_count += 1
		self.fitnesses.append(fitness)
		self.ep_lens.append(ep_len)

		if self.value == None: self.value = fitness
		else: self.value = gamma * fitness + (1-gamma) * self.value
