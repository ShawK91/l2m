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

class Agent:
	"""Abstract Class specifying an object
	"""

	def __init__(self, id, algo_name):
		self.algo_name = algo_name
		self.id = id

		#Agent Stats
		self.fitnesses = []
		self.ep_lens = []
		self.value = None
		self.visit_count = 0


	def update_parameters(self, replay_buffer, batch_size, iterations):
		for _ in range(iterations):
			s, ns, g, ng, a, r, done = replay_buffer.sample(batch_size)

			if torch.cuda.is_available():
				s =s.cuda()
				ns = ns.cuda()
				g=g.cuda()
				ng=ng.cuda()
				a=a.cuda()
				r=r.cuda()
				done=done.cuda()
			self.algo.update_parameters(s, ns, g, ng, a, r, done, 1)


	def update_stats(self, fitness, ep_len, gamma=0.2):
		self.visit_count += 1
		self.fitnesses.append(fitness)
		self.ep_lens.append(ep_len)

		if self.value == None: self.value = fitness
		else: self.value = gamma * fitness + (1-gamma) * self.value
