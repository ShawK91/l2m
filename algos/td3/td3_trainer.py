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

import numpy as np, time
from core import utils
from core.runner import rollout_worker
from torch.multiprocessing import Process, Pipe, Manager
import torch
from core.buffer import Buffer
from algos.td3.td3 import TD3




class TD3_Trainer:
	"""Main CERL class containing all methods for CERL

		Parameters:
		args (object): Parameter class with all the parameters

	"""

	def __init__(self, args, model_constructor, env_constructor):
		self.args = args

		#MP TOOLS
		self.manager = Manager()

		#Algo
		self.algo = TD3(model_constructor,  actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma, tau=args.tau, polciy_noise=0.1, policy_noise_clip=0.2, policy_ups_freq=2)

		#Save best policy
		self.best_policy = model_constructor.make_model('Gaussian_FF')
		self.best_policy.stochastic = False

		#Init BUFFER
		self.replay_buffer = Buffer(args.buffer_size)
		self.data_bucket = self.replay_buffer.tuples

		#Initialize Rollout Bucket
		self.rollout_bucket = self.manager.list()
		self.rollout_bucket.append(model_constructor.make_model('Gaussian_FF'))
		for actor in self.rollout_bucket:
			actor.stochastic = False
			actor.eval()

		############## MULTIPROCESSING TOOLS ###################
		#Learner rollout workers
		self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.workers = [Process(target=rollout_worker, args=(id, 'pg', self.task_pipes[id][1], self.result_pipes[id][0], self.data_bucket, self.rollout_bucket, env_constructor)) for id in range(args.rollout_size)]
		for worker in self.workers: worker.start()
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = self.manager.list()
		self.test_bucket.append(model_constructor.make_model('Gaussian_FF'))
		for actor in self.test_bucket:
			actor.stochastic=False
			actor.eval()

		#5 Test workers
		self.test_task_pipes = [Pipe() for _ in range(env_constructor.dummy_env.test_size)]
		self.test_result_pipes = [Pipe() for _ in range(env_constructor.dummy_env.test_size)]
		self.test_workers = [Process(target=rollout_worker, args=(id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], None, self.test_bucket, env_constructor)) for id in range(env_constructor.dummy_env.test_size)]
		for worker in self.test_workers: worker.start()
		self.test_flag = False

		#Trackers
		self.best_score = 0.0; self.gen_frames = 0; self.total_frames = 0; self.test_score = None; self.test_std = None; self.test_trace = []
		self.ep_len = 0
		self.r1_reward = 0
		self.num_footsteps = 0

	def forward_epoch(self, epoch, tracker):
		"""Main training loop to do rollouts, neureoevolution, and policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""
		################ START ROLLOUTS ##############
		#Sync all learners actor to cpu (rollout) actor
		self.algo.actor.cpu()
		utils.hard_update(self.rollout_bucket[0], self.algo.actor)
		utils.hard_update(self.test_bucket[0], self.algo.actor)
		self.algo.actor.cuda()

		# Start Learner rollouts
		for rollout_id in range(self.args.rollout_size):
			if self.roll_flag[rollout_id]:
				self.task_pipes[rollout_id][0].send(0)
				self.roll_flag[rollout_id] = False

		#Start Test rollouts
		if epoch % 1 == 0:
			self.test_flag = True
			for pipe in self.test_task_pipes: pipe[0].send(0)


		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		if self.replay_buffer.__len__() > self.args.learning_start: ###BURN IN PERIOD
			self.replay_buffer.tensorify()  # Tensorify the buffer for fast sampling
			for _ in range(self.gen_frames):
				s, ns, a, r, done = self.replay_buffer.sample(self.args.batch_size)
				if torch.cuda.is_available():
					s = s.cuda()
					ns = ns.cuda()
					a = a.cuda()
					r = r.cuda()
					done = done.cuda()
					r = r * self.args.reward_scaling
				self.algo.update_parameters(s, ns, a, r, done)

			self.gen_frames = 0


		######### HARD -JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		if self.args.rollout_size > 0:
			for i in range(self.args.rollout_size):
				entry = self.result_pipes[i][1].recv()
				learner_id = entry[0]; fitness = entry[1]; num_frames = entry[2]

				self.gen_frames += num_frames; self.total_frames += num_frames
				if fitness > self.best_score: self.best_score = fitness

				self.roll_flag[i] = True

			#Referesh buffer (housekeeping tasks - pruning to keep under capacity)
			self.replay_buffer.referesh()
		######################## END OF PARALLEL ROLLOUTS ################


		###### TEST SCORE ######
		if self.test_flag:
			self.test_flag = False
			test_scores = []; eplens = []; r1_reward = []; num_footsteps = []
			for pipe in self.test_result_pipes: #Collect all results
				entry = pipe[1].recv()
				test_scores.append(entry[1])
				eplens.append(entry[3])
				r1_reward.append(entry[4])
				num_footsteps.append(entry[5])

			test_scores = np.array(test_scores)
			test_mean = np.mean(test_scores); test_std = (np.std(test_scores))
			self.test_trace.append(test_mean)
			self.num_footsteps = np.mean(np.array(num_footsteps))
			self.ep_len = np.mean(np.array(eplens))
			self.r1_reward = np.mean(np.array(r1_reward))
			tracker.update([test_mean, self.r1_reward], self.total_frames)

		else:
			test_mean, test_std = None, None

		return test_mean, test_std


	def train(self, frame_limit):
		# Define Tracker class to track scores
		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag, 'r1_'+self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()

		for gen in range(1, 1000000000):  # Infinite generations

			# Train one iteration
			test_mean, test_std = self.forward_epoch(gen, test_tracker)

			print('Gen/Frames', gen,'/',self.total_frames, 'max_ever:','%.2f'%self.best_score, ' Avg:','%.2f'%test_tracker.all_tracker[0][1],
		      ' Frames/sec:','%.2f'%(self.total_frames/(time.time()-time_start)),
			   ' Test Trace', ['%.2f'% i for i in self.test_trace[-5:]],
				  				  'Ep_len', '%.2f'%self.ep_len, '#Footsteps', '%.2f'%self.num_footsteps, 'R1_Reward', '%.2f'%self.r1_reward,
				  'savetag', self.args.savetag)

			# if gen % 5 == 0:
			# 	print()
			#
			# 	print('Entropy', self.algo.entropy, 'Next_Entropy', self.algo.next_entropy, 'Poilcy_Q', self.algo.policy_q, 'Critic_Loss', self.algo.critic_loss)
			#
			# 	print()

			if self.total_frames > frame_limit:
				break




