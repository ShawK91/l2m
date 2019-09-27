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

import numpy as np, os, time, random, torch, sys
from algos.cerl.neuroevolution import SSNE
from core import utils
from algos.cerl.ucb import ucb
from core.runner import rollout_worker
from algos.cerl.portfolio import initialize_portfolio
from torch.multiprocessing import Process, Pipe, Manager
import threading
from core.buffer import Buffer
from algos.cerl.genealogy import Genealogy



class CERL_Trainer:
	"""Main CERL class containing all methods for CERL

		Parameters:
		args (object): Parameter class with all the parameters

	"""

	def __init__(self, args, model_constructor, env_constructor):
		self.args = args
		self.policy_string = self.compute_policy_type()

		#Evolution
		self.evolver = SSNE(self.args)

		#MP TOOLS
		self.manager = Manager()

		#Genealogy tool
		self.genealogy = Genealogy()

		#Initialize population
		self.population = self.manager.list()
		for _ in range(args.pop_size):
			self.population.append(model_constructor.make_model(self.policy_string))

		#SEED
		#self.population[0].load_state_dict(torch.load('Results/Auxiliary/_bestcerl_td3_s2019_roll10_pop10_portfolio10'))


		#Save best policy
		self.best_policy = model_constructor.make_model(self.policy_string)

		#Turn off gradients and put in eval mod
		for actor in self.population:
			actor = actor.cpu()
			actor.eval()

		#Init BUFFER
		self.replay_buffer = Buffer(args.buffer_size)
		self.data_bucket = self.replay_buffer.tuples

		#Intialize portfolio of learners
		self.portfolio = []
		self.portfolio = initialize_portfolio(self.portfolio, self.args, self.genealogy, args.portfolio_id, model_constructor)

		#Initialize Rollout Bucket
		self.rollout_bucket = self.manager.list()
		for _ in range(len(self.portfolio)):
			self.rollout_bucket.append(model_constructor.make_model(self.policy_string))

		############## MULTIPROCESSING TOOLS ###################

		#Evolutionary population Rollout workers
		self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
		self.evo_workers = [Process(target=rollout_worker, args=(id, 'evo', self.evo_task_pipes[id][1], self.evo_result_pipes[id][0], self.data_bucket, self.population, env_constructor)) for id in range(args.pop_size)]
		for worker in self.evo_workers: worker.start()
		self.evo_flag = [True for _ in range(args.pop_size)]

		#Learner rollout workers
		self.task_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.result_pipes = [Pipe() for _ in range(args.rollout_size)]
		self.workers = [Process(target=rollout_worker, args=(id, 'pg', self.task_pipes[id][1], self.result_pipes[id][0], self.data_bucket, self.rollout_bucket, env_constructor)) for id in range(args.rollout_size)]
		for worker in self.workers: worker.start()
		self.roll_flag = [True for _ in range(args.rollout_size)]

		#Test bucket
		self.test_bucket = self.manager.list()
		self.test_bucket.append(model_constructor.make_model(self.policy_string))

		#5 Test workers
		self.test_task_pipes = [Pipe() for _ in range(env_constructor.dummy_env.test_size)]
		self.test_result_pipes = [Pipe() for _ in range(env_constructor.dummy_env.test_size)]
		self.test_workers = [Process(target=rollout_worker, args=(id, 'test', self.test_task_pipes[id][1], self.test_result_pipes[id][0], None, self.test_bucket, env_constructor)) for id in range(env_constructor.dummy_env.test_size)]
		for worker in self.test_workers: worker.start()
		self.test_flag = False

		#Meta-learning controller (Resource Distribution)
		self.allocation = [] #Allocation controls the resource allocation across learners
		for i in range(args.rollout_size): self.allocation.append(i % len(self.portfolio)) #Start uniformly (equal resources)

		#Trackers
		self.best_score = 0.0; self.gen_frames = 0; self.total_frames = 0; self.test_score = None; self.test_std = None

		self.ep_len = 0
		self.r1_reward = 0
		self.num_footsteps = 0
		self.test_trace = []

	def checkpoint(self):
		utils.pickle_obj(self.args.aux_folder+self.args.algo+'_checkpoint_frames'+str(self.total_frames), self.portfolio)


	def load_checkpoint(self, filename):
		self.portfolio = utils.unpickle_obj(filename)


	def forward_generation(self, gen, tracker):
		"""Main training loop to do rollouts, neureoevolution, and policy gradients

			Parameters:
				gen (int): Current epoch of training

			Returns:
				None
		"""
		################ START ROLLOUTS ##############

		#Start Evolution rollouts
		if self.args.pop_size > 1:
			for id, actor in enumerate(self.population):
				if self.evo_flag[id]:
					self.evo_task_pipes[id][0].send(id)
					self.evo_flag[id] = False

		#Sync all learners actor to cpu (rollout) actor
		for i, learner in enumerate(self.portfolio):
			learner.algo.actor.cpu()
			utils.hard_update(self.rollout_bucket[i], learner.algo.actor)
			learner.algo.actor.cuda()

		# Start Learner rollouts
		for rollout_id, learner_id in enumerate(self.allocation):
			if self.roll_flag[rollout_id]:
				self.task_pipes[rollout_id][0].send(learner_id)
				self.roll_flag[rollout_id] = False

		#Start Test rollouts
		if gen % 1 == 0:
			self.test_flag = True
			for pipe in self.test_task_pipes: pipe[0].send(0)


		############# UPDATE PARAMS USING GRADIENT DESCENT ##########
		if self.replay_buffer.__len__() > self.args.learning_start: ###BURN IN PERIOD
			self.replay_buffer.tensorify()  # Tensorify the buffer for fast sampling

			#Spin up threads for each learner
			threads = [threading.Thread(target=learner.update_parameters, args=(self.replay_buffer, self.args.batch_size, int(self.gen_frames * self.args.gradperstep))) for learner in
			           self.portfolio]

			# Start threads
			for thread in threads: thread.start()

			#Join threads
			for thread in threads: thread.join()
			self.gen_frames = 0


		########## SOFT -JOIN ROLLOUTS FOR EVO POPULATION ############
		if self.args.pop_size > 1:
			all_fitness = []; all_net_ids = []; all_eplens = []
			while True:
				for i in range(self.args.pop_size):
					if self.evo_result_pipes[i][1].poll():
						entry = self.evo_result_pipes[i][1].recv()
						all_fitness.append(entry[1]); all_net_ids.append(entry[0]); all_eplens.append(entry[2]); self.gen_frames+= entry[2]; self.total_frames += entry[2]
						self.evo_flag[i] = True

				# Soft-join (50%)
				if len(all_fitness) / self.args.pop_size >= self.args.asynch_frac: break

		########## HARD -JOIN ROLLOUTS FOR LEARNER ROLLOUTS ############
		if self.args.rollout_size > 0:
			for i in range(self.args.rollout_size):
				entry = self.result_pipes[i][1].recv()
				learner_id = entry[0]; fitness = entry[1]; num_frames = entry[2]
				self.portfolio[learner_id].update_stats(fitness, num_frames)

				self.gen_frames += num_frames; self.total_frames += num_frames
				if fitness > self.best_score: self.best_score = fitness

				self.roll_flag[i] = True

			#Referesh buffer (housekeeping tasks - pruning to keep under capacity)
			self.replay_buffer.referesh()
		######################### END OF PARALLEL ROLLOUTS ################

		############ PROCESS MAX FITNESS #############
		if self.args.pop_size > 1:
			champ_index = all_net_ids[all_fitness.index(max(all_fitness))]
			utils.hard_update(self.test_bucket[0], self.population[champ_index])
			if max(all_fitness) > self.best_score:
				self.best_score = max(all_fitness)
				utils.hard_update(self.best_policy, self.population[champ_index])
				torch.save(self.population[champ_index].state_dict(), self.args.aux_folder + '_best'+self.args.savetag)
				print("Best policy saved with score", '%.2f'%max(all_fitness))

		else: #Run PG in isolation
			utils.hard_update(self.test_bucket[0], self.rollout_bucket[0])

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


		#NeuroEvolution's probabilistic selection and recombination step
		if self.args.pop_size > 1:
			if gen % 5 == 0:
				self.evolver.epoch(gen, self.genealogy, self.population, all_net_ids, all_fitness, self.rollout_bucket)
			else:
				self.evolver.epoch(gen, self.genealogy, self.population, all_net_ids, all_fitness, [])

		#META LEARNING - RESET ALLOCATION USING UCB
		if self.args.rollout_size > 0: self.allocation = ucb(len(self.allocation), self.portfolio, self.args.ucb_coefficient)


		#Metrics
		if self.args.pop_size > 1:
			champ_len = all_eplens[all_fitness.index(max(all_fitness))]
			#champ_wwid = int(self.pop[champ_index].wwid.item())
			max_fit = max(all_fitness)
		else:
			champ_len = num_frames
			all_fitness = [fitness]; max_fit = fitness; all_eplens = [num_frames]


		return max_fit, champ_len, all_eplens, test_mean, test_std


	def train(self, frame_limit):
		# Define Tracker class to track scores
		test_tracker = utils.Tracker(self.args.savefolder, ['score_' + self.args.savetag, 'r1_'+self.args.savetag], '.csv')  # Tracker class to log progress
		time_start = time.time()

		for gen in range(1, 1000000000):  # Infinite generations

			# Train one iteration
			max_fitness, champ_len, all_eplens, test_mean, test_std = self.forward_generation(gen, test_tracker)

			print('Gen/Frames', gen,'/',self.total_frames, ' Pop_max/max_ever:','%.2f'%max_fitness, '/','%.2f'%self.best_score, ' Avg:','%.2f'%test_tracker.all_tracker[0][1],
		      ' Frames/sec:','%.2f'%(self.total_frames/(time.time()-time_start)),
			  ' Champ_len', '%.2f'%champ_len, ' Test_score u/std', utils.pprint(test_mean), utils.pprint(test_std),
			  'Ep_len', '%.2f'%self.ep_len, '#Footsteps', '%.2f'%self.num_footsteps, 'R1_Reward', '%.2f'%self.r1_reward,
			  'savetag', self.args.savetag)

			if gen % 5 == 0:
				print('Learner Fitness', [utils.pprint(learner.value) for learner in self.portfolio],
					  'Sum_stats_resource_allocation', [learner.visit_count for learner in self.portfolio])
				try:
					print('Entropy', ['%.2f'%algo.algo.entropy['mean'] for algo in self.portfolio],
						  'Next_Entropy', ['%.2f'%algo.algo.next_entropy['mean'] for algo in self.portfolio],
						  'Poilcy_Q', ['%.2f'%algo.algo.policy_q['mean'] for algo in self.portfolio],
						  'Critic_Loss', ['%.2f'%algo.algo.critic_loss['mean'] for algo in self.portfolio])
					print()
				except: None

			if self.total_frames > frame_limit:
				break

		###Kill all processes
		try:
			for p in self.task_pipes: p[0].send('TERMINATE')
			for p in self.test_task_pipes: p[0].send('TERMINATE')
			for p in self.evo_task_pipes: p[0].send('TERMINATE')
		except:
			None


	def compute_policy_type(self):
		if self.args.algo == 'ddqn':
			return 'DDQN'

		elif self.args.algo == 'sac':
			return 'Gaussian_FF'

		elif self.args.algo == 'td3':
			return 'Deterministic_FF'
