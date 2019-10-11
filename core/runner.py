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
from core import utils as utils
import numpy as np


# Rollout evaluate an agent in a complete game
def rollout_worker(id, type, task_pipe, result_pipe, data_bucket, model_bucket, env_constructor):
	"""Rollout Worker runs a simulation in the environment to generate experiences and fitness values

        Parameters:
            task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
            result_pipe (pipe): Sender end of the pipe used to report back results
            is_noise (bool): Use noise?
            data_bucket (list of shared object): A list of shared object reference to s,ns,a,r,done (replay buffer) managed by a manager that is used to store experience tuples
            model_bucket (shared list object): A shared list object managed by a manager used to store all the models (actors)
			env_constructor (str): Environment Constructor


        Returns:
            None
    """

	env = env_constructor.make_env()
	np.random.seed(id) ###make sure the random seeds across learners are different

	###LOOP###
	while True:
		identifier = task_pipe.recv()  # Wait until a signal is received  to start rollout
		if identifier == 'TERMINATE': exit(0) #Kill yourself

		# Get the requisite network
		net = model_bucket[identifier]

		fitness = 0.0
		total_frame = 0
		state = env.reset()
		rollout_trajectory = []
		state = utils.to_tensor(state)
		while True:  # unless done

			if type == 'pg': action = net.noisy_action(state)
			else: action = net.clean_action(state)


			action = utils.to_numpy(action)

			next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment

			next_state = utils.to_tensor(next_state)
			fitness += reward

			# If storing transitions
			if data_bucket != None: #Skip for test set
				rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(next_state),
				                        np.float32(action), np.reshape(np.float32(np.array([reward])), (1, 1)),
				                           np.reshape(np.float32(np.array([float(done)])), (1, 1))])
			state = next_state
			total_frame += 1

			# DONE FLAG IS Received
			if done:
				# Push experiences to main
				for entry in rollout_trajectory:
					data_bucket.append(entry)
				break

		# Send back id, fitness, total length and shaped fitness using the result pipe
		result_pipe.send([identifier, fitness, total_frame, env.istep, env.original_reward, env.shaped_reward['num_footsteps']])
