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
import numpy as np



class L2MWrapper:
	"""Wrapper around the Environment to expose a cleaner interface for RL

		Parameters:
			env_name (str): Env name


	"""
	def __init__(self, visualize=False, integrator_accuracy=5e-5, seed=0, report=None, frameskip=4, T=1000):
		"""
		A base template for all environment wrappers.
		"""
		from osim.env import L2M2019Env
		self.env = L2M2019Env(visualize=visualize, integrator_accuracy=integrator_accuracy, seed=0, report=None)
		self.frameskip=frameskip
		self.T=T; self.istep = 0


	def reset(self):
		"""Method overloads reset
			Parameters:
				None

			Returns:
				next_obs (list): Next state
		"""
		return np.expand_dims(np.array(flatten(self.env.reset())), 0)



	def step(self, action): #Expects a numpy action
		"""Take an action to forward the simulation

			Parameters:
				action (ndarray): action to take in the env

			Returns:
				next_obs (list): Next state
				reward (float): Reward for this step
				done (bool): Simulation done?
				info (None): Template from OpenAi gym (doesnt have anything)
		"""

		reward = 0.0; done=False
		for _ in range(self.frameskip):
			if done: continue
			self.istep += 1
			next_state, rew, done, info = self.env.step(action)
			reward += rew

		next_state = np.expand_dims(np.array(flatten(next_state)), 0)
		return next_state, reward, done, info

	def render(self):
		self.env.render()


def flatten(d):
    """Recursive method to flatten a dict -->list
        Parameters:
            d (dict): dict
        Returns:
            l (list)
    """
    res = []  # Result list
    if isinstance(d, dict):
        for key, val in sorted(d.items()):
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    elif isinstance(d, np.ndarray):
        res = list(d.flatten())
    else:
        res = [d]
    return res
