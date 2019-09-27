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
import gym

def is_discrete(env):
    try:
        k = env.action_space.n
        return True
    except:
        return False


class GymWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, frameskip=None):
        """
        A base template for all environment wrappers.
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.frameskip=frameskip
        self.is_discrete = is_discrete(self.env)



        #State and Action Parameters
        self.state_dim = self.env.observation_space.shape[0]
        if self.is_discrete:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.action_low = float(self.env.action_space.low[0])
            self.action_high = float(self.env.action_space.high[0])
        self.test_size = 10

    def reset(self):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """
        state = self.env.reset()
        return np.expand_dims(state, axis=0)



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
        if  self.is_discrete:
           action = action[0]
        else:
            action = self.action_low + (action + 1.0) / 2.0 * (self.action_high - self.action_low)


        reward = 0
        for _ in range(self.frameskip):
            next_state, rew, done, info = self.env.step(action)
            reward += rew
            if done: break


        next_state = np.expand_dims(next_state, axis=0)

        return next_state, reward, done, info


    def render(self):
        self.env.render()

