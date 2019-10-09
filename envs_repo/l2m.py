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
from envs_repo import rs


class L2M:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, visualize=False, integrator_accuracy=5e-5, frameskip=4, T=2500, action_clamp=False, difficulty=2):
        """
        A base template for all environment wrappers.
        """
        from osim.env import L2M2019Env
        self.env = L2M2019Env(visualize=visualize, integrator_accuracy=integrator_accuracy, seed=0, report=None, difficulty=difficulty)
        self.frameskip=frameskip
        self.T=T; self.istep = 0
        self.action_clamp = action_clamp

        #Self Params
        self.state_dim = 169
        self.action_dim = 22
        self.test_size = 1

        #Trackers
        self.r2_reward = 0
        self.num_footsteps = 0



    def reset(self):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """

        self.istep = 0
        self.r1_reward = 0
        self.num_footsteps = 0


        state_dict = self.env.reset()
        obs = flatten(state_dict)
        goal = state_dict['v_tgt_field']
        goal = goal[:, 0::2, 0::2].flatten()
        state = np.concatenate((obs, goal))
        state = np.expand_dims(state, 0)

        if check_nan_inf(state):
            print(state)
            raise Exception ('Nan or Inf encountered')

        return state



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

        if self.action_clamp:
            bounded_action = (action + 1.0) / 2.0  # Tanh --> Sigmoid
        else:
            bounded_action = action
        reward = 0.0; done=False
        for _ in range(self.frameskip):
            self.istep += 1

            next_state_dict, rew, done, info = self.env.step(bounded_action)

            self.r1_reward += rew
            if self.env.footstep['new']: self.num_footsteps+= 1

            rew = rs.get_reward_footsteps_r2(self.env)
            reward += rew

            if done: break


        next_obs = flatten(next_state_dict)
        next_goal = next_state_dict['v_tgt_field']
        next_goal = next_goal[:, 0::2, 0::2].flatten()
        next_state = np.concatenate((next_obs, next_goal))
        next_state = np.expand_dims(next_state, 0)

        #Update trackers
        return next_state, reward, done, info

    def render(self):
        self.env.render()


class L2MRemote:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, client, token, frameskip):
        """
        A base template for all environment wrappers.
        """
        self.client = client
        self.token = token
        self.frameskip = frameskip

        state_dict = self.client.env_create(self.token, env_id='L2M2019Env')
        self.first_state = process_dict(state_dict)


    def reset(self):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """

        state_dict = self.client.env_reset()
        if not state_dict: return None, True
        print()
        print('One Run Concluded')
        print()
        state = process_dict(state_dict)
        return state, False





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

        reward = 0.0
        #action = (action + 1.0) / 2.0  # Tanh --> Sigmoid
        for _ in range(self.frameskip):
            next_state_dict, rew, done, info = self.client.env_step(action)
            reward += rew
            if done: break


        next_state = process_dict(next_state_dict)


        return next_state, reward, done, info




def process_dict(dict):
    obs = flatten(dict)
    goal = dict['v_tgt_field']
    #print(goal)
    if isinstance(goal, list): goal = np.array(goal)
    goal = goal[:, 0::2, 0::2].flatten()

    state = np.concatenate((obs, goal))
    state = np.expand_dims(state, 0)

    if check_nan_inf(state):
        print(state)
        raise Exception('Nan or Inf encountered')

    return state



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
            if key == 'v_tgt_field':
               continue
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    elif isinstance(d, np.ndarray):
        res = list(d.flatten())
    else:
        res = [d]

    return res



def check_nan_inf(array):
    return np.isnan(array).any() or np.isinf(array).any()
