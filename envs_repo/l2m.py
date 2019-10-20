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
from envs_repo import rs, obs_wrapper


class L2M:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, visualize=False, integrator_accuracy=5e-5, frameskip=4, T=2500, action_clamp=False, difficulty=2, project=True):
        """
        A base template for all environment wrappers.
        """
        from osim.env import L2M2019Env
        self.env = L2M2019Env(visualize=visualize, integrator_accuracy=integrator_accuracy, seed=0, report=None, difficulty=difficulty)
        self.frameskip=frameskip
        self.T=T; self.istep = 0
        self.action_clamp = action_clamp
        self.project = project

        #Self Params
        self.state_dim = 169 if self.project else 228+72
        self.action_dim = 22
        self.test_size = 5

        #Trackers
        self.shaped_reward = {'num_footsteps':[],
                              'crouch_bonus':[],
                              'knee_bend':[],
                              'toes_low': [],
                              'x_penalty':[],
                              'z_penalty':[]}
        self.original_reward = 0.0
        self.fell_down = False

        #Reward Shaping components
        self.ltoes = {'x':[], 'y':[], 'z':[]}; self.rtoes= {'x':[], 'y':[], 'z':[]}
        self.ltibia = {'x':[], 'y':[], 'z':[]}; self.rtibia = {'x':[], 'y':[], 'z':[]}
        self.pelvis = {'x':[], 'y':[], 'z':[]}

        self.ltibia_angle = []; self.rtibia_angle = []
        self.lfemur_angle = []; self.rfemur_angle = []


    def reset(self):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """

        self.istep = 0
        self.shaped_reward = {'num_footsteps':[],
                              'crouch_bonus':[],
                              'knee_bend':[],
                              'toes_low': [],
                              'x_penalty':[],
                              'z_penalty':[]}
        self.original_reward = 0.0
        self.fell_down = False

        #Reward Shaping components
        self.ltoes = {'x':[], 'y':[], 'z':[]}; self.rtoes= {'x':[], 'y':[], 'z':[]}
        self.ltibia = {'x':[], 'y':[], 'z':[]}; self.rtibia = {'x':[], 'y':[], 'z':[]}
        self.pelvis = {'x':[], 'y':[], 'z':[]}

        self.ltibia_angle = []; self.rtibia_angle = []
        self.lfemur_angle = []; self.rfemur_angle = []


        state_dict = self.env.reset(project=self.project)
        self.update_vars()
        if self.project: obs = flatten(state_dict)
        else: obs = obs_wrapper.shape_observation(state_dict)

        goal = state_dict['v_tgt_field']
        goal = goal[:, 0::2, 0::2].flatten()
        state = np.concatenate((obs, goal))
        state = np.expand_dims(state, 0)



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

            next_state_dict, rew, done, info = self.env.step(bounded_action, project=self.project)

            if self.istep > self.T: done = True



            self.update_vars()
            self.update_shaped_reward()

            #Fall Down?
            if self.pelvis['y'][-1] < 0.6: self.fell_down = True

            #UnShaped Reward
            self.original_reward += rew


            reward += 2.0 + \
                      1.0 * self.shaped_reward['x_penalty'][-1] + \
                      0.2 * self.shaped_reward['z_penalty'][-1] + \
                      0.2 * self.shaped_reward['knee_bend'][-1] + \
                      0.5 * self.shaped_reward['crouch_bonus'][-1] + \
                      0.3 * self.shaped_reward['toes_low'][-1] +\
                      -10.0 * self.fell_down

            if done: break



        if self.project: next_obs = flatten(next_state_dict)
        else: next_obs = obs_wrapper.shape_observation(next_state_dict)

        next_goal = next_state_dict['v_tgt_field']
        next_goal = next_goal[:, 0::2, 0::2].flatten()

        next_state = np.concatenate((next_obs, next_goal))
        next_state = np.expand_dims(next_state, 0)

        #Update trackers
        return next_state, reward, done, info

    def render(self):
        self.env.render()


    def update_vars(self):
        state = self.env.get_state_desc()
        #TIBIA
        self.ltibia['x'].append(state["body_pos"]["tibia_l"][0])
        self.ltibia['y'].append(state["body_pos"]["tibia_l"][1])
        self.ltibia['z'].append(state["body_pos"]["tibia_l"][2])

        self.rtibia['x'].append(state["body_pos"]["tibia_r"][0])
        self.rtibia['y'].append(state["body_pos"]["tibia_r"][1])
        self.rtibia['z'].append(state["body_pos"]["tibia_r"][2])

        #TOES
        self.ltoes['x'].append(state["body_pos"]["toes_l"][0])
        self.ltoes['y'].append(state["body_pos"]["toes_l"][1])
        self.ltoes['z'].append(state["body_pos"]["toes_l"][2])

        self.rtoes['x'].append(state["body_pos"]["toes_r"][0])
        self.rtoes['y'].append(state["body_pos"]["toes_r"][1])
        self.rtoes['z'].append(state["body_pos"]["toes_r"][2])

        #PELVIS
        self.pelvis['x'].append(state["body_pos"]["pelvis"][0])
        self.pelvis['y'].append(state["body_pos"]["pelvis"][1])
        self.pelvis['z'].append(state["body_pos"]["pelvis"][2])

        #ANGLES
        self.ltibia_angle.append(state["body_pos_rot"]["tibia_l"][2])
        self.rtibia_angle.append(state["body_pos_rot"]["tibia_r"][2])
        self.lfemur_angle.append(state["body_pos_rot"]["femur_l"][2])
        self.rfemur_angle.append(state["body_pos_rot"]["femur_r"][2])

    def update_shaped_reward(self):

        if self.env.footstep['new']: self.shaped_reward['num_footsteps'].append(1)

        self.shaped_reward['crouch_bonus'].append(rs.crouch(self.pelvis['y'][-1]))


        self.shaped_reward['knee_bend'].append(rs.knee_bend(self.ltibia_angle[-1], self.lfemur_angle[-1], self.rtibia_angle[-1], self.lfemur_angle[-1]))

        self.shaped_reward['toes_low'].append(rs.toes_low(self.ltoes['y'][-1], self.rtoes['y'][-1]))



        x_pen, z_pen = rs.vel_follower(self.env)
        self.shaped_reward['x_penalty'].append(x_pen)
        self.shaped_reward['z_penalty'].append(z_pen)



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
