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
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from core import utils


class TD3(object):
    """Classes implementing TD3 and DDPG off-policy learners

         Parameters:
               args (object): Parameter class


     """
    def __init__(self, model_constructor, actor_lr, critic_lr, gamma, tau, polciy_noise, policy_noise_clip,policy_ups_freq):

        self.gamma = gamma; self.tau = tau; self.policy_noise = polciy_noise; self.policy_noise_clip = policy_noise_clip; self.policy_ups_freq = policy_ups_freq

        #Initialize actors
        self.actor = model_constructor.make_model('Gaussian_FF')
        self.actor.stochastic = False
        #if init_w: self.actor.apply(utils.init_weights)
        self.actor_target = model_constructor.make_model('Gaussian_FF')
        self.actor_target.stochastic = False
        utils.hard_update(self.actor_target, self.actor)
        self.actor_optim = Adam(self.actor.parameters(), actor_lr, weight_decay=0.01)


        self.critic = model_constructor.make_model('Tri_Head_Q')
        #if init_w: self.critic.apply(utils.init_weights)
        self.critic_target = model_constructor.make_model('Tri_Head_Q')
        utils.hard_update(self.critic_target, self.critic)
        self.critic_optim = Adam(self.critic.parameters(), critic_lr, weight_decay=0.01)

        self.loss = nn.MSELoss()

        if torch.cuda.is_available():
            self.actor_target.cuda(); self.critic_target.cuda(); self.actor.cuda(); self.critic.cuda()
        self.num_critic_updates = 0

        #Statistics Tracker
        self.action_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.policy_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.critic_loss = {'mean':[]}
        self.q = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.val = {'min':[], 'max': [], 'mean':[], 'std':[]}

    def compute_stats(self, tensor, tracker):
        """Computes stats from intermediate tensors

             Parameters:
                   tensor (tensor): tensor
                   tracker (object): logger

             Returns:
                   None


         """
        tracker['min'].append(torch.min(tensor).item())
        tracker['max'].append(torch.max(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())

    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, num_epoch=1):
        """Runs a step of Bellman upodate and policy gradient using a batch of experiences

             Parameters:
                  state_batch (tensor): Current States
                  next_state_batch (tensor): Next States
                  action_batch (tensor): Actions
                  reward_batch (tensor): Rewards
                  done_batch (tensor): Done batch
                  num_epoch (int): Number of learning iteration to run with the same data

             Returns:
                   None

         """

        if isinstance(state_batch, list):
            state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch)

        for _ in range(num_epoch):
            ########### CRITIC UPDATE ####################

            #Compute next q-val, next_v and target
            with torch.no_grad():
                #Policy Noise
                policy_noise = np.random.normal(0, self.policy_noise, (action_batch.size()[0], action_batch.size()[1]))
                policy_noise = torch.clamp(torch.Tensor(policy_noise), -self.policy_noise_clip, self.policy_noise_clip,)
                if torch.cuda.is_available():
                    policy_noise = policy_noise.cuda()

                #Compute next action_bacth
                next_action_batch = self.actor_target.clean_action(next_state_batch) + policy_noise
                next_action_batch = torch.clamp(next_action_batch, -1,1)

                #Compute Q-val and value of next state masking by done
                q1, q2, _ = self.critic_target.forward(next_state_batch, next_action_batch)
                q1 = (1 - done_batch) * q1
                q2 = (1 - done_batch) * q2

                #Select which q to use as next-q (depends on algo)
                next_q = torch.min(q1, q2)

                #Compute target q and target val
                target_q = reward_batch + (self.gamma * next_q)


            self.critic_optim.zero_grad()
            current_q1, current_q2, current_val = self.critic.forward(state_batch, action_batch)
            self.compute_stats(current_q1, self.q)

            dt = self.loss(current_q1, target_q)

            dt = dt + self.loss(current_q2, target_q)
            self.critic_loss['mean'].append(dt.item())

            dt.backward()

            self.critic_optim.step()
            self.num_critic_updates += 1


            #Delayed Actor Update
            if self.num_critic_updates % self.policy_ups_freq == 0:

                actor_actions = self.actor.clean_action(state_batch)
                Q1, Q2, val = self.critic.forward(state_batch, actor_actions)

                # if self.args.use_advantage: policy_loss = -(Q1 - val)
                policy_loss = -Q1

                self.compute_stats(policy_loss,self.policy_loss)
                policy_loss = policy_loss.mean()


                self.actor_optim.zero_grad()



                policy_loss.backward(retain_graph=True)
                self.actor_optim.step()


            if self.num_critic_updates % self.policy_ups_freq == 0: utils.soft_update(self.actor_target, self.actor, self.tau)
            utils.soft_update(self.critic_target, self.critic, self.tau)










