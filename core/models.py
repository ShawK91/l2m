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
import torch.nn.functional as F
from torch.distributions import Normal

#SAC SPECIFIC
LOG_SIG_MAX = 5
LOG_SIG_MIN = -10
epsilon = 1e-6


class Actor(nn.Module):
    """Actor model
        Parameters:
              args (object): Parameter class
    """

    def __init__(self, num_inputs, num_goals, num_actions, policy_type):
        super(Actor, self).__init__()

        self.policy_type = policy_type

        g1 = 50; h1=100; h2 =50

        #Construct map processig Layer
        self.goal1 = nn.Linear(num_goals, g1)
        self.goal2 = nn.Linear(g1, g1)

        #Shared FF
        self.linear1 = nn.Linear(num_inputs+g1, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.mean_linear = nn.Linear(h2, num_actions)

        if self.policy_type == 'GaussianPolicy':
            self.log_std_linear = nn.Linear(h2, num_actions)

        elif self.policy_type == 'DeterministicPolicy':
            self.noise = torch.Tensor(num_actions)

        self.apply(weights_init)


    def forward_goal_map(self, goal):
        #Goal Processing
        goal_out = torch.relu(self.goal1(goal))
        goal_out = torch.relu(self.goal2(goal_out))
        return goal_out


    def clean_action(self, state, goal, return_only_action=True):
        """Method to forward propagate through the actor's graph
            Parameters:
                  input (tensor): states
            Returns:
                  action (tensor): actions
        """
        goal_out =self.forward_goal_map(goal)
        state = torch.cat([state, goal_out], dim=1)

        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        mean = self.mean_linear(x)

        if return_only_action or self.policy_type == 'DeterministicPolicy': return torch.sigmoid(mean)

        elif self.policy_type == 'GaussianPolicy':
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            return mean, log_std


    def noisy_action(self, state, goal, return_only_action=True):

        if self.policy_type == 'GaussianPolicy':
            mean, log_std = self.clean_action(state, goal, return_only_action=False)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            action = torch.sigmoid(x_t)

            if return_only_action: return action

            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(1 - action.pow(2) + epsilon)
            log_prob = log_prob.sum(-1, keepdim=True)

            #log_prob.clamp(-10, 0)

            return action, log_prob, x_t, mean, log_std

        elif self.policy_type == 'DeterministicPolicy':
            mean = self.clean_action(state, goal)
            action = mean + self.noise.normal_(0., std=0.01)

            if return_only_action: return action
            else: return action, torch.tensor(0.), torch.tensor(0.), mean, torch.tensor(0.)



    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean


class Critic(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, goal_dim, action_dim):
        super(Critic, self).__init__()
        l1 = 100; l2 = 100; g1 = 50

        #Construct map processig Layer
        self.goal1 = nn.Linear(goal_dim, g1)
        self.goal2 = nn.Linear(g1, g1)

        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.q1f1 = nn.Linear(state_dim + action_dim + g1, l1)
        self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q1f2 = nn.Linear(l1, l2)
        self.q1ln2 = nn.LayerNorm(l2)

        #Out
        self.q1out = nn.Linear(l2, 1)


        ######################## Q2 Head ##################
        # Construct Hidden Layer 1 with state
        self.q2f1 = nn.Linear(state_dim + action_dim + g1, l1)
        self.q2ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q2f2 = nn.Linear(l1, l2)
        self.q2ln2 = nn.LayerNorm(l2)

        #Out
        self.q2out = nn.Linear(l2, 1)

        #self.half()


        # ######################## Value Head ##################  [NOT USED IN CERL]
        # # Construct Hidden Layer 1 with
        # self.vf1 = nn.Linear(state_dim, l1)
        # self.vln1 = nn.LayerNorm(l1)
        #
        # # Hidden Layer 2
        # self.vf2 = nn.Linear(l1, l2)
        # self.vln2 = nn.LayerNorm(l2)
        #
        # # Out
        # self.vout = nn.Linear(l2, 1)





    def forward(self, obs, goal, action):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """
        #Goal Processing
        goal_out = torch.tanh(self.goal1(goal))
        goal_out = torch.tanh(self.goal2(goal_out))


        #Concatenate observation+action as critic state
        state = torch.cat([obs, goal_out, action], 1)

        ###### Q1 HEAD ####
        q1 = torch.tanh(self.q1f1(state))
        q1 = self.q1ln1(q1)
        q1 = torch.tanh(self.q1f2(q1))
        q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)

        ###### Q2 HEAD ####
        q2 = torch.tanh(self.q2f1(state))
        q2 = self.q2ln1(q2)
        q2 = torch.tanh(self.q2f2(q2))
        q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)

        # ###### Value HEAD ####
        # v = torch.tanh(self.vf1(obs))
        # v = self.vln1(v)
        # v = torch.tanh(self.vf2(v))
        # v = self.vln2(v)
        # v = self.vout(v)

        #self.half()

        return q1, q2, None


# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight)

