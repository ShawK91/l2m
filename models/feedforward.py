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
from torch.distributions import Normal, Cauchy, RelaxedOneHotCategorical
from core.utils import weights_init_

class Gumbel_FF(nn.Module):
    """Actor model
        Parameters:
              args (object): Parameter class
    """

    def __init__(self, num_inputs, num_actions, num_heads=3):
        super(Gumbel_FF, self).__init__()

        self.num_actions = num_actions
        self.num_heads = num_heads

        h1=128; h2 =128; g1 = 40; g2 = 40

        #Goal+Feature Processor
        self.feature1 = nn.Linear(97, h1)
        self.goal1 = nn.Linear(72, g1)
        self.goal2 = nn.Linear(g1, g2)


        #Shared FF
        self.linear1 = nn.Linear(h1+g2, h1)
        self.linear2 = nn.Linear(h1, h2)

        #Value
        self.val = nn.Linear(h2, num_actions)


        #Advantages
        self.adv = nn.Linear(h2, num_actions*num_heads)

        #Temperatures
        self.temperature = nn.Linear(h2, 1)


        # SAC SPECIFIC
        self.TEMP_MAX = 20
        self.TEMP_MIN = 0.2



    def clean_action(self, state, return_only_action=True):
        """Method to forward propagate through the actor's graph
            Parameters:
                  input (tensor): states
            Returns:
                  action (tensor): actions
        """
        #x = self.knet(state)

        #Goal+Feature Processor
        obs = self.feature1(state[:,0:97])
        obs = torch.selu(obs)

        dict = self.goal1(state[:,97:])
        dict = torch.selu(dict)
        dict = self.goal2(dict)
        dict = torch.selu(dict)

        x = torch.cat([obs, dict], axis=1)

        #Shared
        x = torch.selu(self.linear1(x))
        x = torch.selu(self.linear2(x))

        val = self.val(x)
        adv = self.adv(x)

        #DDQN Style baselining
        for i in range(0, self.num_heads*self.num_actions, self.num_heads):

            # if len(state) > 1:
            #     print(val[:,int(i/self.num_heads)].shape, adv[:,i:i+self.num_heads].shape, adv[:,i:i+self.num_heads].mean().shape)
            #     input()

            adv[:,i:i+self.num_heads] = val[:,int(i/self.num_heads)].unsqueeze(1) + adv[:,i:i+self.num_heads] -  adv[:,i:i+self.num_heads].mean()

        if return_only_action: return self.multi_argmax(logits=adv)

        temp = self.temperature(x)
        temp = torch.clamp(temp, min=self.TEMP_MIN, max=self.TEMP_MAX)
        return adv, temp


    def noisy_action(self, state, return_only_action=True):
        adv, temp = self.clean_action(state, return_only_action=False)

        if return_only_action:
            action = self.multi_dist_sample(adv, temp, return_only_action)
            return action
        else:
            action, log_prob = self.multi_dist_sample(adv, temp, return_only_action)


        # dist = RelaxedOneHotCategorical(logits=adv, temperature=temp)
        # action = dist.rsample()  # for reparameterization trick
        #
        # if return_only_action:
        #     return action.argmax(1)
        #
        # log_prob = dist.log_prob(action).t()[:,0:1]
        #log_prob[log_prob != log_prob] = 0.0 #Set NaNs to 0

        #log_prob = mean[action.argmax(1)][:,1:]

        return action, log_prob, None, None, temp

    def multi_argmax(self, logits):

        out = [logits[:,i:i+self.num_heads].argmax(1).unsqueeze(1) for i in range(0, self.num_heads*self.num_actions, self.num_heads)]
        out = torch.cat(out, axis=1).float()

        # #Epsilon Greedy
        # if epsilon != None:
        #     if random.random() < epsilon:
        #         mask = (torch.rand(out.shape) < 0.1).long()
        #         rand_uniform = torch.randint(0,3, out.shape)
        #         out = out * ( -mask) + rand_uniform * mask
        #         if self.epsilon > self.epsilon_end:
        #             self.epsilon -= self.epsilon_decay_rate

        out -= 1 #Translate 0,1,2 to -1,0,1

        return out

    def multi_dist_sample(self, logits, temp, return_only_action):
        action = []; log_prob = []
        for i in range(0, self.num_heads*self.num_actions, self.num_heads):
            dist = RelaxedOneHotCategorical(logits=logits[:,i:i+self.num_heads], temperature=temp)
            raw_action = dist.rsample()
            action.append(raw_action.argmax(1).unsqueeze(1))


            if not return_only_action: log_prob.append(dist.log_prob(raw_action).t()[:,0:1])

        action = torch.cat(action, 1).float()
        if return_only_action: return action

        log_prob = torch.cat(log_prob, 1).mean(1).unsqueeze(1)
        return action, log_prob


    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean


class Deterministic_FF(nn.Module):
    """Actor model
        Parameters:
              args (object): Parameter class
    """

    def __init__(self, num_inputs, num_actions):
        super(Deterministic_FF, self).__init__()


        self.stochastic = False
        self.num_actions = num_actions

        h1=128; h2 =128

        #Shared FF
        self.linear1 = nn.Linear(num_inputs, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.mean_linear = nn.Linear(h2, num_actions)

        #Knet
        #self.knet = KNet(num_inputs+g1, h1, 5, h2)

        self.noise = Normal(0, 0.01)

        weights_init_(self, lin_gain=1.0, bias_gain=0.1)



    def clean_action(self, state):
        """Method to forward propagate through the actor's graph
            Parameters:
                  input (tensor): states
            Returns:
                  action (tensor): actions
        """
        #x = self.knet(state)
        x = torch.selu(self.linear1(state))
        x = torch.selu(self.linear2(x))
        mean = self.mean_linear(x)

        return torch.tanh(mean)


    def noisy_action(self, state):
        mean = self.clean_action(state)
        action = mean + torch.clamp(self.noise.sample((len(state), self.num_actions)), min=-0.5, max=0.5)

        return action


    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean




class Gaussian_FF(nn.Module):
    """Actor model
        Parameters:
              args (object): Parameter class
    """

    def __init__(self, num_inputs, num_actions):
        super(Gaussian_FF, self).__init__()

        self.stochastic = True
        self.num_actions = num_actions
        h1=128; h2 =128; g1 = 40; g2 = 40


        #Goal+Feature Processor
        self.feature1 = nn.Linear(97, h1)
        self.goal1 = nn.Linear(72, g1)
        self.goal2 = nn.Linear(g1, g2)


        #Shared FF
        self.linear1 = nn.Linear(h1+g2, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.mean_linear = nn.Linear(h2, num_actions)

        #Knet
        #self.knet = KNet(num_inputs+g1, h1, 5, h2)

        self.log_std_linear = nn.Linear(h2, num_actions)


        #weights_init_(self, lin_gain=1.0, bias_gain=0.1)

        # SAC SPECIFIC
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20
        self.epsilon = 1e-6




    def clean_action(self, state, return_only_action=True):
        """Method to forward propagate through the actor's graph
            Parameters:
                  input (tensor): states
            Returns:
                  action (tensor): actions
        """
        #x = self.knet(state)


        #Goal+Feature Processor
        obs = self.feature1(state[:,0:97])
        obs = torch.selu(obs)

        dict = self.goal1(state[:,97:])
        dict = torch.selu(dict)
        dict = self.goal2(dict)
        dict = torch.selu(dict)

        x = torch.cat([obs, dict], axis=1)

        #Shared
        x = torch.selu(self.linear1(x))
        x = torch.selu(self.linear2(x))
        mean = self.mean_linear(x)

        if return_only_action: return torch.tanh(mean)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std


    def noisy_action(self, state,return_only_action=True):
        mean, log_std = self.clean_action(state, return_only_action=False)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)

        if return_only_action:
            return action

        log_prob = self.compute_log_prob(x_t, action, normal)

        return action, log_prob, x_t, normal, torch.tanh(mean)


    def compute_log_prob(self, var, action, dist):
        log_prob = dist.log_prob(var)

        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return log_prob


    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean



class Tri_Head_Q(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim):
        super(Tri_Head_Q, self).__init__()
        l1 = 128; l2 = 128; g1 = 40; g2 = 40


        #Goal+Feature Processor
        self.feature1 = nn.Linear(97, l1)
        self.goal1 = nn.Linear(72, g1)
        self.goal2 = nn.Linear(g1, g2)


        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.q1f1 = nn.Linear(l1 + action_dim + g2, l1)
        #self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q1f2 = nn.Linear(l1, l2)
        #self.q1ln2 = nn.LayerNorm(l2)

        #Out
        self.q1out = nn.Linear(l2, 1)


        ######################## Q2 Head ##################
        # Construct Hidden Layer 1 with state
        self.q2f1 = nn.Linear(l1 + action_dim + g2, l1)
        #self.q2ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q2f2 = nn.Linear(l1, l2)
        #self.q2ln2 = nn.LayerNorm(l2)

        #Out
        self.q2out = nn.Linear(l2, 1)

        #self.half()
        #weights_init_(self, lin_gain=1.0, bias_gain=0.1)







    def forward(self, inp, action):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """
        #Goal+Feature Processor
        obs = self.feature1(inp[:,0:97])
        obs = torch.selu(obs)

        dict = self.goal1(inp[:,97:])
        dict = torch.selu(dict)
        dict = self.goal2(dict)
        dict = torch.selu(dict)



        #Concatenate observation+action as critic state
        state = torch.cat([obs, dict, action], 1)

        ###### Q1 HEAD ####
        q1 = torch.selu(self.q1f1(state))
        #q1 = self.q1ln1(q1)
        q1 = torch.selu(self.q1f2(q1))
        #q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)

        ###### Q2 HEAD ####
        q2 = torch.selu(self.q2f1(state))
        #q2 = self.q2ln1(q2)
        q2 = torch.selu(self.q2f2(q2))
        #q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)

        # ###### Value HEAD ####
        # v = torch.tanh(self.vf1(obs))
        # v = self.vln1(v)
        # v = torch.tanh(self.vf2(v))
        # v = self.vln2(v)
        # v = self.vout(v)

        #self.half()

        return q1, q2, None




# class QActor(nn.Module):
#     """Actor model
#         Parameters:
#               args (object): Parameter class
#     """
#
#     def __init__(self, num_inputs, num_actions, policy_type):
#         super(QActor, self).__init__()
#
#         self.policy_type = policy_type; self.num_actions = num_actions
#
#         h1=128; h2 =128
#
#         #Shared FF
#         self.linear1 = nn.Linear(num_inputs, h1)
#         self.linear2 = nn.Linear(h1, h2)
#         self.mean_linear = nn.Linear(h2, num_actions)
#
#         #Knet
#         #self.knet = KNet(num_inputs+g1, h1, 5, h2)
#
#         if self.policy_type == 'GaussianPolicy':
#             self.log_std_linear = nn.Linear(h2, num_actions)
#
#         elif self.policy_type == 'DeterministicPolicy':
#             self.noise = Normal(0, 0.01)
#
#         self.apply(weights_init_)
#
#
#
#     def clean_action(self, state, return_only_action=True):
#         """Method to forward propagate through the actor's graph
#             Parameters:
#                   input (tensor): states
#             Returns:
#                   action (tensor): actions
#         """
#         #x = self.knet(state)
#         x = torch.relu(self.linear1(state))
#         x = torch.relu(self.linear2(x))
#         mean = self.mean_linear(x)
#
#         if return_only_action or self.policy_type == 'DeterministicPolicy': return mean.argmax(1)
#
#         elif self.policy_type == 'GaussianPolicy':
#             log_std = self.log_std_linear(x)
#             log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
#             return mean, log_std
#
#
#     def noisy_action(self, state,return_only_action=True):
#
#         if self.policy_type == 'GaussianPolicy':
#             mean, log_std = self.clean_action(state, return_only_action=False)
#             std = log_std.exp()
#             normal = Normal(mean, std)
#             x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
#             action = x_t.argmax(1)
#
#             if return_only_action:
#                 return action
#             log_prob = normal.log_prob(x_t)
#
#
#
#             return action, log_prob, x_t
#
#         elif self.policy_type == 'DeterministicPolicy':
#             mean = self.clean_action(state)
#             action = mean + torch.clamp(self.noise.sample((len(state), self.num_actions)), min=-0.5, max=0.5)
#             Exception('Not Implemented Noisy Action for deterministic QActor')
#
#             if return_only_action: return action
#             else: return action, None, None
#
#
#     def get_norm_stats(self):
#         minimum = min([torch.min(param).item() for param in self.parameters()])
#         maximum = max([torch.max(param).item() for param in self.parameters()])
#         means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
#         mean = sum(means)/len(means)
#
#         return minimum, maximum, mean