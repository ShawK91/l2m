import torch, random
import torch.nn as nn
from torch.distributions import Categorical, Normal
from core.utils import GumbelSoftmax



class DDQN(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay_frames):
        super(DDQN, self).__init__()
        self.action_dim = action_dim
        h1=128; h2 =128; g1 = 40; g2 = 40


        #Goal+Feature Processor
        self.feature1 = nn.Linear(97, h1)
        self.goal1 = nn.Linear(72, g1)
        self.goal2 = nn.Linear(g1, g2)


        #Shared FF
        self.linear1 = nn.Linear(h1+g2, h1)
        self.linear2 = nn.Linear(h1, h2)


        #Value
        self.val = nn.Linear(h2, 1)


        #Advantages
        self.adv = nn.Linear(h2, action_dim)

        #self.half()
        #weights_init_(self, lin_gain=1.0, bias_gain=0.1)

        #Epsilon Decay
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_frames



    def clean_action(self, state, return_only_action=True):
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

        logits = val + adv - adv.mean()


        if return_only_action:
            return self.multi_argmax(logits)
        else:
            return self.multi_argmax(logits), None, logits

    def noisy_action(self, obs, return_only_action=True):
        _, _, logits = self.clean_action(obs, return_only_action=False)

        # dist = GumbelSoftmax(temperature=1, logits=q)
        # action = dist.sample()
        # #action = q.argmax(1)

        action = self.multi_argmax(logits, epsilon=self.epsilon)



        if return_only_action:
            return action

        #log_prob = dist.log_prob(action)

        #print(action[0].detach().item(), log_prob[0].detach().item())
        return action, None, logits

    def multi_argmax(self, logits, epsilon=None):

        out = [logits[:,i:i+3].argmax(1).unsqueeze(1) for i in range(0, 66, 3)]
        out = torch.cat(out, axis=1)

        #Epsilon Greedy
        if epsilon != None:
            if random.random() < epsilon:
                mask = (torch.rand(out.shape) < 0.1).long()
                rand_uniform = torch.randint(0,3, out.shape)
                out = out * ( -mask) + rand_uniform * mask
                if self.epsilon > self.epsilon_end:
                    self.epsilon -= self.epsilon_decay_rate

        out -= 1 #Translate 0,1,2 to -1,0,1

        return out

