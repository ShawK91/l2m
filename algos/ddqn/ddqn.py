import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import soft_update, hard_update






class DDQN(object):
    def __init__(self, args, model_constructor, gamma):

        self.gamma = gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = model_constructor.make_model('DDQN').to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.actor_target = model_constructor.make_model('DDQN').to(device=self.device)
        hard_update(self.actor_target, self.actor)


        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(1, args.action_dim)).cuda().item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
            self.log_alpha.cuda()

        self.num_updates = 0

        # Statistics Tracker
        self.entropy = {'mean': 0, 'trace' :[]}
        self.next_entropy = {'mean': 0, 'trace' :[]}
        self.critic_loss = {'mean': 0, 'trace' :[]}
        self.policy_q = {'mean': 0, 'trace' :[]}
        self.next_q = {'mean': 0, 'trace' :[]}

    def compute_stats(self, tensor, tracker):
        """Computes stats from intermediate tensors

             Parameters:
                   tensor (tensor): tensor
                   tracker (object): logger

             Returns:
                   None


         """
        tracker['trace'].append(torch.mean(tensor).item())
        tracker['mean'] = sum(tracker['trace'])/len(tracker['trace'])

        if len(tracker['trace']) > 10000: tracker['trace'].pop(0)




    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch):

        action_batch = action_batch.long()
        with torch.no_grad():
            na = self.actor.clean_action(next_state_batch,  return_only_action=True)
            _, _, ns_logits = self.actor_target.clean_action(next_state_batch, return_only_action=False)

            #Compute Duelling Q-Val
            #next_target = self.compute_duelling_out(ns_logits, na)
            next_target = [ns_logits[:,i:i+3][na[:,i]][:,-1:] for i in range(22)]
            next_target = torch.cat(next_target, axis=1)

            #Entropy
            #next_target -= self.alpha * ns_log_prob.unsqueeze(1)

            next_q_value = reward_batch + (1-done_batch) * self.gamma * next_target
            #self.compute_stats(ns_log_prob, self.next_entropy)
            self.compute_stats(next_q_value, self.next_q)


        # Compute Duelling Q-Val
        _, _, logits  = self.actor.clean_action(state_batch, return_only_action=False)
        q_val = [logits[:,i:i+3][action_batch[:,i]][:,-1:] for i in range(22)]
        q_val = torch.cat(q_val, axis=1)
        #q_val = self.compute_duelling_out(logits, action_batch)
        #self.compute_stats(log_prob, self.entropy)
        self.compute_stats(q_val, self.policy_q)

        loss_function = torch.nn.MSELoss()
        q_loss = loss_function(next_q_value, q_val)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        self.compute_stats(q_loss, self.critic_loss)

        self.actor_optim.zero_grad()
        q_loss.backward()
        self.actor_optim.step()



        # if self.automatic_entropy_tuning:
        #     alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        #
        #     self.alpha_optim.zero_grad()
        #     alpha_loss.backward(retain_graph=True)
        #     self.alpha_optim.step()
        #
        #     self.alpha = self.log_alpha.exp()
        #     alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        # else:
        #     alpha_loss = torch.tensor(0.)
        #     alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        self.num_updates += 1
        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.actor_target, self.actor, self.tau)
            #soft_update(self.actor_target, self.actor, self.tau)

        # if self.num_updates % 1000 == 0:
        #     hard_update(self.QActor_target, self.QActor)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()



    #Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))