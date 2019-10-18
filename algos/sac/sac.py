import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import soft_update, hard_update


class SAC(object):
    def __init__(self, args, model_constructor, gamma, **kwargs):

        self.gamma = gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.sac_kwargs = kwargs

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = kwargs['autotune']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = model_constructor.make_model('Tri_Head_Q').to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=1e-4)

        self.critic_target = model_constructor.make_model('Tri_Head_Q').to(device=self.device)
        hard_update(self.critic_target, self.critic)


        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(1, model_constructor.action_dim)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr, weight_decay=1e-4)
            self.log_alpha = self.log_alpha.to(device=self.device)


        self.actor = model_constructor.make_model('Gaussian_FF', seed=True).to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=1e-4)

        # self.actor_target = model_constructor.make_model('actor')
        # hard_update(self.actor_target, self.actor)

        # if torch.cuda.is_available():
        #     self.actor.cuda()
        #     self.critic.cuda()
        #     self.critic_target.cuda()
        #     #self.actor_target.cuda()

        self.num_updates = 0

        # Statistics Tracker
        self.entropy = {'mean': 0, 'trace' :[]}
        self.next_entropy = {'mean': 0, 'trace' :[]}
        self.policy_q = {'mean': 0, 'trace' :[]}
        self.critic_loss = {'mean': 0, 'trace' :[]}

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

        with torch.no_grad():
            next_state_action, next_state_log_pi,_,_,_= self.actor.noisy_action(next_state_batch,  return_only_action=False)
            qf1_next_target, qf2_next_target,_ = self.critic_target.forward(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            if self.sac_kwargs['entropy']:
                min_qf_next_target -= self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1-done_batch) * self.gamma * (min_qf_next_target)
            self.compute_stats(next_state_log_pi, self.next_entropy)

        qf1, qf2,_ = self.critic.forward(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        self.compute_stats(qf1_loss, self.critic_loss)

        pi, log_pi, _,_,_ = self.actor.noisy_action(state_batch, return_only_action=False)
        self.compute_stats(log_pi, self.entropy)

        qf1_pi, qf2_pi, _ = self.critic.forward(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        self.compute_stats(min_qf_pi, self.policy_q)

        policy_loss = -min_qf_pi # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        if self.sac_kwargs['entropy']:
            policy_loss += self.alpha * log_pi
        policy_loss = policy_loss.mean()


        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward(retain_graph=True)
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        self.num_updates += 1
        if self.num_updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
            #soft_update(self.actor_target, self.actor, self.tau)

        #return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
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