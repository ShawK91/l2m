import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import soft_update, hard_update
from core.models import Actor, Critic


class SAC(object):
    def __init__(self, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy_type
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = False

        self.critic = Critic(args.state_dim, args.goal_dim, args.action_dim)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.critic_lr)

        self.critic_target = Critic(args.state_dim, args.goal_dim, args.action_dim)
        hard_update(self.critic_target, self.critic)


        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        if self.automatic_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor(1, args.action_dim)).cuda().item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = Adam([self.log_alpha], lr=args.alpha_lr)
            self.log_alpha.cuda()

        self.actor = Actor(args.state_dim, args.goal_dim, args.action_dim, args.policy_type)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        if torch.cuda.is_available():
            self.actor.cuda()
            self.critic.cuda()
            self.critic_target.cuda()




    def update_parameters(self, state_batch, next_state_batch, goal_batch, next_goal_batch, action_batch, reward_batch, done_batch):

        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, _ = self.actor.noisy_action(next_state_batch, next_goal_batch, return_only_action=False)
            qf1_next_target, qf2_next_target,_ = self.critic_target.forward(next_state_batch, next_goal_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + done_batch * self.gamma * (min_qf_next_target)

        qf1, qf2,_ = self.critic.forward(state_batch, goal_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _,_,_ = self.actor.noisy_action(state_batch, goal_batch, return_only_action=False)

        qf1_pi, qf2_pi, _ = self.critic.forward(state_batch, goal_batch, action_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss.backward(retain_graph=True)
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


        #if updates % self.target_update_interval == 0:
        soft_update(self.critic_target, self.critic, self.tau)

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