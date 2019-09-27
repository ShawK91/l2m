import torch

class ModelConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, state_dim, action_dim, actor_seed=None, critic_seed=None):
        """
        A general Environment Constructor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_seed = actor_seed
        self.critic_seed = critic_seed


    def make_model(self, type, seed=False):
        """
        Generate and return an model object
        """

        if type == 'Deterministic_FF':
            from models.feedforward import Deterministic_FF
            model = Deterministic_FF(self.state_dim, self.action_dim)
            if seed:
                model.load_state_dict(torch.load(self.actor_seed))
                print('Deterministic FF seeded from', self.actor_seed)

        elif type == 'Gaussian_FF':
            from models.feedforward import Gaussian_FF
            model = Gaussian_FF(self.state_dim, self.action_dim)
            if seed:
                model.load_state_dict(torch.load(self.actor_seed))
                print('Actor seeded from', self.actor_seed)

        elif type == 'Tri_Head_Q':
            from models.feedforward import Tri_Head_Q
            model = Tri_Head_Q(self.state_dim, self.action_dim)
            if seed:
                model.load_state_dict(torch.load(self.critic_seed))
                print('Critic seeded from', self.critic_seed)


        elif type == 'DDQN':
            from models.ddqn import DDQN
            model = DDQN(self.state_dim, self.action_dim*3, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_frames=50000)
            # if seed:
            #     import torch
            #     model.load_state_dict(torch.load(self.QActor_seed))
            #     print('DDQN seeded from', self.QActor_seed)


        return model



