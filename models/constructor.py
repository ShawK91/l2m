from models.models import Actor, Critic


class ModelConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, state_dim, goal_dim, action_dim, policy_type, actor_seed=None, critic_seed=None):
        """
        A general Environment Constructor
        """
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.policy_type = policy_type
        self.actor_seed = actor_seed
        self.critic_seed = critic_seed


    def make_model(self, type, seed=False):
        """
        Generate and return an model object
        """

        if type == 'actor':
            model = Actor(self.state_dim, self.goal_dim, self.action_dim, self.policy_type)
            if seed:
                import torch
                model.load_state_dict(torch.load(self.actor_seed))
                print('Actor seeded from', self.actor_seed)

        elif type == 'critic':
            model = Critic(self.state_dim, self.goal_dim, self.action_dim)
            if seed:
                import torch
                model.load_state_dict(torch.load(self.critic_seed))
                print('Critic seeded from', self.critic_seed)


        return model



