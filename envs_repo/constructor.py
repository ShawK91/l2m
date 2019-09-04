
class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, name, **kwargs):
        """
        A general Environment Constructor
        """
        self.name = name
        self.kwargs = kwargs
        self.dummy_env = self.make_env()



    def make_env(self, **kwargs):
        """
        Generate and return an env object
        """
        #Look if the default arguments are overwritten
        if kwargs == None: kwargs = self.kwargs

        if self.name == 'l2m':
            from envs_repo.l2m import L2M
            env = L2M(self.kwargs['visualize'], self.kwargs['integrator_accuracy'], self.kwargs['frameskip'], self.kwargs['T'])
            return env



