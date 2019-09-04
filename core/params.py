import os

class Parameters:
    def __init__(self, parser, algo):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """

        #Env args
        self.env_name = vars(parser.parse_args())['env']
        self.env_args = get_env_args(self.env_name)


        #OTHER ARGS
        self.algo = algo
        self.total_steps = int(vars(parser.parse_args())['total_steps'] * 1000000)
        self.gradperstep = vars(parser.parse_args())['gradperstep']
        self.savetag = vars(parser.parse_args())['savetag']
        self.seed = vars(parser.parse_args())['seed']
        self.gpu_device = vars(parser.parse_args())['gpu_id']
        self.batch_size = vars(parser.parse_args())['batchsize']
        self.rollout_size = vars(parser.parse_args())['rollsize']

        self.critic_lr = vars(parser.parse_args())['critic_lr']
        self.actor_lr = vars(parser.parse_args())['actor_lr']
        self.tau = vars(parser.parse_args())['tau']
        self.gamma = vars(parser.parse_args())['gamma']
        self.buffer_size = int(vars(parser.parse_args())['buffer'] * 1000000)
        self.policy_type = 'DeterministicPolicy' if (algo == 'td3' or algo == 'cerl_td3') else 'GaussianPolicy'
        self.actor_seed = None
        self.critic_seed = None


        if algo == 'cerl_sac' or algo == 'cerl_td3':
            self.pop_size = vars(parser.parse_args())['popsize']
            self.portfolio_id = vars(parser.parse_args())['portfolio']
            self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution

            #Non-Args Params
            self.ucb_coefficient = 0.9 #Exploration coefficient in UCB
            self.elite_fraction = 0.2
            self.crossover_prob = 0.15
            self.mutation_prob = 0.90
            self.extinction_prob = 0.005  # Probability of extinction event
            self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
            self.weight_magnitude_limit = 10000000
            self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform

        elif algo == 'sac':
            self.alpha = 0.2
            self.target_update_interval = 2
            self.alpha_lr = 1e-3

        #Save Results
        self.savefolder = 'Results/'
        if not os.path.exists('Results/'): os.makedirs('Results/')
        self.aux_folder = self.savefolder + 'Auxiliary/'
        if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)

        self.savetag += str(algo)
        self.savetag += '_s' + str(self.seed)
        self.savetag += '_roll' + str(self.rollout_size)
        if algo == 'cerl_sac' or algo == 'cerl_td3':
            self.savetag += '_pop' + str(self.pop_size)
            self.savetag += '_portfolio' + str(self.portfolio_id)

def get_env_args(env_name):
    args = {}
    if env_name == 'l2m':

        args['visualize'] = False
        args['integrator_accuracy'] = 5e-5
        args['frameskip'] = 4
        args['T'] = 1000



    return args