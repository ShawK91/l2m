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
        self.config = vars(parser.parse_args())['config']
        self.difficulty = vars(parser.parse_args())['difficulty']
        self.action_clamp = vars(parser.parse_args())['action_clamp']
        self.T = vars(parser.parse_args())['T']
        self.env_args = get_env_args(self.env_name, self.difficulty, self.action_clamp, self.T)
        self.is_cerl = vars(parser.parse_args())['cerl']



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
        self.reward_scaling = vars(parser.parse_args())['reward_scale']
        self.buffer_size = int(vars(parser.parse_args())['buffer'] * 1000000)
        self.actor_seed = None
        self.critic_seed = None
        self.learning_start = vars(parser.parse_args())['learning_start']



        if self.is_cerl:
            self.pop_size = vars(parser.parse_args())['popsize']
            self.portfolio_id = vars(parser.parse_args())['portfolio']
            self.asynch_frac = 1.0  # Aynchronosity of NeuroEvolution

            #Non-Args Params
            self.ucb_coefficient = 0.9 #Exploration coefficient in UCB
            self.elite_fraction = 0.15
            self.crossover_prob = 0.15
            self.mutation_prob = 0.90
            self.extinction_prob = 0.005  # Probability of extinction event
            self.extinction_magnitude= 0.5  # Probabilty of extinction for each genome, given an extinction event
            self.weight_clamp = 10000000
            self.mut_distribution = 1  # 1-Gaussian, 2-Laplace, 3-Uniform
            self.lineage_depth = 10
            self.ccea_reduction = 'leniency'
            self.num_anchors = 4
            self.num_blends = 1
            self.scheme = vars(parser.parse_args())['scheme']


        self.alpha = 0.2
        self.autotune = vars(parser.parse_args())['autotune']
        self.target_update_interval = 1
        self.alpha_lr = 1e-3

        #Save Results
        self.savefolder = 'Results/'
        if not os.path.exists('Results/'): os.makedirs('Results/')
        self.aux_folder = self.savefolder + 'Auxiliary/'
        if not os.path.exists(self.aux_folder): os.makedirs(self.aux_folder)

        self.savetag += str(self.config)
        self.savetag += '_' + str(algo)
        self.savetag += '_cerl' if self.is_cerl else ''
        self.savetag += '_seed' + str(self.seed)
        self.savetag += '_roll' + str(self.rollout_size)
        self.savetag += '_diff' + str(self.difficulty)
        self.savetag += '_actionClamp' if self.action_clamp else ''
        self.savetag += '_entropyAuto' if self.autotune and self.algo == 'sac' else ''
        self.savetag += '_T' + str(self.T)



        if self.is_cerl:
            self.savetag += '_pop' + str(self.pop_size)
            self.savetag += '_portfolio' + str(self.portfolio_id)
            self.savetag += '_scheme_' + self.scheme

def get_env_args(env_name, difficulty, action_clamp, T):
    args = {}
    if env_name == 'l2m':

        args['visualize'] = False
        args['integrator_accuracy'] = 5e-5
        args['frameskip'] = 4
        args['T'] = T
        args['difficulty'] = difficulty
        args['action_clamp'] = action_clamp


    if env_name == 'gym':
        args['frameskip'] = 1



    return args