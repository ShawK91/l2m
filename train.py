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
import numpy as np, os, time, random
from core import utils as utils
from envs_repo.constructor import EnvConstructor
from models.constructor import ModelConstructor
from core.params import Parameters
import argparse, torch
parser = argparse.ArgumentParser()


#######################  COMMANDLINE - ARGUMENTS ######################
parser.add_argument('--seed', type=int, help='Seed', default=2019)
parser.add_argument('--savetag', type=str, help='#Tag to append to savefile',  default='debug')
parser.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)
parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=40)
parser.add_argument('--algo', type=str, help='Which algo? - CERL_SAC, CERL_TD3, TD3, SAC ',  default='cerl_td3')
parser.add_argument('--buffer', type=float, help='Buffer size in million',  default=1)
parser.add_argument('--env', type=str, help='Env Name',  default='l2m')

parser.add_argument('--critic_lr', type=float, help='Critic learning rate? - Actor', default='5e-4')
parser.add_argument('--actor_lr', type=float, help='Actor learning rate? - Actor', default='5e-4')
parser.add_argument('--tau', type=float, help='Actor learning rate? - Actor', default='1e-3')
parser.add_argument('--gamma', type=float, help='Actor learning rate? - Actor', default='0.99')
parser.add_argument('--batchsize', type=int, help='Seed',  default=256)

#ALGO SPECIFIC ARGS
parser.add_argument('--popsize', type=int, help='#Policies in the population',  default=15)
parser.add_argument('--rollsize', type=int, help='#Policies in rolout size',  default=20)
parser.add_argument('--gradperstep', type=float, help='#Gradient step per env step',  default=1.0)
parser.add_argument('--portfolio', type=int, help='Portfolio ID',  default=10)

ALGO = vars(parser.parse_args())['algo']
#Figure out GPU to use if any
if vars(parser.parse_args())['gpu_id'] != -1: os.environ["CUDA_VISIBLE_DEVICES"]=str(vars(parser.parse_args())['gpu_id'])

#######################  Construct ARGS Class to hold all parameters ######################
args = Parameters(parser, ALGO)

#Set seeds
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

################################## Find and Set MDP (environment constructor) ########################
env_constructor = EnvConstructor(args.env_name, **args.env_args)

#######################  Actor, Critic and ValueFunction Model Constructor ######################
model_constructor = ModelConstructor(env_constructor.dummy_env.state_dim, env_constructor.dummy_env.goal_dim, env_constructor.dummy_env.action_dim, args.policy_type, args.actor_seed, args.critic_seed)


if ALGO == 'cerl_sac' or ALGO == 'cerl_td3':
	from algos.cerl.cerl_trainer import CERL_Trainer
	ai = CERL_Trainer(args, model_constructor, env_constructor)
	ai.train(args.total_steps)

elif ALGO == 'td3':
	from algos.td3.td3 import TD3


elif ALGO == 'sac':
	from algos.sac.sac_trainer import SAC_Trainer
	ai = SAC_Trainer(args)
	ai.train(args.total_steps)

else:
	raise ValueError('Unknown choice for Algo')

#Initial Print
print('Running', ALGO)

time_start = time.time()
