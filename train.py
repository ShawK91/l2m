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
parser.add_argument('--seed', type=int, help='Seed', default=59)
parser.add_argument('--savetag', type=str, help='#Tag to append to savefile',  default='')
parser.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)
parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=40)
parser.add_argument('--buffer', type=float, help='Buffer size in million',  default=1.0)
parser.add_argument('--env', type=str, help='Env Name',  default='l2m')
parser.add_argument('--config', type=str, help='Config Name',  default='')
parser.add_argument('--action_clamp', type=utils.str2bool, help='Clamp action?',  default=True)
parser.add_argument('--algo', type=str, help='Which algo? - CERL_SAC, CERL_TD3, TD3, SAC ',  default='sac')
parser.add_argument('--cerl', type=utils.str2bool, help='#Use CERL?',  default=False)

parser.add_argument('--critic_lr', type=float, help='Critic learning rate?', default=1e-3)
parser.add_argument('--actor_lr', type=float, help='Actor learning rate?', default=1e-3)
parser.add_argument('--tau', type=float, help='Tau', default=1e-3)
parser.add_argument('--gamma', type=float, help='Discount Rate', default=0.99)
parser.add_argument('--batchsize', type=int, help='Seed',  default=256)
parser.add_argument('--reward_scale', type=float, help='Reward Scaling Multiplier',  default=1.0)
parser.add_argument('--learning_start', type=int, help='Frames to wait before learning starts',  default=5000)

#ALGO SPECIFIC ARGS
parser.add_argument('--popsize', type=int, help='#Policies in the population',  default=10)
parser.add_argument('--rollsize', type=int, help='#Policies in rollout size',  default=10)
parser.add_argument('--scheme', type=str, help='#Neuroevolution Scheme? standard Vs. multipoint',  default='standard')
parser.add_argument('--gradperstep', type=float, help='#Gradient step per env step',  default=0.5)
parser.add_argument('--portfolio', type=int, help='Portfolio ID',  default=-1)

parser.add_argument('--autotune', type=utils.str2bool, help='Autotune SAC entropy?',  default=True)
parser.add_argument('--T', type=int, help='Time Length?',  default=200)
parser.add_argument('--difficulty', type=int, help='Difficulty Level',  default=0)
parser.add_argument('--alpha', type=float, help='SAC Alpha',  default=0.3)
parser.add_argument('--project', type=utils.str2bool, help='#Project OBS?',  default=False)

USE_CERL = vars(parser.parse_args())['cerl']
ALGO = vars(parser.parse_args())['algo']
#Figure out GPU to use if any
if vars(parser.parse_args())['gpu_id'] != -1: os.environ["CUDA_VISIBLE_DEVICES"]=str(vars(parser.parse_args())['gpu_id'])

#######################  Construct ARGS Class to hold all parameters ######################
args = Parameters(parser, ALGO)

#Set seeds
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

################################## Find and Set MDP (environment constructor) ########################
env_constructor = EnvConstructor(args.env_name, args.config, **args.env_args)

#######################  Actor, Critic and ValueFunction Model Constructor ######################
model_constructor = ModelConstructor(env_constructor.dummy_env.state_dim, env_constructor.dummy_env.action_dim, args.actor_seed, args.critic_seed)



if USE_CERL:
	from algos.cerl.cerl_trainer import CERL_Trainer
	ai = CERL_Trainer(args, model_constructor, env_constructor)
	ai.train(args.total_steps)

else:
	if ALGO == 'td3':
		from algos.td3.td3_trainer import TD3_Trainer
		ai = TD3_Trainer(args, model_constructor, env_constructor)
		ai.train(args.total_steps)

	elif ALGO == 'sac':
		from algos.sac.sac_trainer import SAC_Trainer
		ai = SAC_Trainer(args, model_constructor, env_constructor)
		ai.train(args.total_steps)

	elif ALGO == 'sac_discrete':
		from algos.sac_discrete.sac_discrete_trainer import SAC_Discrete_Trainer
		ai = SAC_Discrete_Trainer(args, model_constructor, env_constructor)
		ai.train(args.total_steps)

	elif ALGO == 'ddqn':
		from algos.ddqn.ddqn_trainer import DDQN_Trainer
		ai = DDQN_Trainer(args, model_constructor, env_constructor)
		ai.train(args.total_steps)

	else:
		raise ValueError('Unknown choice for Algo')

#Initial Print
print('Running', ALGO)

time_start = time.time()
