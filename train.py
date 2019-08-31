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
from core.params import Parameters
import argparse, torch
parser = argparse.ArgumentParser()


#COMMON ARGS
parser.add_argument('--seed', type=int, help='Seed', default=2019)
parser.add_argument('--savetag', type=str, help='#Tag to append to savefile',  default='')
parser.add_argument('--gpu_id', type=int, help='#GPU ID ',  default=0)
parser.add_argument('--total_steps', type=float, help='#Total steps in the env in millions ', default=40)
parser.add_argument('--algo', type=str, help='Which algo? - CERL_SAC, CERL_TD3, TD3, SAC ',  default='sac')
parser.add_argument('--buffer', type=float, help='Buffer size in million',  default=1)

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

#Get all arguments
args = Parameters(parser, ALGO)

#Set seeds
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)


if ALGO == 'cerl_sac' or ALGO == 'cerl_td3':
	from algos.cerl.cerl_trainer import CERL_Trainer
	ai = CERL_Trainer(args)
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
