import opensim as osim
from osim.http.client import Client
import numpy as np
import torch
from models.constructor import ModelConstructor
from core import utils
from envs_repo.l2m import L2MRemote, L2M

FRAMESKIP = 4
seed = 'Results/Auxiliary/_bestR1__sac_cerl_seed2019_roll10_diff0_pop20_portfolio10_scheme_multipoint'

model_constructor = ModelConstructor(169, 22, actor_seed=seed, critic_seed=None)
net = model_constructor.make_model('Gaussian_FF', seed=True)

remote_base = "http://osim-rl-grader.aicrowd.com/"
aicrowd_token = "326a2beeb0b614b8c3f2f24124fdc2a5"  # use your aicrowd token


# your aicrowd token (API KEY) can be found at your profile page at https://www.aicrowd.com

def submit_server():



    client = Client(remote_base)
    env = L2MRemote(client, aicrowd_token, frameskip=FRAMESKIP)
    state = torch.Tensor(env.first_state)

    time = 0; total_reward = 0
    while True:
        time=time+FRAMESKIP

        action = net.clean_action(state).detach()
        hack_action = []
        for i in range(22):
            action_i = action[0, i].item()
            #action_i = (action_i+1.0)/2.0
            hack_action.append(action_i)

        state, reward, done, info = env.step(hack_action)
        state = torch.Tensor(state)
        total_reward+= reward


        print(time, '%.2f'%total_reward, ['%.2f'%act for act in hack_action])

        if done:
            state, experiment_done = env.reset()
            if experiment_done:
                break
            else:
                state = torch.Tensor(state)

    client.submit()


def test_locally():

    env = L2M(frameskip=FRAMESKIP, difficulty=0)
    state = torch.Tensor(env.reset())

    time = 0; total_reward = 0
    while True:
        time = time + FRAMESKIP

        action = net.clean_action(state).detach()
        hack_action = []
        for i in range(22):
            action_i = action[0, i].item()
            #action_i = (action_i + 1.0) / 2.0
            hack_action.append(action_i)

        state, reward, done, info = env.step(hack_action)
        state = torch.Tensor(state)
        total_reward += reward

        print('Local Test', time, 'R1_Reward','%.2f' % env.r1_reward, 'Shaped_Reward','%.2f' % total_reward, utils.pprint(hack_action))

        if done:
            break


test_locally()
#submit_server()
# client = Client(remote_base)
# env = L2MRemote(client, aicrowd_token, frameskip=FRAMESKIP)
# state = torch.Tensor(env.first_state)
#
# env_local = L2M(frameskip=FRAMESKIP)
# state_local = torch.Tensor(env_local.reset())
#
# a = 0