import opensim as osim
from osim.http.client import Client
import numpy as np
import torch
from core.models import Actor
from envs_repo.env_wrapper import L2MWrapper, flatten
from core import utils



#Upload
file = 'Results/Auxiliary/_bestcerl_td3_s2019_roll10_pop10_portfolio10'

net = Actor(97, 72, 22, 'DeterministicPolicy')
net.load_state_dict(torch.load(file))



# Settings
remote_base = "http://osim-rl-grader.aicrowd.com/"
aicrowd_token = "326a2beeb0b614b8c3f2f24124fdc2a5" # use your aicrowd token
# your aicrowd token (API KEY) can be found at your profile page at https://www.aicrowd.com

client = Client(remote_base)

# Create environment
observation = client.env_create(aicrowd_token, env_id='L2M2019Env')

time = 0; total_reward = 0
while True:
    time=time+1
    obs = np.expand_dims(flatten(observation), 0)
    goal = np.array(observation['v_tgt_field'])
    goal = goal[:, 0::2, 0::2].flatten()
    goal = goal.reshape(1, len(goal))
    obs = utils.to_tensor(obs)
    goal = utils.to_tensor(goal)

    #action = list(net.clean_action(obs, goal).detach().numpy().flatten())
    action = net.clean_action(obs, goal).detach()

    hack_action = []
    for i in range(22):
        hack_action.append(action[0,i].item())

    [observation, reward, done, info] = client.env_step(hack_action)
    total_reward+= reward
    print(time, total_reward)
    if done:
        observation = client.env_reset()
        if not observation:
            break

client.submit()