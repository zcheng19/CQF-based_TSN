from CQFsim import MulEnv
from ddqn import Network, init_weights
from torch import nn
import torch
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import os 


BATCH_SIZE=32
BUFFER_SIZE=int(1e5) # 1000000
MIN_REPLAY_SIZE=50000 # set the minimum update buffer length
EPSILON_START=0.4
EPSILON_END=0.1
EPSILON_DECAY=int(1e3)
EPISODES = 10000
LR = 1e-3
TARGET_UPDATE_FREQ = 50

filefullpath = "./fulldata.csv"
SAVE_INTERVAL = 100
SAVE_PATH = './model_fully/'

def plot_chart(epLst, rewardLst, epLen):
    plt.figure()
    # plt.plot(epLst, rewardLst)
    plt.savefig("rewards.png")
    plt.figure()
    # plt.plot(epLst, epLst)
    plt.savefig("flows.png")

def save_csv(epLst, rewardLst, epLen):
    L = []
    for i, j, z in zip(epLst, rewardLst, epLen):
        L.append([i, j, z])

    name=['episodes','rewards', 'flows']
    test=pd.DataFrame(columns=name,data=L) 
    test.to_csv('ddqndata.csv', mode="a", header=False, index=False)

if __name__ == "__main__":
    if os.path.exists(filefullpath):
        os.remove(filefullpath)

    myenv = MulEnv()
    device = torch.device('cpu' if torch.cuda.is_available() else "cpu")
    obs = myenv.reset()
    flow_len = len(myenv.flows)
    STEP = flow_len
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    episode_count = 0

    online_net = Network(myenv, obs, device=device)
    target_net = Network(myenv, obs, device=device)
    online_net.apply(init_weights)

    online_net = online_net.to(device)
    target_net = target_net.to(device)

    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

    # Initialize Replay Buffer
    reward_lst = []
    ep_lst = []
    ep_len = []
    for ep in range(EPISODES+1):
        reward = 0
        if ep:
            obs = myenv.reset()
        step_count = 0
        epsilon = 1.0
        # print(len(replay_buffer))
        for step in range(1, STEP+1):
            if len(replay_buffer) < MIN_REPLAY_SIZE:
                action = online_net.act(obs, 0.4, myenv, myenv.flows[step-1]["period"])[0]
                new_obs, rew, done = myenv.step(action, myenv.flows[step-1]["period"], myenv.flows[step]["period"])
                if step == flow_len - 3:
                    rew, done = 10, True
                reward += rew
                transition = (obs[0].copy(), action, rew, done, new_obs[0].copy())
                replay_buffer.append(transition)
                if done:
                    break
                obs = new_obs
            else:
                epsilon = np.interp(ep, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
                action = online_net.act(obs, epsilon, myenv, myenv.flows[step-1]["period"])[0]
                new_obs, rew, done = myenv.step(action, myenv.flows[step-1]["period"], myenv.flows[step]["period"])
                if step == flow_len - 3:
                    rew, done = 10, True
                reward += rew
                transition = (obs[0].copy(), action, rew, done, new_obs[0].copy())
                replay_buffer.append(transition)
                if done:
                    break
                obs = new_obs
            step_count += 1

            if len(replay_buffer) >= MIN_REPLAY_SIZE:
                transitions = random.sample(replay_buffer, BATCH_SIZE)
                loss = online_net.compute_loss(transitions, target_net)
                # Gradient Descent
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        # Update Target Network
        if ep % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())
        
        print("Episode "+str(ep)+" reward:", reward, "flows:", step_count)
        reward_lst.append(reward)
        ep_lst.append(ep)
        ep_len.append(step_count)
        
        if ep % SAVE_INTERVAL == 0 and ep != 0:
            save_csv(ep_lst, reward_lst, ep_len)
            reward_lst = []
            ep_lst = []
            ep_len = []

        if ep % SAVE_INTERVAL == 0 and ep != 0:
            print('Saving...')
            online_net.save(SAVE_PATH+str(ep)+".pack".format(LR))





