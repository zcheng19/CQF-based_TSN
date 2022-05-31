import sys
import torch
import numpy as np 
import time
from CQFsim import FRAME_SIZE, MulEnv
from scheduler import Network, init_weights
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()
import pandas as pd 
#import matplotlib.pyplot as plt

LR = 1e-5
LOAD_PATH = "./model_double/"+sys.argv[1]+".pack".format(LR)

myenv = MulEnv()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
obs = myenv.reset()
flow_len = len(myenv.flows)
STEP = flow_len

#episode_count = 0

online_net = Network(myenv, obs, device=device)
online_net.load(load_path=LOAD_PATH)
#online_net.apply(init_weights)
online_net = online_net.to(device)

def test_time():
    global obs
    t1 = time.time()
    reward = 0
    step_count = 0
    for step in range(1, STEP+1):
        action = online_net.act(obs, 0, myenv, myenv.flows[step-1]["period"])[0]
        new_obs, rew, done = myenv.step(action, myenv.flows[step-1]["period"], myenv.flows[step]["period"])
        if step == flow_len - 3:
            rew, done = 10, True
        reward += rew
        if done:
            break
        obs = new_obs
        step_count += 1
    t2 = time.time()
    t = t2 - t1
    print("total time:", t, "s")
    with open("time.txt", "wt") as fw:
        fw.write("total time: "+str(t)+" s\n")
    print("step count:", step_count)

def resource_distribution():
    # 统计不同时间槽中帧的数量，计算占比最大的时间槽的方差
    slot_num = int(myenv.hyper / myenv.T)
    recorder = np.zeros(slot_num,) # 记录每个时间槽有几个帧
    reward = 0
    global obs
    for step in range(1, STEP+1):
        action = online_net.act(obs, 0, myenv, myenv.flows[step-1]["period"])[0]
        new_obs, rew, done = myenv.step(action, myenv.flows[step-1]["period"], myenv.flows[step]["period"])
        if step == flow_len - 3:
            rew, done = 10, True
        reward += rew
        if done:
            break
        obs = new_obs
        flow_count = int(myenv.hyper / myenv.flows[step-1]["period"]) # 流在一个周期内出现几次
        for i in range(flow_count):
            recorder[int(action+i*(myenv.flows[step-1]["period"]/myenv.T))] += 1
            #print(recorder)
    recorder = (recorder * FRAME_SIZE) / (myenv.capacity - myenv.preserv) # 计算每个时间槽的占比
    variance = 1 - np.std(recorder)
    print("Balance factor:", variance)
    with open("balance.txt", "wt") as fw:
        fw.write("balance factor:"+str(variance)+"\n")
    #print("recorder:", recorder)

    def save_file(filename):
        # save format csv
        L = []
        for i, j, z in zip(epLst, rewardLst, epLen):
            L.append([i, j, z])

        name=['rewards', 'flows']
        test=pd.DataFrame(columns=name,data=L) #数据有三列，列名分别为one,two,three
        test.to_csv(filename, mode="a", header=False, index=False)

if __name__ == "__main__":
    test_time()
    #resource_distribution()
    # 0.3的概率选
    #csvfile = open('ddqndata.csv',encoding='utf-8')
    #df = pd.read_csv(csvfile,engine='python')
    #plt.plot(df["episode"], df["reward"])
    #plt.show()

