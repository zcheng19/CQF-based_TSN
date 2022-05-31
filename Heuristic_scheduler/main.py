from tabu import init_res, shifting, add_flows, exchanging
from CQFsim import MulEnv, FRAME_SIZE
from init import Network, init_weights
from torch import nn
import torch
import numpy as np
from collections import deque
import pandas as pd
import time
import os
import random

# 可调参数
ITERS = 10000
#ITERS = 3
STEPS = 50
EXCHANGE = True
DEQUELEN = 50
# 固定参数
filefullpath = "./tabudata.csv"


def save_csv(epLst, rewardLst, epLen):
    L = []
    for i, j, z in zip(epLst, rewardLst, epLen):
        L.append([i, j, z])

    name=['episodes','rewards', 'flows']
    test=pd.DataFrame(columns=name,data=L) #数据有三列，列名分别为one,two,three
    test.to_csv(filefullpath, header=False, index=False)

if os.path.exists(filefullpath):
    os.remove(filefullpath)
t1 = time.time()

myenv = MulEnv()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
obs = myenv.reset()

online_net = Network(myenv, obs, device=device)
target_net = Network(myenv, obs, device=device)
online_net.apply(init_weights)

online_net = online_net.to(device)
target_net = target_net.to(device)

target_net.load_state_dict(online_net.state_dict())

R = [] # global best rewards
Flow_num = [] # global flow number
Resource = [] # global resource util

reward, actLst = init_res(myenv, online_net) # 初始化结束
R.append(reward)
Flow_num.append(len(actLst))
print("init", "reward:", reward, "flows:", len(actLst))
cur_actLst = actLst
tabu_list = deque(maxlen=DEQUELEN)

for itera in range(ITERS):
    actionStepLst = [] # 记录循环中的所有操作
    actionRewardLst = [] # 记录每种操作所对应的reward
    step_count = 0
    for step in range(STEPS):
        if EXCHANGE:
            action_list, pop_flowsLst, rewd = exchanging(cur_actLst.copy(), myenv) # pop_flows=[[ID, slot],[ID, slot],...]
        reward, add_flowsLst = add_flows(myenv, action_list, online_net, obs)
        reward += rewd
        actionRewardLst.append(reward)
        actionStepLst.append([pop_flowsLst, add_flowsLst]) # 两个actionLst索引对应
    
    actionRewardNdarray = np.array(actionRewardLst)
    if random.random() <= 0.2:
        for i in range(len(actionRewardLst)):
            max_index = np.argmax(actionRewardNdarray)
            if actionStepLst[max_index] not in tabu_list:
                for pop_lst in actionStepLst[max_index][0]:
                    cur_actLst.pop(cur_actLst.index(pop_lst))
                for add_lst in actionStepLst[max_index][1]:
                    cur_actLst.append(add_lst)
                tabu_list.append(actionStepLst[max_index])
                break
            else:
                actionRewardNdarray[max_index] = -np.inf
        else:
            # 所有元素都在禁忌表中
            break
    else:
        max_index = random.sample(range(len(actionRewardNdarray)), 1)[0]
        for pop_lst in actionStepLst[max_index][0]:
            cur_actLst.pop(cur_actLst.index(pop_lst))
        for add_lst in actionStepLst[max_index][1]:
            cur_actLst.append(add_lst)
    R.append(actionRewardNdarray[max_index])
    Flow_num.append(len(cur_actLst))

    step_count += 1
    print("iteration:", itera, "max reward:", actionRewardNdarray[max_index], "flows:", len(cur_actLst))
  
    # 统计每个时间槽资源占比
    slot_num = myenv.hyper / myenv.T
    recorder = np.zeros(int(slot_num,))
    for action in cur_actLst:
        freq = myenv.hyper / myenv.flows[action[0]]["period"]
        for cnt in range(int(freq)):
            recorder[int(action[1]+cnt*(myenv.flows[action[0]]["period"]/myenv.T))] += 1
    # 计算均衡率
    recorder = (recorder * FRAME_SIZE) / (myenv.capacity - myenv.preserv)
    variance = 1 - np.std(recorder)
    Resource.append(variance)

mi = np.argmax(np.array(R))
print("max reward:", max(R))
print("flow num:", Flow_num[mi])
save_csv([i for i in range(len(R))], R, Flow_num)
t2 = time.time()
total_time = t2 - t1
print("time:", total_time, "s")
with open("time.txt", "wt") as fw:
    fw.write("total time: "+str(total_time)+"\n")

max_resouce = max(Resource)
print("Max Balance factor:", max_resouce)
with open("balance.txt", "wt") as fw:
    fw.write("max balance factor:"+str(max_resouce))
