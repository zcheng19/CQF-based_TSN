import random

def init_res(env, online_net):
    # 利用epsilon贪婪策略生成初始解
    obs = env.reset()
    flow_len = len(env.flows)
    STEP = flow_len
    reward = 0
    action_record_lst = [] # 记录了每个流插在哪个时间槽 

    for step in range(1, STEP+1):
        action = online_net.act(obs, 0.2, env, env.flows[step-1]["period"])[0]
        new_obs, rew, done = env.step(action, env.flows[step-1]["period"], env.flows[step]["period"])
        if step == flow_len - 3:
            rew, done = 10, True
        reward += rew
        if done:
            break
        action_record_lst.append([step-1, action]) # 流ID, 时间槽
        obs = new_obs
    return reward, action_record_lst

def shifting(env, action_lst): # 该参数记录了每个流插在哪个时间槽
    # 交换调度成功业务流，改变时间槽的分布
    L = [] # 记录交换了哪些流
    num = random.randint(0, 1)
    for i in range(num):
        index = random.randint(0, len(action_lst)-1)
        flow_id = action_lst[index][0]
        valid_slot_num = env.flows[flow_id]["period"] / env.T
        action_lst[index][1] = random.randint(0, valid_slot_num-1)
    while True:
        reward = 0
        for j in range(len(action_lst)):
            new_obs, rew, done = env.step(action_lst[j][1], env.flows[action_lst[j][0]]["period"], env.flows[action_lst[j][0]]["period"])
            reward += rew
            if done:
                action_lst.pop(j)
                env.reset()
                break
        else:
            return reward, action_lst, new_obs

def add_flows(env, actionLst, online_net, obs): # 这里的actionLst是经过exchange剔除后的
    # 随机添加新的业务流
    L = []
    for a in actionLst:
        L.append(a[0])
    add_flow_lst = []
    reward = 0
    for j in range(len(env.flows)):
        if j not in L:
            action = online_net.act(obs, 0.1, env, env.flows[j]["period"])[0]
            new_obs, rew, done = env.step(action, env.flows[j]["period"], env.flows[j]["period"])
            if (len(actionLst) + len(add_flow_lst)) >= len(env.flows)-3:
                rew, done = 10, True
            reward += rew
            if done:
                break
            add_flow_lst.append([j, action])
    return reward, add_flow_lst

def exchanging(act_lst, env): # 这里的act_lst是shifting之后的
    # 随机剔除调度成功的业务流
    reward = 0
    env.reset()
    num = random.randint(0, 1)
    L = [] # 存储所有剔除的流
    for i in range(num):
        if not act_lst:
            break
        act = act_lst.pop(random.randint(0, len(act_lst)-1))
        L.append(act) # [流ID, 时间槽]
    for j in range(len(act_lst)):
        new_obs, rew, done = env.step(act_lst[j][1], env.flows[act_lst[j][0]]["period"], env.flows[act_lst[j][0]]["period"])
        if done:
            raise ValueError ("exchange done error.")
        reward += rew
    return act_lst, L, reward
