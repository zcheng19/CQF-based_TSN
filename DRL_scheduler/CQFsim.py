import numpy as np 
from cus_flow import flow_template

BANDWITH = 10 ** (-3) * 1000 # kbit/us

PRESRV_BASE =  50 # 预留多少空间 kbit
SWITCH_NUM = 2 # 中间共有多少台交换机
MAX_T = True
MULTICQF = True
# 合法动作矩阵与业务流地址和拓扑有关
QUEUE_NUM = 3
FRAME_SIZE = 3 # kbit
ALPHA = 0.1 # 


class MulEnv:
    def __init__(self):
        self.create_graph()

    def create_graph(self):
        self.graph = {
            "h1": ["s1", ],
            "s1": ["h1", "s2"],
            "s2": ["s1", "h2"],
            "h2": ["s2", ],
            }

    def find_all_paths(self, start, end):
        "Find all the paths between any two nodes without loops."
        path = []
        stack = []
        stack.append(start)
        visited = set()
        visited.add(start)
        seen_path = {}
        while (len(stack) > 0):
            start = stack[-1]
            nodes = self.graph[start]
            if start not in seen_path.keys():
                seen_path[start] = []
            g = 0
            for w in nodes:
                if w not in visited and w not in seen_path[start]:
                    g = g+1
                    stack.append(w)  # save each available path temporarily
                    visited.add(w)
                    seen_path[start].append(w)
                    if w == end:
                        path.append(list(stack))
                        old_pop = stack.pop()
                        visited.remove(old_pop)
                    break
            if g == 0:
                old_pop = stack.pop()
                del seen_path[old_pop]
                visited.remove(old_pop)
        return path

    def get_flows(self):
        self.flows = flow_template()
        self.period_lst = []
        for dct in self.flows:
            self.period_lst.append(dct["period"])

    def get_gcd_periods(self):
        gcd_period = self.GCD(self.period_lst.copy())
        return gcd_period
    
    def GCD(self, Lst):
        while True:
            Lst.sort(reverse=True)
            if len(set(Lst)) == 1:
                break
            for i in range(len(Lst)-1):
                if Lst[i] % Lst[i+1] == 0:
                    Lst[i] = Lst[i+1]
                else:
                    Lst[i] = Lst[i] % Lst[i+1]
        return Lst[0]

    def get_hyper_period(self):
        periods = self.period_lst.copy()
        res = periods[0]
        for i in range(len(periods)-1):
            temp_num = res * periods[i+1] / self.division_method(res, periods[i+1])
            res = temp_num
        self.hyper = res

    def division_method(self, x, y):
        "division algorithm"
        while True:
            if x < y:
                x, y = y, x
            if x % y == 0:
                return int(y)
            else:
                temp = x % y
                x = y
                y = temp 

    def count_max_pathlen(self):
        # 找拓扑中最长路径的中间节点数-1
        pathLst = self.find_all_paths("h1", "h2")
        maxlen = len(pathLst[0])
        for lst in pathLst:
            length = len(lst)
            if length > maxlen:
                maxlen = length
        return maxlen-1
   
    def T_Q_mapping(self, t=10, c=QUEUE_NUM*SWITCH_NUM):
        # 记录每个节点上有几个接口, t代表行数(状态空间需要记录多少时间槽), c代表列数(中间节点的接收接口有几个)
        # self.int_num = {
        #     "h1": len(self.graph["h1"]),
        #     "s1": len(self.graph["s1"]),
        #     "s2": len(self.graph["s2"]),
        #     "h2": len(self.graph["h2"]),
        # }

        self.T_Q_array = np.zeros((t, c)) + self.preserv# 时间-队列状态矩阵

    def action_space(self, max_T=True, T=None): # 默认时间槽取最大值
        # 所有时间槽
        self.get_hyper_period() # 计算调度周期
        if max_T:
            self.T = self.get_gcd_periods()
            action_num = self.hyper / self.T
        else:
            if T is None:
                raise ValueError ("T must be set.")
            else:
                self.T = T
                action_num = self.hyper / self.T
        self.capacity = BANDWITH * self.T # 一个时间槽最大容量
        self.action_num = int(action_num)
        self.preserv = PRESRV_BASE * self.T / 800

    def reset(self): # flow_1st表示第一个业务流
        self.get_flows()
        self.action_space()
        maxNum = self.count_max_pathlen()-1 
        if MULTICQF:
            maxNum = (QUEUE_NUM-1) * maxNum - 1 + self.action_num
        self.T_Q_mapping(t=maxNum, c=SWITCH_NUM*QUEUE_NUM) #  构造 时间槽-队列 矩阵, 作为第一个通道
        
        # 构建 时间槽-接收队列 关系矩阵, 作为第二个通道
     #       交换就1(Q1 Q2 Q3)  交换就2(Q1 Q2 Q3)
     #  T0      0  0  1            0  0  1
     #  T1      1  0  0            1  0  0
     #  T2      0  1  0            0  1  0
     #  T3      0  0  1            0  0  1
     #  T4      1  0  0            1  0  0
        T_RXQ_array = np.zeros(self.T_Q_array.shape)
        self.T_TXQ_dct = {} # 记录时间槽与发送队列的映射关系
        self.T_RXQ_dct = {} # 记录时间槽与接收队列的映射关系
        for i in range(maxNum):
            rx = (i%QUEUE_NUM+2)%QUEUE_NUM # 计算在时间槽Ti, 对应的接收队列的索引
            self.T_TXQ_dct[i] = i%QUEUE_NUM
            self.T_RXQ_dct[i] = rx
            for j in range(SWITCH_NUM):
                T_RXQ_array[i][rx+j*QUEUE_NUM] = 1

        # 以下是每个业务流的合法动作矩阵, 作为第三个通道
     #     交换就1(Q1 Q2 Q3)  交换就2(Q1 Q2 Q3)
     # T0      1  1  1            1  1  1
     # T1      1  1  1            1  1  1
     # T2      0  0  0            0  0  0
     # T3      0  0  0            0  0  0
     # T4      0  0  0            1  0  0
        valid_action_array = np.zeros(self.T_Q_array.shape)
        period_1st = self.flows[0]["period"] # 第一个flow的周期
        valid_num = int(period_1st / self.T) # 计算对于即将到来的业务流合法的动作索引有哪些
        for i in range(valid_num):
            valid_action_array[i] = 1
        
        # 构建状态空间
        self.state = np.array([self.T_Q_array, T_RXQ_array, valid_action_array])
        wrapper_state = np.array([self.state[0]/self.capacity, T_RXQ_array, valid_action_array]) # C,H,W
        return np.array([wrapper_state])
    
    def step(self, slot, period, period_nth=None): # slot是时间槽索引, period是流周期, flow_nth是下一条流周期
        last_max_ratio = self.state[0].max()
        freq = int(self.hyper / period) # 计算一个调度周期中, 帧出现的频率
        for cnt in range(freq):
            slot = int(slot + cnt * (period / self.T))
            self.first_channel_transition(slot)

        # 第三通道状态转移
        valid_num = int(period_nth / self.T)
        self.state[2] = 0
        for t in range(valid_num):
            self.state[2][t] = 1

        # 奖励
        if self.check_done():
            reward = -10
            done = True
        else:
            reward = 1 - ((self.state[0].max() - last_max_ratio) / last_max_ratio * ALPHA)
            done = False

        wrapper_state = self.state[0]/self.capacity
        wrapper_state = np.array([[wrapper_state, self.state[1], self.state[2]]])

        return wrapper_state, reward, done

    def first_channel_transition(self, slot):
        # 第一通道状态转移
        cur_rx = self.T_RXQ_dct[slot]
        self.state[0][slot][cur_rx] += FRAME_SIZE
        sending_count = 0
        for i in range(slot+1, slot+(QUEUE_NUM-1)*SWITCH_NUM):
            if self.T_TXQ_dct[i] == cur_rx:
                sending_count += 1
                if sending_count >= SWITCH_NUM:
                    raise ValueError ("sending_count cannot be larger than SWITCH_NUM.")
                cur_rx = self.T_RXQ_dct[i]
                self.state[0][i][self.T_RXQ_dct[i]+sending_count*QUEUE_NUM] += FRAME_SIZE
            else:
                self.state[0][i][cur_rx+sending_count*QUEUE_NUM] += FRAME_SIZE # 存储一个时间槽队列的状态

    def check_done(self):
        if (self.state > self.capacity).any():
            return True
        else:
            return False

    def check_valid_action(self, act, cur_period):
        # 检查动作是否合法
        if act >= cur_period / self.T:
            return False
        else:
            return True



if __name__ == "__main__":
    env = MulEnv()
    env.create_graph()
    env.get_flows()
    env.reset(env.flows[0])
    print(env.step(1, 3200, 1600))
