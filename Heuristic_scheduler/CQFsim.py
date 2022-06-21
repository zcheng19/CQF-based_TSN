import numpy as np 
from cus_flow import flow_template


BANDWIDTH = 10 ** (-3) * 1000 # kbit/us

PRESRV_BASE = 50 # reserved space (kbit)
SWITCH_NUM = 2 # number of intermidiate switches
MAX_T = True
MULTICQF = True

QUEUE_NUM = 3 # cyclic queue number
FRAME_SIZE = 3 # kbit
ALPHA = 0.1 


class MulEnv:
    def __init__(self):
        self.create_graph()

    def create_graph(self):
        # An example of graph
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
        # Count the number of intermidiate switches along the largest path
        pathLst = self.find_all_paths("h1", "h2")
        maxlen = len(pathLst[0])
        for lst in pathLst:
            length = len(lst)
            if length > maxlen:
                maxlen = length
        return maxlen-1
   
    def T_Q_mapping(self, t, c):
        self.T_Q_array = np.zeros((t*2+(t-1)*2, c*4+(c-1)*4)) + self.preserv # time-queue array

    def action_space(self, max_T=True, T=None): 
        self.get_hyper_period() # Compute the hyper period
        if max_T:
            self.T = self.get_gcd_periods()
            action_num = self.hyper / self.T
        else:
            if T is None:
                raise ValueError ("T must be set.")
            else:
                self.T = T
                action_num = self.hyper / self.T
        self.capacity = BANDWIDTH * self.T # maximum value of time interval capacity
        self.action_num = int(action_num)
        self.preserv = PRESRV_BASE * self.T / 800

    def reset(self): # flow_1st represents the first flow
        self.get_flows()
        self.action_space()
        maxNum = self.count_max_pathlen()-1 
        if MULTICQF:
            maxNum = (QUEUE_NUM-1) * maxNum - 1 + self.action_num
        self.T_Q_mapping(t=maxNum, c=SWITCH_NUM*QUEUE_NUM) #  create the slot-queue array as the first channel of state 
        
        # create the slot-reception array as the second channel of state
     #       switch1(Q1 Q2 Q3)  switch2(Q1 Q2 Q3)
     #  T0      0  0  1            0  0  1
     #  T1      1  0  0            1  0  0
     #  T2      0  1  0            0  1  0
     #  T3      0  0  1            0  0  1
     #  T4      1  0  0            1  0  0
        T_RXQ_array = np.zeros(self.T_Q_array.shape)
        self.T_TXQ_dct = {} # The mapping relationship between slot and sending queue 
        self.T_RXQ_dct = {} # The mapping relationship between slot and receiving queue
        for i in range(maxNum):
            rx = (i%QUEUE_NUM+2)%QUEUE_NUM # Calculate the index of receiving queue at slot T_i
            self.T_TXQ_dct[i] = i%QUEUE_NUM
            self.T_RXQ_dct[i] = rx
            for j in range(SWITCH_NUM):
                T_RXQ_array[i*4: i*4+2, rx*8+j*QUEUE_NUM*8: rx*8+j*QUEUE_NUM*8+4] = [[1, 1, 1, 1], [1, 1, 1, 1]]

        # The array of valid actions of each flow as the third channel of state
     #     switch1(Q1 Q2 Q3)  switch2(Q1 Q2 Q3)
     # T0      1  1  1            1  1  1
     # T1      1  1  1            1  1  1
     # T2      0  0  0            0  0  0
     # T3      0  0  0            0  0  0
     # T4      0  0  0            1  0  0
        valid_action_array = np.zeros(self.T_Q_array.shape)
        period_1st = self.flows[0]["period"] # The first flow period
        valid_num = int(period_1st / self.T) # Compute all the valid action indices for the coming flows
        for i in range(valid_num):
            valid_action_array[i*4: i*4+2] = 1

        pad_T_RXQ_array = np.pad(T_RXQ_array, ((1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        pad_valid_action_array = np.pad(valid_action_array, ((1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        
        # Create the state space
        self.state = np.array([self.T_Q_array, T_RXQ_array, valid_action_array]) # C,H,W
        wrapper_state = np.pad(self.T_Q_array/self.capacity, ((1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        wrapper_state = np.array([[wrapper_state, pad_T_RXQ_array, pad_valid_action_array]])
        return wrapper_state
    
    def step(self, slot, period, period_nth=None): # slot is the index of time interval, period is flow period, flow_nth is the next flow
        last_max_ratio = self.state[0].max()
        freq = int(self.hyper / period) # Compute the apperance frequency of the flow in a hyper period
        for cnt in range(freq):
            slot = int(slot + cnt * (period / self.T))
            self.first_channel_transition(slot)

        # state transition of the third channel
        valid_num = int(period_nth / self.T)
        self.state[2] = 0
        for t in range(valid_num):
            self.state[2][t*4: t*4+2] = 1

        # reward
        if self.check_done():
            reward = -10
            done = True
        else:
            reward = 1 - ((self.state[0].max() - last_max_ratio) / last_max_ratio * ALPHA)
            done = False

        pad_state0 = np.pad(self.state[0]/self.capacity, ((1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        pad_state1 = np.pad(self.state[1], ((1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        pad_state2 = np.pad(self.state[2], ((1, 1), (0, 0)), 'constant', constant_values=(0, 0))
        wrapper_state = np.array([[pad_state0, pad_state1, pad_state2]])

        return wrapper_state, reward, done

    def first_channel_transition(self, slot):
        # state transition of the first channel
        # print(slot)
        cur_rx = self.T_RXQ_dct[slot]
        self.state[0][slot*4: slot*4+2, cur_rx*8: cur_rx*8+4] += FRAME_SIZE
        sending_count = 0
        for i in range(slot+1, slot+(QUEUE_NUM-1)*SWITCH_NUM):
            if self.T_TXQ_dct[i] == cur_rx:
                sending_count += 1
                if sending_count >= SWITCH_NUM:
                    raise ValueError ("sending_count cannot be larger than SWITCH_NUM.")
                cur_rx = self.T_RXQ_dct[i]
                self.state[0][i*4: i*4+2, self.T_RXQ_dct[i]*8+sending_count*QUEUE_NUM*8: self.T_RXQ_dct[i]*8+sending_count*QUEUE_NUM*8+4] += FRAME_SIZE
            else:
                self.state[0][i*4: i*4+2, cur_rx*8+sending_count*QUEUE_NUM*8: cur_rx*8+sending_count*QUEUE_NUM*8+4] += FRAME_SIZE # store the slot-queue state

    def check_done(self):
        if (self.state > self.capacity).any():
            return True
        else:
            return False

    def check_valid_action(self, act, cur_period):
        # check whether the action is valid
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
