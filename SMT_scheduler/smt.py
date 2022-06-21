import numpy as np 
from env import MulEnv, FRAME_SIZE, SWITCH_NUM, QUEUE_NUM
import z3 as z 
import time

if __name__ == "__main__":
    t1 = time.time()
    myenv = MulEnv()
    obs = myenv.reset()
    flow_nums = len(myenv.flows)
    slot_counts = int(myenv.hyper/myenv.T)

    """
              slot1  slot2  slot3  slot4
    flow_1      0      1      1      0         period: 800
    flow_2      1      0      0      1         period: 800
    flow_3      0      0      1      0         period: 1600
      ...
    flow_n      1      1      1      1         period: 400
    """
    solution_array = [[z.Int("alloc_[%s][%s]" % (i, j)) for j in range(slot_counts)] for i in range(flow_nums)]
    # for each value, either 0 or 1
    constraint1 = [z.Or(solution_array[i][j] == 0, solution_array[i][j] == 1) for j in range(slot_counts) for i in range(flow_nums)]
    # line summary constraint, i.e., the summary for each line should be equal to the total frame number of each flow in a scheduled cycle.
    # appearance frequency for each flow = sched_cycle / flow_period
    constraint2 = []
    for line in range(flow_nums):
        s = 0
        # line = flow_id
        frame_nums = myenv.hyper / myenv.flows[line]["period"]
        for column in range(slot_counts):
            s += solution_array[line][column]
        constraint2 += [s == frame_nums]
    # The time offset between every two frames belonging to the same flow must be equal to the period of the flow
    constraint3 = []
    for line in range(flow_nums):
        intervals = myenv.flows[line]["period"] // myenv.T
        frame_nums = myenv.hyper / myenv.flows[line]["period"]
        for sta in range(intervals):
            for j in range(int(frame_nums-1)):
                constraint3 += [solution_array[line][sta+j*intervals] == solution_array[line][sta+(j+1)*intervals]] 
    # the frame with the same next hop interface and the same sending slot could be aggregated to the same queue
    # the total size of all the frames that aggregated in the same queue should not be exceeded the upper limits for one slot capacity.
    constraint4 = []
    for column in range(slot_counts):
        s = 0
        for line in range(flow_nums):
            s += solution_array[line][column] * FRAME_SIZE
        constraint4 += [s <= myenv.capacity]
    
    res = z.Solver()
    res.add(constraint1)
    res.add(constraint2)
    res.add(constraint3)
    res.add(constraint4)
    if res.check() == z.sat:
        m = res.model()
        # with open("res.txt", "wt") as f:
        #     for i in range(flow_nums):
        #         for j in range(slot_counts):
        #             print("solution_array [%s][%s]:" % (i, j), m[solution_array[i][j]])
        #             f.write("solution_array ["+str(i)+"]["+str(j)+"]: "+str(m[solution_array[i][j]].as_long())+'\n')
        print("successfully scheduled flow number=", flow_nums)
        util = np.zeros((flow_nums, slot_counts))
        for i in range(flow_nums):
            for j in range(slot_counts):
                util[i, j] = int(str(m[solution_array[i][j]]))
        # Count the number of flows in each time interval
        slot_lst = []
        for j in range(slot_counts):
            s = 0
            for i in range(flow_nums):
                s += util[i, j]
            slot_lst.append(s)
        slot_lst = np.array(slot_lst)
        recorder = (slot_lst * FRAME_SIZE) / (myenv.capacity - myenv.preserv) # Compute the usage of each time interval
        variance = 1 - np.std(recorder)
        print("Balance factor:", variance)
    else:
        print("failed to solve")
    t2 = time.time()
    print("total_time=", t2-t1)




    
