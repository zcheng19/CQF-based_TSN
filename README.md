# CQF-based TSN Simulator
The CQF-based TSN simulator supports a general scheduling environment of the CQF or enhanced Multi-CQF model for Time-Sensitive Networks (TSN). Several packed schedulers can be used to do the test.

# Install the necessary modules
Experimental Environment: Python version: 3.8.10. Ubuntu version: 20.04.1, 5.13.0-30-generic kernel. 

Run "pip3 install -r requirements.txt" to install the necessary modules.

# Run the test
Run "python3 main.py" under different folders to start solving the NP-hard scheduling problem by utilizing the corresponding schedulers.

Run "python3 play.py checkpoint" to test the performance of model.

The CQF-based TSN simulator can support multiple cyclic queues from 2 to K, where K is an integer.

# Parameters description
## Variables and functions
### Variables in CQFsim.py
BANDWIDTH: the link bandwidth of TSN.

PRESRV: the preserved bandwidth for bursting flows.

SWITCH_NUM: the number of switches in the network.

MAX_T: the maximum time interval value of CQF.

MULTICQF: decide whether to choose the CQF model with mutiple queues.

QUEUE_NUM: the specific number of queues of the multi-CQF.

FRAME_SIZE: the size of each frame.

ALPHA: a coefficient for adjusting the weight of the load balance in calculating the reward.

self.graph: define a network graph.

path: a list to store all of the available paths from the source node to destination one with no loop.

stack: save each available path temporarily. 

visited: store the visited nodes.

seen_path: store the searched paths.

self.flows: a list to store all of the defined flows. 

self.period_lst: a list to store all of flow periods.

gcd_period: get the greatest common divisor (GCD) of the flow periods.

self.hyper: get the hyper period for all of the flows to schedule.

self.T_Q_array: the two-dimensional time-resource matrix for observation.

self.T: the time interval length.

self.capacity: the capacity of each time interval.

self.action_num: the number of actions that each flow can take.

T_RXQ_array: the shape of the time-resource matrix.

self.T_TXQ_dct: the mapping relationship between the time intervals and sending queues.

self.T_RXQ_dct: the mapping relationship between the time intervals and receiving queues.

valid_action_array: the encoded valid action array for each flow.

valid_num: the valid number of actions for each flow.

pad_valid_action_array: zero paddings for the valid action array.

self.state: the original state for observation.

wrapper_state: the processed state as the input information for DRL algorithm. The processing includes extending the dimension of state array, and zero paddings.

freq: count the frequency a flow can appear in a hyper period.

reward: the immediate reward the environment feeds back after the action is taken.

done: the training status reflected by the environment. 

### Functions in CQFsim.py
create_graph: build a TSN graph by using adjacent table.

find_all_paths: find all the paths from the source node to the destination with no loops.

get_flows: get all the customized flows.

get_gcd_periods: calculate the greatest common divisor of all the flow periods.

GCD: the algorithm for computing the greatest common divisor.

get_hyper_period: calculate the hyper period for flow scheduling.

division_method: the division algorithm used for getting the hyper period.

count_max_pathlen: compute the number of hops along the flow transmission path.

T_Q_mapping: build the time-resource matrix as the observation.

action_space: get the action space of DRL.

reset: the state resets at each episode.

step: run the MDP of training process of DRL.


## User customized experiment part
