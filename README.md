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

first_channel_transition: the network resource state (i.e., the first channel of the observation) transition. 

check_done: check the training status.

check_valid_action: check whether the action taken for each flow is valid.

### Variables in main.py
BATCH_SIZE: the batch size for sampling the trajectories during the training process.

MIN_REPLAY_SIZE: the minimum threshold of trajectories amount for starting the training. 

EPSILON_START: the start value of epsilon used to choose the action.

EPSILON_END: the end value of epsilon used to choose the action.

EPSILON_DECAY: the decay of the value of epsilon with the increasing of episodes.

EPISODES: the total number of episodes for training.

LR: the learning rate during training.

TARGET_UPDATE_FREQ: the update frequency of training.

myenv: the instance of CQF-based TSN system.

device: the device (i.e., CPU, GPU) used for training the DRL algorithm.

obs: the observations at each step.

flow_len: the total number of flows.

STEP: the total number of step within each episode.

replay_buffer: the instance of replay buffer.

episode_count: count how many flows are scheduled in each episode.

online_net: the instance of the evaluate neural network.

target_net: the instance of the target neural network.

optimizer: the optimizer of DRL algorithm.

reward_lst: a list to store the accumulated rewards for each episode.

ep_lst: a list to store each episode label.

ep_len: count how many episodes are continued until now.

step_count: count how many steps are passed until now.

epsilon: the epsilon parameter used for epsilon-greedy algorithm to choose the actions.

new_obs: the transited observation of the old one.

transition: the state transition after the actions are taken.

loss: the loss between evaluated result and taget one.

### Functions in main.py
myenv.reset: reset the environment at each episode.

online_net.apply: apply the initialized parameters of neural network.

target_net.load_state_dict: load the parameters for target network.

torch.optim.Adam: use Adam optimizer to update the neural network parameters.

online_net.act: choose an action by the DRL agent according to the Q-values computed by the neural network.

myenv.step: take the actions, and the do the MDP.

replay_buffer.append: add the trajectories into replay buffer.

np.interp: use the linear interpolation method to update the epsilon.

random.sample: randomly sample the trajectories.

online_net.compute_loss: compute the loss between the evaluate and target values by using the loss function.

optimizer.zero_grad: clear the historial gradient.

loss.backward: back forward the gradient.

optimizer.step: update the network parameters based on the defined learning rate.

online_net.save: save the updated parameters of neural networks.

### Variables in cus_flow.py
flow_features: a list for all the flows, where the features are defined.

period: the period of each flow.

latency: the latency requirement customized by users.

jitter: the latnecy variation of transmitting flows.

src: the source nodes of each flow.

dst: the destination nodes of each flow.

### Functions in cus_flow.py
flow_template: build all the flows by using the flow template.

## User customized experiment part
