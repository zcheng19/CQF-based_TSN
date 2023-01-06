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
## Variables
### Variables in CQFsim.py
BANDWIDTH: the link bandwidth of TSN.

PRESRV: the preserved bandwidth for bursting flows.

SWITCH_NUM: the number of switches in the network.

MAX_T: the maximum time interval value of CQF.

MULTICQF: Decide whether to choose the CQF model with mutiple queues.

QUEUE_NUM: The specific number of queues of the multi-CQF.

FRAME_SIZE: The size of each frame.

ALPHA: A coefficient for adjusting the weight of the load balance in calculating the reward.

self.graph: define a network graph.

path: A list to store all of the available paths from the source node to destination one with no loop.

stack: save each available path temporarily. 

visited: store the visited nodes.

seen_path: store the searched paths.


## User customized experiment part
