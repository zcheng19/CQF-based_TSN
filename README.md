# CQF-based TSN Simulator
The CQF-based TSN simulator supports a general scheduling environment of the CQF or Multi-CQF model for Time-Sensitive Networks (TSN). Several packed schedulers can be used to do the test.

Run "pip3 install -r requirements.txt" to install the necessary modules.

Experimental Environment: Python version: 3.8.10. Ubuntu version: 20.04.1, 5.13.0-30-generic kernel. 

Run "python3 main.py" under different folders to start solving the NP-hard scheduling problem by utilizing the corresponding schedulers.

Run "python3 play.py checkpoint" to test the performance of model.

The CQF-based TSN simulator can support multiple cyclic queues from 2 to K, where K is an integer.
