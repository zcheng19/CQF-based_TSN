import os
from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random
import math
# import msgpack
# from msgpack_numpy import patch as msgpack_numpy_patch
# msgpack_numpy_patch()


GAMMA=0.99
BATCH_SIZE=32

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # m.weight?


def nature_cnn(observation_space):
    n_input_channels = observation_space.shape[1] # C,H,W
    #print(observation_space.shape)

    cnn = nn.Sequential(
        nn.Flatten(),)

    # Compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(np.zeros((observation_space.shape[0],
        observation_space.shape[1],observation_space.shape[2],
        observation_space.shape[3]))).float()).shape[1] # 可以用形状相同的0矩阵试探

    #final_layer = 2**(int(math.log(n_flatten, 2))+1)
    out = nn.Sequential(cnn,)

    return out, n_flatten

class Network(nn.Module):
    def __init__(self, env, obs, device, double=True):
        super().__init__()

        self.num_actions = env.action_num
        self.device = device
        self.double = double

        conv_net, flatten = nature_cnn(obs)

        self.net = nn.Sequential(conv_net, nn.Linear(flatten, self.num_actions))

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon, env, period):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)
        
        rnd_sample = random.random()
        while True:
            max_q_indices = torch.argmax(q_values, dim=1)
            actions = max_q_indices.detach().tolist()

            if rnd_sample <= epsilon:
                actions[0] = random.randint(0, self.num_actions - 1)

            valid = env.check_valid_action(actions[0], period)
            if not valid: # 如果不合法，则重新选择动作
                q_values[0][actions[0]] = -np.inf
            else:
                break
        return actions

    def compute_loss(self, transitions, target_net):
        obses = np.asarray([t[0] for t in transitions])
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = np.asarray([t[4] for t in transitions])

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        # Compute Targets
        with torch.no_grad():
            if self.double:
                targets_online_q_values = self(new_obses_t)
                targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)
                targets_target_q_values = target_net(new_obses_t)
                targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)
                targets = rews_t + GAMMA * (1 - dones_t) * targets_selected_q_values
            else:    
                target_q_values = target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.mse_loss(action_q_values, targets)

        return loss

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()}

        self.load_state_dict(params)


