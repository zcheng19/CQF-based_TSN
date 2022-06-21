import os
from torch import nn
import torch
from collections import deque
import itertools
import numpy as np
import random
from torch.nn import functional as F
import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()


GAMMA=0.99
BATCH_SIZE=32
# SAVE_PATH = './atari_model_double.pack'
# SAVE_INTERVAL = 10000 
# LOG_DIR = './logs/atari_double'

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') 

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    #for i in range(num_residuals):
    #if i == 0 and not first_block:
    if num_residuals==0 and not first_block:
        blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
    else:
        blk.append(Residual(num_channels, num_channels))
    return blk

def res_net(observation_space, final_layer=512):
    n_input_channels = observation_space.shape[1] # C,H,W
    b2 = nn.Sequential(*resnet_block(n_input_channels, 64, 0))
    b21=nn.Sequential(*resnet_block(64, 64, 1))
    b3 = nn.Sequential(*resnet_block(64, 128, 0))
    b31=nn.Sequential(*resnet_block(128, 128, 1))
    b4 = nn.Sequential(*resnet_block(128, 256, 0))
    b41=nn.Sequential(*resnet_block(256, 256, 1))
    b5 = nn.Sequential(*resnet_block(256, 512, 0))
    b51=nn.Sequential(*resnet_block(512, 512, 1))

    resnet = nn.Sequential(b2,b21,b3,b31,b4,b41,b5,b51,nn.Flatten(),)

    with torch.no_grad():
        # print(observation_space.shape)
        n_flatten = resnet(torch.as_tensor(np.zeros((observation_space.shape[0], observation_space.shape[1], observation_space.shape[2], observation_space.shape[3]))).float()).shape[1]
        # x = torch.as_tensor(np.zeros((observation_space.shape[0], observation_space.shape[1], observation_space.shape[2], observation_space.shape[3]))).float()
        # for layer in resnet:
        #     x = layer(x)
        #     print(x.shape)
        # raise
    return resnet, n_flatten

class Network(nn.Module):
    def __init__(self, env, obs, device, double=True):
        super().__init__()

        self.num_actions = env.action_num
        self.device = device
        self.double = double

        conv_net, n_flatten = res_net(obs)

        self.net = nn.Sequential(conv_net, nn.Linear(n_flatten, self.num_actions))

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
            if not valid: # If the action is invalid, find another one
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

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

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


