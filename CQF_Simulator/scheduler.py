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
# SAVE_INTERVAL = 10000 # 每隔？存一次
# LOG_DIR = './logs/atari_double'

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') # m.weight?

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

# if __name__ == '__main__':
#     device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

#     make_env = lambda: Monitor(make_atari_deepmind('BreakoutNoFrameskip-v4', scale_values=True), allow_early_resets=True)

#     # vec_env = DummyVecEnv([make_env for _ in range(NUM_ENVS)])
#     vec_env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

#     env = BatchedPytorchFrameStack(vec_env, k=4)

#     replay_buffer = deque(maxlen=BUFFER_SIZE)
#     epinfos_buffer = deque([], maxlen=100) # info?

#     episode_count = 0

#     online_net = Network(env, device=device)
#     target_net = Network(env, device=device)

#     online_net.apply(init_weights)

#     online_net = online_net.to(device)
#     target_net = target_net.to(device)

#     target_net.load_state_dict(online_net.state_dict())

#     optimizer = torch.optim.Adam(online_net.parameters(), lr=LR)

#     # Initialize Replay Buffer
#     obses = env.reset()
#     for _ in range(MIN_REPLAY_SIZE):
#         actions = [env.action_space.sample() for _ in range(NUM_ENVS)]

#         new_obses, rews, dones, _ = env.step(actions)

#         for obs, action, rew, done, new_obs in zip(obses, actions, rews, dones, new_obses):
#             transition = (obs, action, rew, done, new_obs)
#             replay_buffer.append(transition)

#         obses = new_obses

#     # Main Training Loop
#     obses = env.reset()

#     for step in itertools.count():
#         epsilon = np.interp(step * NUM_ENVS, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END]) #(待插入值, [(0, epsilon_start),(epsilon_decay, epsilon_end)]),在两坐标之间连线，并在求出连线与x=插入值交点的纵坐标

#         if isinstance(obses[0], PytorchLazyFrames):
#             act_obses = np.stack([o.get_frames() for o in obses]) # why use stack?
#             actions = online_net.act(act_obses, epsilon)
#         else:
#             actions = online_net.act(obses, epsilon)

#         new_obses, rews, dones, infos = env.step(actions)

#         for obs, action, rew, done, new_obs, info in zip(obses, actions, rews, dones, new_obses, infos):
#             transition = (obs, action, rew, done, new_obs)
#             replay_buffer.append(transition)

#             if done:
#                 epinfos_buffer.append(info['episode'])
#                 episode_count += 1
#                 new_obses = env.reset()

#         obses = new_obses

#         # Start Gradient Step
#         transitions = random.sample(replay_buffer, BATCH_SIZE)
#         loss = online_net.compute_loss(transitions, target_net)

#         # Gradient Descent
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Update Target Network
#         if step % TARGET_UPDATE_FREQ == 0:
#             target_net.load_state_dict(online_net.state_dict())

#         # Logging
#         if step % LOG_INTERVAL == 0:
#             rew_mean = np.mean([e['r'] for e in epinfos_buffer]) or 0
#             len_mean = np.mean([e['l'] for e in epinfos_buffer]) or 0

#             print()
#             print('Step', step)
#             print('Avg Rew', rew_mean)
#             print('Avg Ep Len', len_mean)
#             print('Episodes', episode_count)

#         # Save
#         if step % SAVE_INTERVAL == 0 and step !=0:
#             print('Saving...')
#             online_net.save(SAVE_PATH)


