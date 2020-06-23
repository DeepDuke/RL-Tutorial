import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
import time
import random
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
from tabulate import tabulate

import gym


def preprocess(img):
    """ Convert RGB img into gray img and normalize it"""
    # Grayscale img
    gray_img = rgb2gray(img)
    # Normalize Pixel Values
    normalized_img = gray_img / 255.0
    # Resize img
    resized_img = transform.resize(normalized_img, [84, 84])
    return resized_img


def stacking_img(img, is_new_episode, stacked_num, stacked_imgs):
    """ Stacked four img"""
    preprocessed_img = preprocess(img)
    if is_new_episode:
        stacked_imgs.extend([preprocessed_img]*stacked_num)
        # Stacked the imgs to form a stacked state
    else:
        stacked_imgs.append(preprocessed_img)
    stacked_state = np.stack(stacked_imgs, axis=2)
    # Convert into tensor
    stacked_state = stacked_state.transpose((2, 0, 1))
    return stacked_state, stacked_imgs


class Memory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        index = np.random.choice(np.arange(self.buffer_size),
                                 size=batch_size,
                                 replace=False)
        return [self.buffer[i] for i in index]


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input size [None, 84, 84, 4]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # conv1 out: [None, 20, 20, 32]
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # conv2 out: [None, 9, 9, 64]
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # conv3 out: [None, 3, 3, 128]
        self.fc1 = nn.Linear(3*3*128, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        output = self.conv1(x)
        # print('image shape after conv1: ', output.shape)
        output = self.conv2(output)
        # print('image shape after conv2: ', output.shape)
        output = self.conv3(output)
        # print('image shape after conv3: ', output.shape)
        output = output.view(-1, 3*3*128)  # flatten
        # print('image shape after flatten: ', output.shape)
        output = self.fc1(output)
        # print('image shape after fc1: ', output.shape)
        output = self.fc2(output)
        # print('image shape after fc2: ', output.shape)
        return output


class DQNAgent:
    def __init__(self, action_space_size=6, buffer_size=10000, lr=0.05, gamma_=0.95, device_="cuda"):
        self.action_space_size = action_space_size
        self.device = device_
        self.lr = lr
        self.gamma = gamma_
        self.memory = Memory(buffer_size)
        self.eval_model = Net()
        self.target_model = Net()

        self.optimizer = optim.Adam(self.eval_net.parameters(), self.lr)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./summary')

    def choose_action(self, explore_prob, state):
        p = np.random.rand()
        if p < explore_prob:  # choose a random action
            action = random.choice(list(range(self.action_space_size)))
        else:  # choose the action with maximum Q value
            self.eval_model.eval()
            with torch.no_grad():
                state = state[np.newaxis, :]
                q_predict = self.eval_model(torch.from_numpy(state).to(self.device, dtype=torch.float))
                choice = torch.argmax(q_predict).item()
                action = choice
        return action

    def learn(self):

        pass


if __name__ == '__main__':
    # Hyper-parameters
    seed = 1
    render = False
    num_episodes = 2000
    env = gym.make('SpaceInvaders-v0')
    num_action = env.action_space.n
    torch.manual_seed(seed)
    env.seed(seed)

    stacked_num = 4
    stacked_imgs = deque([np.zeros((84, 84) for i in range(stacked_num))], maxlen=4)

    explore_prob_max = 1.0  # exploration probability at start
    explore_prob_min = 0.01  # minimum exploration probability
    decay_rate = 0.0001
    memory_buffer_size = 10000

    # hyper-parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda: 0
    learning_rate = 0.05
    gamma = 0.95
    batch_size = 64
    agent = DQNAgent(action_space_size=num_action, buffer_size=memory_buffer_size, lr=learning_rate,
                     gamma=gamma, device_=device)
    # pre-store a batch data
    for i in range(batch_size):
        print('Pre Memory step #{}'.format(i))
        if i == 0:
            state = env.reset()
            state, stacked_imgs = stacking_img(state, True, stacked_num, stacked_imgs)
        action = agent.choose_action(1.0, state)
        next_state, reward, done, info = env.step(action)
        if done:
            next_state = np.zeros(state.shape)
            agent.memory.add((state, action, reward, next_state, done))
            state = env.reset()
            # Stack the imgs
            state, stacked_frames = stacking_img(state, True, stacked_num, stacked_imgs)
        else:
            next_state, stacked_frames = stacking_img(next_state, False, stacked_num, stacked_imgs)
            # Add experience to memory
            agent.memory.add((state, action, reward, next_state, done))
            # Our state is now the next_state
            state = next_state

    # main loop
    for episode in range(num_episodes):
        



