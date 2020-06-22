import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import numpy as np
from vizdoom import *  # Game Doom environment
import time
import random
from skimage import transform
from collections import deque
from matplotlib import pyplot as plt
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')


def create_environment():
    game = DoomGame()
    # Load the correct configuration
    game.load_config("basic.cfg")

    # Load the correct scenario (in our case basic scenario)
    game.set_doom_scenario_path("basic.wad")

    game.init()

    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    possible_actions = [left, right, shoot]

    return game, possible_actions


def test_environment():
    game = DoomGame()
    game.load_config("basic.cfg")
    game.set_doom_scenario_path("basic.wad")
    game.init()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]
    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            misc = state.game_variables
            action = random.choice(actions)
            print(action)
            reward = game.make_action(action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)
    game.close()


# Init Game
game, possible_actions = create_environment()

stack_size = 4  # We stack 4 frames
# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

state_size = [84, 84, 4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = game.get_available_buttons_size()              # 3 possible actions: left, right, shoot
learning_rate = 0.0002      # Alpha (aka learning rate)

# Training parameters
total_episodes = 10000       # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning parameters
gamma = 0.95               # Discounting rate

# Memory parameters
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True

# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False

# Summary writer
writer = SummaryWriter('./summary')


def preprocess_frame(frame):
    # Greyscale frame already done in our vizdoom config
    # x = np.mean(frame,-1)

    # Crop the screen (remove the roof because it contains no information)
    cropped_frame = frame[30:-10, 30:-30]

    # Normalize Pixel Values
    normalized_frame = cropped_frame / 255.0

    # Resize
    preprocessed_frame = transform.resize(normalized_frame, [84, 84])

    return preprocessed_frame


def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)

        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2)
    stacked_state = stacked_state.transpose((2, 0, 1))

    return stacked_state, stacked_frames


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Input size [None, 84, 84, 4]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        # conv1 out: [None, 20, 20, 32]
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        # conv2 out: [None, 9, 9, 64]
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ELU()
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


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


# Instantiate memory
memory = Memory(max_size=memory_size)
# Render the environment
game.new_episode()

for i in range(pretrain_length):
    # If it's the first step
    print('Pre Memory step #{}'.format(i))
    if i == 0:
        # First we need a state
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    # Random action
    action = random.choice(possible_actions)

    # Get the rewards
    reward = game.make_action(action)

    # Look if the episode is finished
    done = game.is_episode_finished()

    # If we're dead
    if done:
        # We finished the episode
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        game.new_episode()

        # First we need a state
        state = game.get_state().screen_buffer

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:
        # Get the next state
        next_state = game.get_state().screen_buffer
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Our state is now the next_state
        state = next_state


if __name__ == '__main__':
    torch.manual_seed(0)  # set random seed
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda: 0

    game.init()
    decay_step = 0
    # Initiate training settings
    model = DQN().to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Starting Training ...')
    for episode in range(total_episodes):
        step = 0
        episode_rewards = []
        # Make a new episode and observe the first state
        game.new_episode()
        state = game.get_state().screen_buffer
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        print('Running Episode #{}'.format(episode))
        while step < max_steps:
            # print('Episode #{} --- step #{}'.format(episode, step))
            # INTERACTION PART
            step += 1
            decay_step += 1
            # Choose action by decayed epsilon-greedy method
            trade_off = np.random.rand()
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
            if explore_probability < trade_off:  # choose a random action to explore
                action = random.choice(possible_actions)
            else:  # choose a highest q value action to exploit
                model.eval()
                with torch.no_grad():
                    state = state[np.newaxis, :]
                    # print('state shape:', state.shape)
                    Q_predict = model(torch.from_numpy(state).to(device, dtype=torch.float))  # convert numpy array to torch tensor
                    choice = torch.argmax(Q_predict).item()  # convert torch tensor back to numpy
                    action = possible_actions[int(choice)]

            # Do the action
            reward = game.make_action(action)

            # Look if the episode is finished
            done = game.is_episode_finished()

            # Add the reward to total reward
            episode_rewards.append(reward)
            # If the game is finished
            if done:
                # the episode ends so no next state
                next_state = np.zeros((84, 84), dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Set step = max_steps to end the episode
                step = max_steps

                # Get the total reward of the episode
                # total_reward = np.sum(episode_rewards)
                # print("Episode #{} Rewards: --- {:.4f}".format(episode, total_reward))
                memory.add((state, action, reward, next_state, done))
            else:
                # Get the next state
                next_state = game.get_state().screen_buffer

                # Stack the frame of the next_state
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))

                # st+1 is now our current state
                state = next_state

            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states_mb = np.array([each[0] for each in batch], ndmin=3)
            actions_mb = np.array([each[1] for each in batch])
            rewards_mb = np.array([each[2] for each in batch])
            next_states_mb = np.array([each[3] for each in batch], ndmin=3)
            dones_mb = np.array([each[4] for each in batch])
            # Compute Q target
            target_Qs_batch = []
            # model.eval()
            # with torch.no_grad():

            # LEARNING PART
            model.train()
            optimizer.zero_grad()

            Qs_next_state = model(torch.from_numpy(next_states_mb).to(device, dtype=torch.float))

            for i in range(0, len(batch)):
                terminal = dones_mb[i]
                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(torch.from_numpy(np.array([rewards_mb[i]])).to(device, dtype=torch.float).item())
                else:
                    target = torch.from_numpy(np.array([rewards_mb[i]])).to(device, dtype=torch.float) + \
                             gamma * torch.max(Qs_next_state[i])
                    target_Qs_batch.append(target.item())
            # print('target_Qs_batch:', target_Qs_batch)
            targets_mb = np.array(target_Qs_batch)
            # Compute Q estimate
            estimates_Qs_batch = []
            Qs_state = model(torch.from_numpy(next_states_mb).to(device, dtype=torch.float))
            for i in range(0, len(batch)):
                output = torch.matmul(Qs_state[i], torch.from_numpy(actions_mb[i]).to(device, dtype=torch.float))
                estimates_Qs_batch.append(output.item())
            estimates_mb = np.array([each for each in estimates_Qs_batch])
            # Compute loss
            # print('estimates_mb:', estimates_mb.dtype)
            # print('targets_mb:', targets_mb.dtype)
            loss = criterion(torch.from_numpy(estimates_mb).to(device, dtype=torch.float),
                             torch.from_numpy(targets_mb).to(device, dtype=torch.float).requires_grad_())
            # Optimize
            loss.backward()
            optimizer.step()
            # Summary writer
            if done or (step == max_steps-1):
                writer.add_scalar('Show/Explore_probability', explore_probability, episode)
                writer.add_scalar('Show/Loss', loss.item(), episode)
                writer.add_scalar('Show/RewardsPerEpisode', np.sum(episode_rewards), episode)
                table = [['Episode', 'Explore_probability', 'Loss', 'Rewards'],
                         [episode, explore_probability, loss.item(), np.sum(episode_rewards)]]
                # print('-' * 30)
                print(tabulate(table, headers='firstrow', tablefmt='grid'))
                # print('-' * 30)
                # print("train loss: --- {:.4f}".format(loss.item()))
            # Save model every  5 episode
            if episode % 5 == 0:
                saved_model_name = os.path.join('./model/model' + '_' + str(episode) + '.pt')
                torch.save(model.state_dict(), saved_model_name)


# TODO: To add testing code
# TODO: To add argparse
# TODO: To prettify the code




