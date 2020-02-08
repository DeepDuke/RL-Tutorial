# Value Iteration through optimal Bellman equation

import gym
import numpy as np
from time import sleep


class ValueIteration:
    def __init__(self, env):
        self.env = env
        self.gamma = 1.0
        self.threshold = 1e-3

    def compute_value_function(self):
        value_table = np.zeros(self.env.observation_space.n)
        cnt = 0
        while True:
            last_value_table = np.copy(value_table)
            for state in range(self.env.observation_space.n):
                q_table = np.zeros(self.env.action_space.n)
                for action in range(self.env.action_space.n):
                    for trans_prob, next_state, reward, _ in self.env.P[state][action]:
                        q_table[action] += trans_prob * (reward + self.gamma * last_value_table[next_state])
                value_table[state] = np.max(q_table)
            if sum(np.fabs(value_table-last_value_table)) < self.threshold:
                break
            print('Iteration #{}'.format(cnt))
            print(value_table)
            cnt += 1
        return value_table

    def extract_policy(self, value_table):
        policy = np.zeros(self.env.observation_space.n)
        while True:
            last_policy = np.copy(policy)
            for state in range(self.env.observation_space.n):
                q_table = np.zeros(self.env.action_space.n)
                for action in range(self.env.action_space.n):
                    for trans_prob, next_state, reward, _ in self.env.P[state][action]:
                        q_table[action] += trans_prob * (reward + self.gamma * value_table[next_state])
                policy[state] = np.argmax(q_table)
            if sum(np.fabs(policy-last_policy)) < self.threshold:
                break
        return policy

    def run(self):
        value_table = self.compute_value_function()
        return self.extract_policy(value_table)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = ValueIteration(env)
    optimal_policy = agent.run()
    print('optimal policy:', optimal_policy)

    observation = env.reset()
    for i in range(1000):
        print('Iteration #{}'.format(i))
        env.render()
        action = int(optimal_policy[observation])  # your agent here (this takes random actions)
        observation, reward, done, info = env.step(action)
        sleep(1)
        if done:
            print('Game Finished!')
            observation = env.reset()
            break

    env.close()