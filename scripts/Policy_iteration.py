# Policy Iteration:
# 1. initiate a random policy
# 2. compute value function through Bellman equation based on policy
# 3. extract new policy based on value function
# Repeat step 2-3 until policy function converges
import gym
import numpy as np
from time import sleep


class PolicyIteration:
    def __init__(self, env):
        self.env = env
        self.gamma = 1.0
        self.threshold = 1e-10

    def compute_value_function(self, policy):
        value_table = np.zeros(self.env.observation_space.n)  # state nums
        while True:
            former_value_table = np.copy(value_table)
            for state in range(self.env.observation_space.n):
                action = policy[state]
                value_table[state] = 0
                for trans_prob, next_state, reward, _ in self.env.P[state][action]:
                    value_table[state] += trans_prob * (reward + self.gamma * former_value_table[next_state])
            if sum(np.fabs(value_table - former_value_table)) < self.threshold:
                break
        return value_table

    def extract_policy_function(self, value_table):
        policy = np.zeros(self.env.observation_space.n)
        for state in range(self.env.observation_space.n):
            Q_table = np.zeros(self.env.action_space.n)
            for action in range(self.env.action_space.n):
                for trans_prob, next_state, reward, _ in self.env.P[state][action]:
                    Q_table[action] += trans_prob * (reward + self.gamma * value_table[next_state])
            policy[state] = np.argmax(Q_table)
        return policy

    def run(self):
        policy = np.zeros(self.env.observation_space.n)
        while True:
            value_table = self.compute_value_function(policy)
            temp_policy = self.extract_policy_function(value_table)
            if sum(np.fabs(policy-temp_policy)) < self.threshold:
                break
            policy = np.copy(temp_policy)
        return policy


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = PolicyIteration(env)
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
