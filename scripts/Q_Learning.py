import gym
import numpy as np


class QLearning:
    def __init__(self, action_space_n, observation_space_n, learning_rate=0.80, gamma=0.95,
                 max_epsilon=1.0, min_epsilon=0.01, decay=0.010):
        self.action_space = list(range(action_space_n))   # action choices
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.decay = decay
        self.q_table = np.zeros((observation_space_n, action_space_n))  # (state, action)

    def choose_action(self, observation):
        flag = np.random.rand()
        if flag <= self.epsilon:  # choose random action
            action = np.random.randint(len(self.action_space))
        else:
            action = self.q_table[observation].argmax()
        return action

    def update_q_table(self, observation, action, reward, next_state, done):
        if done:
            q_target = reward
        else:
            max_next_action = self.q_table[next_state].argmax()
            q_target = reward+self.gamma*self.q_table[next_state][max_next_action]
        q_predict = self.q_table[observation][action]
        self.q_table[observation][action] += self.lr * (q_target - q_predict)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = QLearning(env.action_space.n, env.observation_space.n)
    observation = env.reset()
    total_rewards = 0
    n_episode = 15000
    # Learning from interactions
    for episode in range(n_episode):
        while True:
            action = agent.choose_action(observation)
            next_state, reward, done, info = env.step(action)

            total_rewards += reward
            # Update Q Table
            agent.update_q_table(observation, action, reward, next_state, done)
            # Update observation
            observation = next_state

            if done:
                print('episode #{} Finished!'.format(episode))
                # print('Q Table:\n{}'.format(agent.q_table))
                observation = env.reset()
                break
        # epsilon decay
        agent.epsilon = agent.min_epsilon + (agent.max_epsilon - agent.min_epsilon) * np.exp(-agent.decay*episode)

    # print average reward
    print('Average reward over each episode is: {:.4f}'.format(total_rewards/n_episode))

    # extracting policy
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        best_action = agent.q_table[state].argmax()
        policy[state] = best_action
        # print('state: {}, action: {}'.format(state, best_action))
    print('Policy:\n{}'.format(policy))

    # Validation for q table
    print('Validation Starting ...')
    for i in range(5):
        while True:
            action = agent.choose_action(observation)
            next_state, reward, done, info = env.step(action)
            if done:
                env.render()
                print('Validation Finished!')
                observation = env.reset()
                break

            # Update observation
            observation = next_state
    env.close()
