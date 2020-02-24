import collections
import random
from environment import Env
import numpy as np
import sys

from dqn import *

class DqnAgent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space
        self.actions = 25
        # DQN Agent Variables
        self.replay_buffer_size = int(1e6)
        self.train_start = int(1e5)
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99999
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-4
        self.model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model.update_model(self.model)
        self.batch_size = 32

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            valid_moves = np.where(state[:,0] == 1)[0]
            return np.random.choice(valid_moves)
        else:
            q_values = self.model.predict(np.reshape(state, (1,) + state.shape))[0]
            for i in range(25):
                if state[i, 0] == 0:
                    q_values[i] = -sys.float_info.max
            action = np.argmax(q_values)
            return action

    def train(self, num_episodes):
        best_total_reward = 0.0
        reward_sum = 0
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()
            episode_buffer = []
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                # self.remember(state, action, reward, next_state, done)
                episode_buffer.append((state, action, reward, next_state, done))
                self.replay()
                total_reward += reward
                state = next_state
                if episode == 0:
                    best_total_reward = total_reward
                if done:
                    # remember episode
                    reward = total_reward
                    reward_sum += total_reward
                    for step in reversed(episode_buffer):
                        (state, action, _, next_state, done) = step
                        self.remember(state, action, reward, next_state, done)
                        reward *= self.gamma # discount reward
                    if episode % 1000 == 0:
                        self.target_model.update_model(self.model)
                        print("Episode: ", episode+1,
                        " Total Reward: ", total_reward,
                        " Epsilon: ", round(self.epsilon,3),
                        " Avg Reward: ", reward_sum / 1000)
                        reward_sum = 0
                        self.model.save_model("knister_dqn.h5")
                    if total_reward > best_total_reward:
                        best_total_reward = total_reward
                        print("NEW BEST REWARD: ", best_total_reward)
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min and len(self.memory) >= self.train_start:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.asarray(states)
        states_next = np.asarray(states_next)

        q_values = self.model.predict(states)
        q_values_next = self.target_model.predict(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        self.model.train(states, q_values)

    def play(self, num_episodes, render=True):
        self.model.load_model("knister_dqn.h5")
        self.epsilon = -1.0 # do not use random moves
        for episode in range(num_episodes):
            state = self.env.reset()
            if render:
                self.env.render()
                input('press enter')
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if render:
                    self.env.render()
                    input('press enter')
                if done:
                    print(f'Score: {self.env.get_score()}')
                    break

if __name__ == "__main__":
    env = Env()
    agent = DqnAgent(env)
    import os
    # if os.path.exists("knister_dqn.h5"):
    #     agent.model.load_model("knister_dqn.h5")
    #     agent.target_model.load_model("knister_dqn.h5")
    #agent.train(num_episodes=int(1e12))
    input("Play?")
    agent.play(num_episodes=1, render=True)