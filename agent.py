import collections
import random
from environment import Env
import numpy as np
import sys

from dqn import *

class DqnAgent:
    def __init__(self, env):
        # environment
        self.env = env
        self.observations = self.env.observation_space
        self.actions = self.env.action_space
        # agent properties
        self.replay_buffer_size = int(1e6)
        self.train_start = int(1e5)
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.99999
        # nq network properties
        self.state_shape = self.observations
        self.learning_rate = 1e-4
        self.model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.target_model.update_model(self.model)
        self.batch_size = 32

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # select random move from possible moves
            valid_moves = np.where(state[:,0] == 1)[0]
            return np.random.choice(valid_moves)
        else:
            # select valid move with best predicted q-value
            q_values = self.model.predict(np.reshape(state, (1,) + state.shape))[0]
            for i in range(25):
                if state[i, 0] == 0:
                    q_values[i] = -sys.float_info.max
            action = np.argmax(q_values)
            return action

    def train(self, target_avg_reward):
        best_total_reward = 0.0
        avg_reward = 0.0
        reward_sum = 0
        episode = 0
        while avg_reward < target_avg_reward:
            # play one episode and learn
            episode += 1
            total_reward = 0.0
            state = self.env.reset()
            episode_buffer = []
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_buffer.append((state, action, reward, next_state, done))
                self.replay()
                total_reward += reward
                state = next_state
                if done:
                    reward = total_reward
                    reward_sum += total_reward
                    if total_reward > best_total_reward:
                        best_total_reward = total_reward
                    # remember episode
                    for step in reversed(episode_buffer):
                        (state, action, _, next_state, done) = step
                        self.remember(state, action, reward, next_state, done)
                        reward *= self.gamma # discount reward
                    if episode % 1000 == 0:
                        # update target model
                        self.target_model.update_model(self.model)
                        avg_reward = reward_sum / 1000
                        reward_sum = 0
                        print(f'Episode #{episode}, avg score {avg_reward}, best score {best_total_reward}, epsilon {self.epsilon}')
                        # save progress in case of crash or user interrupt
                        self.model.save_model("knister_dqn.h5")
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

    def play(self):
        self.model.load_model('knister_dqn.h5')
        self.epsilon = -1.0 # do not use random moves
        state = self.env.reset()
        self.env.render()
        input('press enter')
        while True:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            self.env.render()
            input('press enter')
            if done:
                print(f'Score: {self.env.get_score()}')
                break

if __name__ == '__main__':
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    env = Env()
    agent = DqnAgent(env)
    if mode == 'train':
        agent.train(50.0)
    elif mode == 'play':
        agent.play()
    else:
        print('python agent.py (train|play)')