# pmt_ddqn_agent.py (پیاده‌سازی PMT-DDQN با DDQN و ترکیب خطای TD و نمایش متخصص)
import numpy as np
import random
import tensorflow as tf
from collections import deque

class PMT_DDQN_Agent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = np.prod(env.action_space.nvec)
        self.memory_demo = deque(maxlen=2000)
        self.memory_self = deque(maxlen=2000)
        self.gamma = 0.99
        self.epsilon = 0.8
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.target_network.set_weights(self.q_network.get_weights())
        self.update_target_freq = 100

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                      loss='huber_loss')
        return model

    def store_transition(self, state, action, reward, next_state, demo=False):
        if demo:
            self.memory_demo.append((state, action, reward, next_state))
        else:
            self.memory_self.append((state, action, reward, next_state))

    def train(self, episodes=500):
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(200):
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_transition(state, action, reward, next_state)
                total_reward += reward
                state = next_state
                if done:
                    break
                self.replay()
                if step % self.update_target_freq == 0:
                    self.target_network.set_weights(self.q_network.get_weights())
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")
        return total_reward

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return [np.random.randint(env.action_space.nvec[i]) for i in range(len(env.action_space.nvec))]
        q_values = self.q_network.predict(state.reshape(1, -1))
        return np.argmax(q_values, axis=1)[0]

    def replay(self):
        batch_demo = random.sample(self.memory_demo, min(len(self.memory_demo), self.batch_size // 2))
        batch_self = random.sample(self.memory_self, min(len(self.memory_self), self.batch_size // 2))
        batch = batch_demo + batch_self
        
        states, actions, rewards, next_states = zip(*batch)
        target_qs = self.q_network.predict(np.array(states))
        next_qs = self.target_network.predict(np.array(next_states))
        for i, (action, reward) in enumerate(zip(actions, rewards)):
            target_qs[i][action] = reward + self.gamma * np.max(next_qs[i])
        self.q_network.fit(np.array(states), target_qs, epochs=1, verbose=0)

# main.py (اجرای PMT-DDQFD، PMT-DDQN و مقایسه با DQN)
from environment.crn_env import CRNEnvironment
from agents.ddqn_agent import PMT_DDQFD_Agent
from agents.pmt_ddqn_agent import PMT_DDQN_Agent
from agents.dqn_agent import DQNAgent
from utils.plotting import plot_results

env = CRNEnvironment(num_sus=5)
agents_pmt_ddqfd = [PMT_DDQFD_Agent(env) for _ in range(env.num_new_SUs)]
agents_pmt_ddqn = [PMT_DDQN_Agent(env) for _ in range(env.num_new_SUs)]
agents_dqn = [DQNAgent(env) for _ in range(env.num_new_SUs)]

results_pmt_ddqfd, results_pmt_ddqn, results_dqn = [], [], []
for agent in agents_pmt_ddqfd:
    rewards = agent.train(episodes=500)
    results_pmt_ddqfd.append(rewards)

for agent in agents_pmt_ddqn:
    rewards = agent.train(episodes=500)
    results_pmt_ddqn.append(rewards)

for agent in agents_dqn:
    rewards = agent.train(episodes=500)
    results_dqn.append(rewards)

plot_results(results_pmt_ddqfd, ["PMT-DDQFD" for _ in results_pmt_ddqfd], "Reward Comparison")
plot_results(results_pmt_ddqn, ["PMT-DDQN" for _ in results_pmt_ddqn], "Reward Comparison")
plot_results(results_dqn, ["DQN" for _ in results_dqn], "Reward Comparison")
