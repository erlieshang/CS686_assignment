# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class PoleAgent(object):
    def __init__(self):
        self.state_size = 4
        self.action_size = 2
        self.buffer = deque(maxlen=1000)
        self.batch_size = 50
        self.discount = 0.99
        self.greedy = 0.05
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(10, input_dim=self.state_size, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.greedy:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def train_0(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        target[0][action] = reward if done else reward + self.discount * np.amax(self.model.predict(next_state)[0])
        self.model.fit(state, target, epochs=1, verbose=0)

    def train_1(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            self.train_0(state, action, reward, next_state, done)

    def train_2(self, state, action, reward, next_state, done):
        target = self.model.predict(state)
        target[0][action] = reward if done else reward + self.discount * np.amax(self.target.predict(next_state)[0])
        self.model.fit(state, target, epochs=1, verbose=0)

    def train_3(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        for state, action, reward, next_state, done in mini_batch:
            self.train_2(state, action, reward, next_state, done)


def mode0():
    # no experience replay and no target network
    agent0 = PoleAgent()
    for e in range(EPISODES):
        state = env.reset().reshape(1, 4)
        for step in range(500):
            env.render()
            action = agent0.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, 4)
            reward = reward if not done else -10
            agent0.train_0(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, step: {}".format(e + 1, EPISODES, step))
                break


def mode1():
    # no target network
    agent1 = PoleAgent()
    for e in range(EPISODES):
        state = env.reset().reshape(1, 4)
        for step in range(500):
            env.render()
            action = agent1.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, 4)
            reward = reward if not done else -10
            agent1.remember(state, action, reward, next_state, done)
            if len(agent1.buffer) > agent1.batch_size:
                agent1.train_1()
            state = next_state
            if done:
                print("episode: {}/{}, step: {}".format(e + 1, EPISODES, step))
                break


def mode2():
    # no experience replay
    agent2 = PoleAgent()
    for e in range(EPISODES):
        state = env.reset().reshape(1, 4)
        for step in range(500):
            env.render()
            action = agent2.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, 4)
            reward = reward if not done else -10
            agent2.train_2(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, step: {}".format(e + 1, EPISODES, step))
                break
        if e % 2 == 1:
            agent2.update_target()


def mode3():
    # both enabled
    agent3 = PoleAgent()
    for e in range(EPISODES):
        state = env.reset().reshape(1, 4)
        for step in range(500):
            env.render()
            action = agent3.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(1, 4)
            reward = reward if not done else -10
            agent3.remember(state, action, reward, next_state, done)
            if len(agent3.buffer) > agent3.batch_size:
                agent3.train_3()
            state = next_state
            if done:
                print("episode: {}/{}, step: {}".format(e + 1, EPISODES, step))
                break
        if e % 2 == 1:
            agent3.update_target()


if __name__ == "__main__":
    EPISODES = 1000
    env = gym.make('CartPole-v1')
    # mode0()
    # mode1()
    # mode2()
    mode3()








