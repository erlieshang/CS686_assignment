# Author: erlie.shang@uwaterloo.ca

import gym
import tensorflow as tf
import random
from tensorflow.contrib import learn
import numpy as np


class PoleAgent(object):
    def __init__(self):
        self.buffer = list()
        self.factor = 0.95
        self.model = None
        self.learning_rate = 0.01
        self.min_greedy = 0.05
        self.greedy = 1.0
        self.greedy_step = 0.001
        self._build_model()
        self.sess = tf.Session()

    def _build_model(self):
        self.input = tf.placeholder(tf.float32, [None, 4], name='input')
        self.q_target = tf.placeholder(tf.float32, [None, 2], name='Q_target')
        w1 = tf.Variable(tf.random_uniform([4, 24], minval=-1.0, maxval=1.0))
        b1 = tf.Variable(tf.random_uniform([24], minval=-1.0, maxval=1.0))
        l1 = tf.nn.relu(tf.matmul(self.input, w1) + b1)
        w2 = tf.Variable(tf.random_uniform([24, 24], minval=-1.0, maxval=1.0))
        b2 = tf.Variable(tf.random_uniform([24], minval=-1.0, maxval=1.0))
        l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
        w3 = tf.Variable(tf.random_uniform([24, 2], minval=-1.0, maxval=1.0))
        b3 = tf.Variable(tf.random_uniform([2], minval=-1.0, maxval=1.0))
        self.output = tf.matmul(l2, w3) + b3

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.output))
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def act(self, observation):
        if random.random() <= self.greedy:
            return random.randrange(0, 2)
        else:
            actions = self.sess.run(self.output, feed_dict={self.input: observation})
            return np.argmax(actions)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train(self, batch_size=100):
        mini_batch = random.sample(self.buffer, batch_size)
        for state, action, reward, next_state, done in mini_batch:


def main():
    env = gym.make('CartPole-v0')

    for i_episode in range(20):
        observation = env.reset()
        done = False
        while not done:
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))


if __name__ == "__main__":
    main()
