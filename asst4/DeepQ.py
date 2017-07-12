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
        self.learn_counter = 0
        self.update_target_number = 300
        self.sess = tf.Session()

    def _build_model(self):
        self.input = tf.placeholder(tf.float32, [None, 4], name='input')
        self.q_target = tf.placeholder(tf.float32, [None, 2], name='Q_target')

        self.w1 = tf.Variable(tf.random_uniform([4, 24], minval=-1.0, maxval=1.0))
        self.b1 = tf.Variable(tf.random_uniform([24], minval=-1.0, maxval=1.0))
        self.l1 = tf.nn.relu(tf.matmul(self.input, self.w1) + self.b1)
        self.w2 = tf.Variable(tf.random_uniform([24, 24], minval=-1.0, maxval=1.0))
        self.b2 = tf.Variable(tf.random_uniform([24], minval=-1.0, maxval=1.0))
        self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2)
        self.w3 = tf.Variable(tf.random_uniform([24, 2], minval=-1.0, maxval=1.0))
        self.b3 = tf.Variable(tf.random_uniform([2], minval=-1.0, maxval=1.0))
        self.output = tf.matmul(self.l2, self.w3) + self.b3

        self.t_input = tf.placeholder(tf.float32, [None, 4], name='input')
        self.t_w1 = tf.Variable(tf.random_uniform([4, 24], minval=-1.0, maxval=1.0))
        self.t_b1 = tf.Variable(tf.random_uniform([24], minval=-1.0, maxval=1.0))
        self.t_l1 = tf.nn.relu(tf.matmul(self.t_input, self.t_w1) + self.t_b1)
        self.t_w2 = tf.Variable(tf.random_uniform([24, 24], minval=-1.0, maxval=1.0))
        self.t_b2 = tf.Variable(tf.random_uniform([24], minval=-1.0, maxval=1.0))
        self.t_l2 = tf.nn.relu(tf.matmul(self.t_l1, self.t_w2) + self.t_b2)
        self.t_w3 = tf.Variable(tf.random_uniform([24, 2], minval=-1.0, maxval=1.0))
        self.t_b3 = tf.Variable(tf.random_uniform([2], minval=-1.0, maxval=1.0))
        self.t_output = tf.matmul(self.t_l2, self.t_w3) + self.t_b3

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
        if self.learn_counter >= self.update_target_number:
            self.learn_counter = 0
            self.t_w1 = tf.assign(self.t_w1, self.w1)
            self.t_b1 = tf.assign(self.t_b1, self.b1)
            self.t_w2 = tf.assign(self.t_w2, self.w2)
            self.t_b2 = tf.assign(self.t_b2, self.b2)
            self.t_w3 = tf.assign(self.t_w3, self.w3)
            self.t_b3 = tf.assign(self.t_b3, self.b3)


        mini_batch = random.sample(self.buffer, batch_size)
        mini_batch = np.array(mini_batch)
        state = mini_batch[:, 0]
        next_state = mini_batch[:, 3]
        t_output, output = self.sess.run(
            [self.t_output, self.output],
            feed_dict={
                self.t_input: next_state,
                self.input: state
            }
        )

        q_target = output.copy()
        index = range(batch_size)
        actions = mini_batch[:, 1]
        rewards = mini_batch[:, 2]
        q_target[index, actions] = rewards + self.factor*np.max(t_output, axis=1)

        self.sess.run(self.train_step, feed_dict={self.input: state, self.q_target: q_target})

        self.greedy = self.greedy - self.greedy_step if self.greedy > self.min_greedy else self.min_greedy
        self.learn_counter += 1


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
