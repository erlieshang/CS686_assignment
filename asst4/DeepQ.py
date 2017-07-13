# Author: erlie.shang@uwaterloo.ca

import gym
import os
import tensorflow as tf
from collections import deque
import random
import numpy as np


class PoleAgent(object):
    def __init__(self):
        self.buffer = deque()
        self.buffer_size = 1000
        self.factor = 0.99
        self.learning_rate = 0.01
        self.greedy = 0.05
        self._build_model()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def save(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)

    def load(self):
        if os.path.isfile('./tmp/model.ckpt'):
            saver = tf.train.Saver()
            saver.restore(self.sess, "./tmp/model.ckpt")
            print("Model restored.")

    def _build_model(self):
        self.input = tf.placeholder(tf.float32, [None, 4], name='input')
        self.q_target = tf.placeholder(tf.float32, [None, 2], name='Q_target')

        self.w1 = tf.Variable(tf.random_uniform([4, 10], minval=-1.0, maxval=1.0))
        self.b1 = tf.Variable(tf.random_uniform([10], minval=-1.0, maxval=1.0))
        self.l1 = tf.nn.relu(tf.matmul(self.input, self.w1) + self.b1)
        self.w2 = tf.Variable(tf.random_uniform([10, 10], minval=-1.0, maxval=1.0))
        self.b2 = tf.Variable(tf.random_uniform([10], minval=-1.0, maxval=1.0))
        self.l2 = tf.nn.relu(tf.matmul(self.l1, self.w2) + self.b2)
        self.w3 = tf.Variable(tf.random_uniform([10, 2], minval=-1.0, maxval=1.0))
        self.b3 = tf.Variable(tf.random_uniform([2], minval=-1.0, maxval=1.0))
        self.output = tf.matmul(self.l2, self.w3) + self.b3

        self.t_input = tf.placeholder(tf.float32, [None, 4], name='t_input')
        self.t_w1 = tf.Variable(tf.random_uniform([4, 10], minval=-1.0, maxval=1.0))
        self.t_b1 = tf.Variable(tf.random_uniform([10], minval=-1.0, maxval=1.0))
        self.t_l1 = tf.nn.relu(tf.matmul(self.t_input, self.t_w1) + self.t_b1)
        self.t_w2 = tf.Variable(tf.random_uniform([10, 10], minval=-1.0, maxval=1.0))
        self.t_b2 = tf.Variable(tf.random_uniform([10], minval=-1.0, maxval=1.0))
        self.t_l2 = tf.nn.relu(tf.matmul(self.t_l1, self.t_w2) + self.t_b2)
        self.t_w3 = tf.Variable(tf.random_uniform([10, 2], minval=-1.0, maxval=1.0))
        self.t_b3 = tf.Variable(tf.random_uniform([2], minval=-1.0, maxval=1.0))
        self.t_output = tf.matmul(self.t_l2, self.t_w3) + self.t_b3

        self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.output))
        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

    def act(self, observation):
        if random.random() <= self.greedy:
            return random.randrange(0, 2)
        else:
            state = observation.reshape(1, 4)
            actions = self.sess.run(self.output, feed_dict={self.input: state})
            return np.argmax(actions)

    def predict(self, observation):
        state = observation.reshape(1, 4)
        actions = self.sess.run(self.output, feed_dict={self.input: state})
        return np.argmax(actions)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def update_target(self):
        self.t_w1 = tf.assign(self.t_w1, self.w1)
        self.t_b1 = tf.assign(self.t_b1, self.b1)
        self.t_w2 = tf.assign(self.t_w2, self.w2)
        self.t_b2 = tf.assign(self.t_b2, self.b2)
        self.t_w3 = tf.assign(self.t_w3, self.w3)
        self.t_b3 = tf.assign(self.t_b3, self.b3)
        self.t_l1 = tf.nn.relu(tf.matmul(self.t_input, self.t_w1) + self.t_b1)
        self.t_l2 = tf.nn.relu(tf.matmul(self.t_l1, self.t_w2) + self.t_b2)
        self.t_output = tf.matmul(self.t_l2, self.t_w3) + self.t_b3
        print 'target updated'

    def fix_shape(self, nparray):
        ret = [[0, 0, 0, 0] for _ in range(nparray.size)]
        for i in range(nparray.size):
            for j in range(4):
                ret[i][j] = nparray[i][j]
        return np.array(ret)

    def train(self, batch_size=50):
        mini_batch = random.sample(self.buffer, batch_size)
        mini_batch = np.array(mini_batch)
        state = mini_batch[:, 0]
        next_state = mini_batch[:, 3]
        state = self.fix_shape(state)
        next_state = self.fix_shape(next_state)
        t_output, output = self.sess.run([self.t_output, self.output], {self.t_input: next_state, self.input: state, })
        q_target = output.copy()
        index = range(batch_size)
        actions = mini_batch[:, 1].astype(int)
        rewards = mini_batch[:, 2]
        q_target[index, actions] = rewards + self.factor * np.max(t_output, axis=1)

        self.sess.run(self.train_step, feed_dict={self.input: state, self.q_target: q_target})


def main():
    env = gym.make('CartPole-v0')
    agent = PoleAgent()
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        while True:
            env.render()
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.add(state, action, reward, next_state)
            state = next_state
            if len(agent.buffer) > 100:
                agent.train()
            if done:
                break
        print("episode: {}/{}".format(e + 1, episodes))
        if (e + 1) % 2 == 0:
            agent.update_target()


if __name__ == "__main__":
    main()
