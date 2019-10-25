import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PolicyGradient import pong_inference
import random
import gym
import numpy as np
from collections import deque
import cv2
import os

EPISODE_NUM = 500000 #训练回合数
GAME = "PongDeterministic-v4" #游戏名

#超参数
#LEARNING_RATE = 0.00025 #学习率
LEARNING_RATE = 0.01
GAMMA = 0.99
REGULARIZATION_RATE = 0.0001
action_size = 4

def train(env, render, train):
    input_image = tf.placeholder(tf.float32, [None, pong_inference.IMAGE_SIZE, pong_inference.IMAGE_SIZE,
                                    pong_inference.NUM_CHANNELS], name='x-input')
    action = tf.placeholder(tf.int32, shape=[None])
    target = tf.placeholder(tf.float32, shape=[None])


    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    pi = pong_inference.inference(input_image, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    action_mask = tf.one_hot(action, action_size, 1.0, 0.0)
    action_value_pred = tf.reduce_sum(pi * action_mask, 1) #实际为交叉熵

    loss = tf.reduce_mean(-tf.log(action_value_pred) * target)
    total_loss = loss + tf.add_n(tf.get_collection('losses'))
    train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)\
        .minimize(total_loss,global_step=global_step)


    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        while True:
            episode_reward, G, sample_action, sample_observation = play_montecarlo(env, pi, input_image, render, train)
            _, loss_value, step = sess.run([train_step, loss, global_step],
                                               feed_dict={input_image: sample_observation,
                                                action: sample_action,
                                                target: G})
            print("Step: ",step," LossValue: ",loss_value,"EpisodeReward: ",episode_reward)



    return 0

def play_montecarlo(env, pi, input_image, render=True, train=True):
    #image process
    obeservation = env.reset()
    x_t = cv2.cvtColor(cv2.resize(obeservation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)   #去噪音
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) #输入

    episode_reward = 0
    sample_reward = [] #轨迹
    sample_action = []
    sample_observation = []
    while "I am a single dog":
        if render:
            env.render()
        p = pi.eval(feed_dict={input_image:[s_t]})
        action = np.random.choice(a=range(p[0].shape[0]), size=1, replace=False, p=p[0])#随机选择
        next_observation, reward, done, _ = env.step(action[0])
        x_t1 = cv2.cvtColor(cv2.resize(next_observation, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        episode_reward += reward
        if train:
            sample_observation.append(s_t)
            sample_reward.append(reward)
            sample_action.append(action[0])
            s_t = s_t1
        if done:
            break
    G = [0]*len(sample_reward)
    for i in reversed(range(len(sample_reward))):
        G[i] = (GAMMA*G[(i+1)%len(sample_reward)] + sample_reward[i]) * GAMMA**i  #Gt = GAMMA * Gt+1 + Rt+1
    return episode_reward,G,sample_action,sample_observation

def main(argv=None):
    env = gym.make(GAME)
    env.seed(0)
    train(env, render=False, train=True)
    env.close()

if __name__ == '__main__':
    tf.app.run()
