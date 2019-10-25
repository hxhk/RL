import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PolicyGradient import pong_inference
import random
import gym
import numpy as np
from collections import deque
import cv2
import os
#参数 from flappy bird
OBSERVE = 1000
EXPLORE = 3000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.3
#INITIAL_EPSILON = 0.0001
FRAME_PER_ACTION = 1
REPLAY_MEMORY = 50000
GAMMA = 0.99
#参数 from mnist_train
BATCH_SIZE = 100
LEARNING_RATE = 0.00025
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="pong_model/"
MODEL_NAME="pong_model"

"""
        checkpoint = tf.train.get_checkpoint_state("MNIST_model")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
"""


def train(mnist,env,render):
    #x输入图像 y x状态下的输出的动作价值  y_动作价值  action_p动作概率
    x = tf.placeholder(tf.float32, [None, pong_inference.IMAGE_SIZE, pong_inference.IMAGE_SIZE,
                                    pong_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None], name='y-input')
    action_p = tf.placeholder(tf.float32, [None, pong_inference.OUTPUT_NODE], name='action-probability')


    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = pong_inference.inference(x, True, regularizer)

    readout_action = tf.reduce_sum(tf.multiply(y, action_p), reduction_indices=1) #结果是BATCHx1的矩阵
    #readout_action 动作价值函数
    global_step = tf.Variable(0, trainable=False)


    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=readout_action, labels=y_)
    cost = tf.reduce_mean(tf.square(y_ - readout_action))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cost + tf.add_n(tf.get_collection('losses'))



    # 梯度裁剪
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train_step = optimizer.minimize(loss, global_step=global_step)
#    with tf.control_dependencies([train_step, variables_averages_op]):
        #train_op = tf.no_op(name='train')

    grads, variables = zip(*optimizer.compute_gradients(loss))
    grads, global_norm = tf.clip_by_global_norm(grads, 5)
    train_op = optimizer.apply_gradients(zip(grads, variables))



    #经验袋
    D = deque()
    epsilon = INITIAL_EPSILON


    with tf.Session() as sess:

        saver = tf.train.Saver()
        tf.global_variables_initializer().run()

        #获得观察
        observation = env.reset()
        observation = cv2.resize(observation, (80, 80))
        t = 0


        while t < TRAINING_STEPS:
            if render:
                env.render()
            a_t = np.zeros([pong_inference.NUM_LABELS]) #动作空间
            action_index = 0
            y_t = y.eval(feed_dict={x: [observation]})[0]
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon: # epsilon-greed
                    print("----------Random Action----------")
                    action_index = random.randrange(pong_inference.NUM_LABELS)
                    a_t[action_index] = 1
                else:
                    action_index = np.argmax(y_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1 # do nothing
            #降低 epsilon 探索概率
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            next_observation, reward, done, _ = env.step(action_index)

            next_observation = cv2.resize(next_observation, (80, 80))
            D.append((observation, a_t, reward, next_observation, done))
            if len(D) > REPLAY_MEMORY:
                D.popleft()
                    # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH_SIZE)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = y.eval(feed_dict = {x: s_j1_batch})
#                graph = tf.get_default_graph()
#                x1 = graph.get_tensor_by_name("weight:0")
#                print(x1.eval())

                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                __, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={y_: y_batch,
                                                action_p: a_batch,
                                                x: s_j_batch}
                )  #给placeholder赋值
                print("loss_value: ", loss_value)

            # 更新现在状态
            observation = next_observation.copy()
            t += 1

            # print info

            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, \
                "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward, \
            "/ Q_MAX %e" % np.max(y_t), "y_t: ", y_t)

            if t % 500000 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
"""
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                atari_inference.IMAGE_SIZE,
                atari_inference.IMAGE_SIZE,
                atari_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
"""




def main(argv=None):
    env = gym.make('Pong-v0')
    env.seed(0)
    print('观测空间 = {}'.format(env.observation_space))
    print('动作空间 = {}'.format(env.action_space))

    mnist = input_data.read_data_sets("C:/Users/huangxi/PycharmProjects/ReinforcementLearning/MNIST_data", one_hot=True)
    train(mnist, env, render=True)
    env.close()

if __name__ == '__main__':
    tf.app.run()
