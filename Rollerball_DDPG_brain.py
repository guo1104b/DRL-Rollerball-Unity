"""
Deep Deterministic Policy Gradient (DDPG), an algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Based on:
1. Deterministic Policy Gradient Algorithms, Silver et al. 2014
2. Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016

Dependencies:
tensorflow >=2.0.0a0
tensorflow-probability 0.6.0
tensorlayer >=2.0.0
mlagents_envs >=0.27.0  [https://github.com/Unity-Technologies/ml-agents]
"""

import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl

#####################  hyper parameters  ####################

ENV_ID = 'Rollerball'  # environment id
ALG_NAME = 'DDPG'

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 32  # update batchsize

MAX_EPISODES = 200  # total number of episodes for training
MAX_EP_STEPS = 200  # total number of steps for each episode
TEST_PER_EPISODES = 10  # test the model per episodes
VAR = 3  # control exploration


###############################  DDPG  ####################################

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, a_dim, s_dim, a_bound):
        # memory is used to store array of data when running.
        # the number of MEMORY_CAPACITY: two states, one action and a reward.
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.var = VAR

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(layer)
            a_ = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(layer)
            a = tl.layers.Lambda(lambda x: x * a_bound, name='lambda')(a_)
            # a = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(a_)  # The range of x is limited to [-1,1] by tanh, and then mapped
            return tl.models.Model(inputs=inputs, outputs=a, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s, a])
            x = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
        self.actor.train()
        self.critic.train()

        # For first assignment only
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        # Build actor_target network, consistent with the actor parameters but not trainable
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        # Build critic_target network, consistent with the critic parameters but not trainable
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        # Exponential moving average
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: action
        """

        a = self.actor(np.array([s], dtype=np.float32))[0]
        print('a:',a)
        if greedy:
            return a
        return np.clip(np.random.normal(a, self.var), -self.a_bound, self.a_bound)  # add randomness to action selection for exploration

    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= .9995
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1:-self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        # Critic update is similar to DQN，but it is calculated with critic_target rather than argmax
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            td_error = tf.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))

        # The goal of actor is to get the maximum Q value
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # Gradient ascent to maximize tht q
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
        self.ema_update()

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))

        # pointer records how much data has come in.
        # index is the location where the latest data is recorded.
        # So it's a cycle, when MEMORY_CAPACITY is full, index starts at the bottom again.
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'ddpg_actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'ddpg_actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'ddpg_critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'ddpg_critic_target.hdf5'), self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'ddpg_actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'ddpg_actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'ddpg_critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'ddpg_critic_target.hdf5'), self.critic_target)

