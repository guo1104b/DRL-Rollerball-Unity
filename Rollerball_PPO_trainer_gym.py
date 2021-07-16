"""
Proximal Policy Optimization (PPO)

To run
------
python Rollerball_PPO_trainer_gym.py --train/test
"""
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from Rollerball_PPO_brain import PPO
from gym_unity.envs import UnityToGymWrapper

ENV_ID = 'Rollerball'  # environment id
ALG_NAME = 'PPO'
TRAIN_EPISODES = 50000  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
MAX_STEPS = 200  # total number of steps for each episode
BATCH_SIZE = 32  # update batch size
RENDER = False  # render while training
RANDOM_SEED = 1  # random seed

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()
channel = EngineConfigurationChannel()
unity_env = UE(file_name='RollerBall', seed=47, side_channels=[channel], no_graphics=True)
channel.set_configuration_parameters(time_scale=20)
env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)

# reproducible
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

agent = PPO(state_dim, action_dim, action_bound)

t0 = time.time()
if args.train:
    all_episode_reward = []
    for episode in range(1,TRAIN_EPISODES+1):
        state = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS):  # in one episode
            if RENDER:
                env.render()
            action = agent.get_action(state)
            state_, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward)
            state = state_
            episode_reward += reward

            # update ppo
            if len(agent.state_buffer) >= BATCH_SIZE:
                agent.finish_path(state_, done)
                agent.update()
            if done:
                break

        if episode == 1:
            all_episode_reward.append(episode_reward)
        else:
            # Calculate cumulative reward
            all_episode_reward.append(all_episode_reward[-1] + (episode_reward - all_episode_reward[-1]) / (len(all_episode_reward)+1))
        if episode == 1 or episode % 1000 == 0:
            print(
                'Training  | Episode: {}/{}  | All Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode, TRAIN_EPISODES, all_episode_reward[-1], time.time() - t0)
            )
        plt.ion()
        plt.cla()
        plt.title('PPO')
        plt.plot(np.arange(len(all_episode_reward)), all_episode_reward)
        plt.xlabel('Episode')
        plt.ylabel('Moving averaged episode reward')
        plt.show()
        plt.pause(0.1)
    agent.save()
    env.close()

    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

if args.test:
    agent.load()
    unity_env = UE(file_name='RollerBall', worker_id=1, seed=47, side_channels=[], no_graphics=False)
    env = UnityToGymWrapper(unity_env, uint8_visual=True, allow_multiple_obs=False)
    mean_reward = []
    for episode in range(TEST_EPISODES):
        state = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS):
            env.render()
            state, reward, done, info = env.step(agent.get_action(state, greedy=True))
            episode_reward += reward
            if done:
                break
        print(
            'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode + 1, TEST_EPISODES, episode_reward,
                time.time() - t0))
        mean_reward.append(episode_reward)
    plt.plot(mean_reward)
    plt.show()