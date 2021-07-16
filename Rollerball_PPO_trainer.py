"""
Proximal Policy Optimization (PPO)

To run
------
python Rollerball_PPO_trainer.py --train/test
"""
import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mlagents_envs.environment import UnityEnvironment as UE
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
from Rollerball_PPO_brain import PPO

ENV_ID = 'Rollerball'  # environment id
ALG_NAME = 'PPO'
TRAIN_EPISODES = 50000  # total number of episodes for training
TEST_EPISODES = 100  # total number of episodes for testing
MAX_STEPS = 200  # total number of steps for each episode
BATCH_SIZE = 32  # update batch size

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

channel = EngineConfigurationChannel()
env = UE(file_name='RollerBall', seed=47, side_channels=[channel], no_graphics=True)
channel.set_configuration_parameters(time_scale=20)

env.reset()
behavior_name = list(env.behavior_specs)[0]
spec = env.behavior_specs[behavior_name]

state_dim = spec.observation_specs[0].shape[0] # dimention of state
action_dim = spec.action_spec.continuous_size # dimention of action
action_bound = 1

agent = PPO(state_dim, action_dim, action_bound)

t0 = time.time()
if args.train:
    all_episode_reward = []
    for episode in range(1,TRAIN_EPISODES+1):
        env.reset()
        episode_reward = 0
        done = False
        decision_steps, terminal_steps = env.get_steps(behavior_name) # to get initial state
        if len(decision_steps) == 0:
            continue
        else:
            state = np.array(tf.squeeze(decision_steps.obs))
            tracked_agent = decision_steps.agent_id[0]

        for step in range(MAX_STEPS):  # in one episode

            action = agent.get_action(state)
            env.set_actions(behavior_name, ActionTuple(continuous=action[np.newaxis, :]))
            """ 
           Set Actions :env.set_actions(behavior_name: str, action: ActionTuple)
           action is an ActionTuple, which is made up of a 2D np.array
           of dtype=np.int32 for discrete actions, and dtype=np.float32 for continuous actions;
           The first dimension of np.array in the tuple is the number of agents 
           that requested a decision since the last call to env.step();
           The second dimension is the number of discrete or continuous actions for the corresponding array.
           """
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(terminal_steps) == 1:  # The agent terminated its episode
                reward = terminal_steps.reward[tracked_agent]
                episode_reward += reward
                state_ = np.array(tf.squeeze(terminal_steps.obs))
                done = True
            else:
                reward = decision_steps.reward[tracked_agent]
                episode_reward += reward
                state_ = np.array(tf.squeeze(decision_steps.obs))

            agent.store_transition(state, action, reward)
            state = state_

            # update ppo
            if len(agent.state_buffer) >= BATCH_SIZE:
                agent.finish_path(state_, done)
                agent.update()
            if done:
                break
        agent.finish_path(state_, done)

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
    t1 = time.clock()
    agent.load()
    env = UE(file_name='RollerBall', worker_id=1, seed=47, side_channels=[], no_graphics=False)
    mean_reward = []
    for episode in range(1,TEST_EPISODES+1):
        env.reset()
        episode_reward = 0
        done = False
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        if len(decision_steps) == 0:
            continue
        else:
            state = np.array(tf.squeeze(decision_steps.obs))
            tracked_agent = decision_steps.agent_id[0]
        for step in range(MAX_STEPS):

            action = agent.get_action(state,greedy=True)
            env.set_actions(behavior_name, ActionTuple(continuous=action[np.newaxis, :]))
            env.step()
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            if len(terminal_steps) == 1:  # The agent terminated its episode
                reward = terminal_steps.reward[tracked_agent]
                episode_reward += reward
                state_ = np.array(tf.squeeze(terminal_steps.obs))
                done = True
            else:
                reward = decision_steps.reward[tracked_agent]
                episode_reward += reward
                state_ = np.array(tf.squeeze(decision_steps.obs))
            state = state_
            if done:
                break
        print(
            'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode, TEST_EPISODES, episode_reward,
                time.time() - t1))
        mean_reward.append(episode_reward)
    env.close()
    print('Mean_reward:', np.mean(mean_reward))
    plt.plot(mean_reward)
    plt.show()