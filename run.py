import numpy as np
import tensorflow as tf
from tf_agents.environments import utils, tf_py_environment, tf_environment

from Dice21Env import Dice21Env

tf.compat.v1.enable_v2_behavior()

if __name__ == '__main__':
    env = Dice21Env()

    tf_env = tf_py_environment.TFPyEnvironment(env)

    print("TimeStep Specs:", tf_env.time_step_spec())
    print("Action Specs:", tf_env.action_spec())

    time_step = tf_env.reset()
    rewards = []
    steps = []
    num_episodes = 15

    for e in range(num_episodes):
        print("======================")
        print("episode: ", e)
        episode_reward = 0
        episode_steps = 0
        while not time_step.is_last():
            print(f"observation: {time_step.observation}")
            action = tf.random.uniform((), 0, 2, dtype=tf.int32)
            time_step = tf_env.step(action)
            episode_steps += 1
            episode_reward += time_step.reward.numpy()
            print(f"action: {action}")
            print(
              f"reward: {time_step.reward}")

        rewards.append(episode_reward)
        steps.append(episode_steps)
        time_step = tf_env.reset()

    num_steps = np.sum(steps)
    avg_length = np.mean(steps)
    avg_reward = np.mean(rewards)

    print('num_episodes:', num_episodes, 'num_steps:', num_steps)
    print('avg_length', avg_length, 'avg_reward:', avg_reward)
