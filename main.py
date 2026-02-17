import os
import time
import pickle
import numpy as np
import torch
import pygame
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY
from tqdm import tqdm

# Import Agents and Utilities
from Agents.QL_Agent import QL_Agent
from Utils.environment import MaxAndSkipEnv, ProcessFrame84, ImageToPyTorch, BufferWrapper, ScaledFloatFrame, PixelNormalization
from Utils.setup_env import custom_rewards, init_pygame, show_state, CUSTOM_REWARDS


def make_env_ql(enviroment):
    """Wraps the environment for QL/SARSA agents."""
    enviroment = MaxAndSkipEnv(enviroment)
    enviroment = ProcessFrame84(enviroment)
    enviroment = ImageToPyTorch(enviroment)
    enviroment = BufferWrapper(enviroment, 4)
    enviroment = ScaledFloatFrame(enviroment)
    return JoypadSpace(enviroment, RIGHT_ONLY)


def make_env_ddqn(enviroment):
    """Wraps the environment for DDQN/DQN agents."""
    enviroment = MaxAndSkipEnv(enviroment)
    enviroment = ProcessFrame84(enviroment)
    enviroment = ImageToPyTorch(enviroment)
    enviroment = BufferWrapper(enviroment, 4)
    enviroment = PixelNormalization(enviroment)
    return JoypadSpace(enviroment, RIGHT_ONLY)


def agent_training_ql(agent, env, num_episodes, total_rewards, agent_type='QL'):
    """Performs training for Q-Learning or SARSA."""
    update_func = agent.update_qval if agent_type == 'QL' else agent.update_qval_sarsa

    with tqdm(total=num_episodes, desc=f"{agent_type} Training Episodes") as progress_bar:
        for i_episode in range(num_episodes):
            observation = env.reset()
            state = agent.obs_to_state(observation)
            episode_reward = 0
            tmp_info = {
                'coins': 0, 'flag_get': False, 'life': 2, 'status': 'small',
                'TimeLimit.truncated': True, 'x_pos': 40, 'score': 0, 'time': 400
            }
            start_time = time.time()
            action = agent.take_action(state) if agent_type == 'SARSA' else None

            while True:
                current_action = action if agent_type == 'SARSA' else agent.take_action(state)
                next_obs, _, terminal, info = env.step(current_action)

                if info["x_pos"] != tmp_info["x_pos"]:
                    start_time = time.time()

                custom_reward, tmp_info = custom_rewards(info, tmp_info)

                end_time = time.time()
                if end_time - start_time > 15:
                    custom_reward -= CUSTOM_REWARDS["death"]
                    terminal = True

                next_state = agent.obs_to_state(next_obs)

                if agent_type == 'SARSA':
                    next_action = agent.take_action(next_state)
                    agent.update_qval_sarsa(current_action, state, custom_reward, next_state, next_action, terminal)
                    action = next_action
                else: # QL
                    agent.update_qval(current_action, state, custom_reward, next_state, terminal)

                state = next_state
                episode_reward += custom_reward

                if terminal:
                    break

            if isinstance(total_rewards, np.ndarray):
                total_rewards = np.append(total_rewards, episode_reward)
            else:
                total_rewards.append(episode_reward)

            progress_bar.update(1)
            progress_bar.set_postfix({'Reward': episode_reward})

            # Saving the reward array and agent every 10 episodes
            if i_episode % 10 == 0:
                model_dir = f"models/{agent_type}/"
                os.makedirs(model_dir, exist_ok=True)
                np.save(os.path.abspath(model_dir + "rewards.npy"), np.array(total_rewards))
                with open(os.path.abspath(model_dir + "model.pkl"), 'wb') as file:
                    pickle.dump(agent.state_a_dict, file)
                print(f"\n{agent_type} Rewards and model are saved.\n")


def agent_testing_ql(agent, env, num_episodes):
    """Performs testing for Q-Learning or SARSA."""
    total_rewards = []
    init_pygame()

    for i_episode in range(num_episodes):
        observation = env.reset()
        state = agent.obs_to_state(observation)
        episode_reward = 0
        tmp_info = {
            'coins': 0, 'flag_get': False, 'life': 2, 'status': 'small',
            'TimeLimit.truncated': True, 'x_pos': 40, 'score': 0, 'time': 400
        }

        while True:
            # Exploitation: always choose the best Q-value action for testing
            show_state(env, i_episode)
            action = np.argmax(agent.get_qval(state))
            next_obs, _, terminal, info = env.step(action)

            custom_reward, tmp_info = custom_rewards(info, tmp_info)
            episode_reward += custom_reward

            next_state = agent.obs_to_state(next_obs)
            state = next_state

            if terminal:
                break

        total_rewards.append(episode_reward)
        print(f"Total reward after testing episode {i_episode + 1} is {episode_reward}")

    pygame.quit()


def main():
    # --- Configuration Variables ---

    # Global settings
    TRAINING_MODE = False  # Set to True for training, False for testing/evaluation
    PRETRAINED_AGENT = True  # Set to True to load saved Q-values/Network weights

    # Agent selection: choose one from 'QL', 'SARSA', or 'DDQN'
    AGENT_TYPE = 'DDQN'

    # Training/Testing settings
    NUM_EPISODES = 500 # Number of episodes to run
    EXPLORATION_MAX = 0.05 if not TRAINING_MODE else 1.0 # Max exploration rate (lower for testing)

    # DDQN specific settings: True for Double DQN, False for STANDARD DQN
    DOUBLE_DQN = True

    # --- Run Agent ---

    if AGENT_TYPE in ['QL', 'SARSA']:
        print(f"--- Running {AGENT_TYPE} in {'TRAINING' if TRAINING_MODE else 'TESTING'} mode ---")

        env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
        env = make_env_ql(env)
        agent_mario = QL_Agent(env)

        rewards = []
        model_dir = f"models/{AGENT_TYPE}/"

        # Load Q-values and rewards if pretrained
        if PRETRAINED_AGENT:
            q_values_path = os.path.abspath(os.path.join(model_dir, "model.pkl"))
            rewards_path = os.path.abspath(os.path.join(model_dir, "rewards.npy"))

            if os.path.exists(q_values_path) and os.path.exists(rewards_path):
                try:
                    with open(q_values_path, 'rb') as f:
                        trained_q_values = pickle.load(f)
                    rewards = np.load(rewards_path)
                    agent_mario.state_a_dict = trained_q_values
                    print(f"Loaded pretrained {AGENT_TYPE} model.")
                except Exception as e:
                    print(f"Error loading pretrained model: {e}. Starting from scratch.")
                    rewards = []
            else:
                print(f"No pretrained {AGENT_TYPE} model found. Starting from scratch.")

        if TRAINING_MODE:
            agent_training_ql(agent_mario, env, NUM_EPISODES, rewards, AGENT_TYPE)

        if not TRAINING_MODE:
            agent_testing_ql(agent_mario, env, NUM_EPISODES)

        env.close()

    else:
        print(f"Error: Unknown AGENT_TYPE '{AGENT_TYPE}'. Please choose 'QL', 'SARSA', or 'DDQN'.")


if __name__ == "__main__":
    main()
