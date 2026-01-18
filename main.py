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
from Agents.MarioDDQN import DQNAgent
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


def run_ddqn(training_mode, pretrained, double_dqn, num_episodes, exploration_max):
    """Runs the DDQN/DQN agent training or testing."""
    
    # --- 1. Dynamic Naming and Path Definition ---
    agent_name = "Double DQN" if double_dqn else "STANDARD DQN" 
    tqdm_description = f"{agent_name} Episodes"

    # Dynamic output folder definition to separate standard DQN and Double DQN results
    if double_dqn:
        model_dir = "models/DDQN_Double/"
    else:
        model_dir = "models/DQN_Standard/"
    
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = make_env_ddqn(env)
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    
    # DDQN Agent Initialization (model_dir is passed to the agent)
    agent = DQNAgent(state_space=observation_space, action_space=action_space,
                     max_memory_size=30000, batch_size=32, gamma=0.90, lr=0.00025,
                     dropout=0.2, exploration_max=exploration_max, exploration_min=0.02,
                     exploration_decay=0.99, double_dqn=double_dqn, pretrained=pretrained,
                     model_dir=model_dir) # Passing the dynamic folder path

    env.reset()
    total_rewards = []
    
    # --- 2. Load Pretrained Rewards (if continuing training) ---
    rewards_path = os.path.join(model_dir, "total_rewards.pkl")
    
    # Load pretrained DDQN/DQN rewards if training continues from a previous session
    if training_mode and pretrained and os.path.exists(rewards_path):
        try:
            with open(rewards_path, 'rb') as f:
                total_rewards = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load total_rewards.pkl from {rewards_path}. Starting reward list from scratch. Error: {e}")
    
    # --- Setup for GIF/Video Saving (Testing Mode Only) ---
    frames_to_save = []
    if not training_mode:
        init_pygame()
        # Define a folder to temporarily save images (optional, but needed if saving many individual frames)
        # GIF_FOLDER = os.path.join(model_dir, "gif_frames")
        # os.makedirs(GIF_FOLDER, exist_ok=True)
        
    # --- 3. Execution Loop ---
    for ep_num in tqdm(range(num_episodes), desc=tqdm_description):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        is_victory = False # Flag to track victory status for GIF saving
        tmp_info = {
            'coins': 0, 'flag_get': False, 'life': 2, 'status': 'small',
            'TimeLimit.truncated': True, 'x_pos': 40, 'score': 0, 'time': 400
        }
        start_time = time.time()
        
        while True:
            if not training_mode:
                show_state(env, ep_num)
                # Capture frame for GIF/video
                frame_img = env.render(mode='rgb_array')
                frames_to_save.append(frame_img)
            
            action = agent.act(state)
            state_next, _, terminal, info = env.step(int(action[0]))

            if info["x_pos"] != tmp_info["x_pos"]:
                start_time = time.time()

            custom_reward, tmp_info = custom_rewards(info, tmp_info)

            # Check for victory using the same logic as in custom_rewards
            if info['x_pos'] > 3159 or (tmp_info['flag_get'] != info['flag_get'] and info['flag_get']):
                is_victory = True 

            end_time = time.time()
            if end_time - start_time > 15:
                custom_reward -= CUSTOM_REWARDS["death"]
                terminal = True

            total_reward += custom_reward

            state_next = torch.Tensor([state_next])
            custom_reward = torch.tensor([custom_reward]).unsqueeze(0)
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)

            if training_mode:
                agent.remember(state, action, custom_reward, state_next, terminal)
                # Q-network update
                agent.experience_replay()

            state = state_next

            if terminal:
                break
        
        # --- GIF Saving Logic (After Episode Ends) ---
        if not training_mode:
            if is_victory and len(frames_to_save) > 0:
                output_gif_path = os.path.join(model_dir, f"victory_episode_{ep_num + 1}.gif")
                print(f"\nVictory detected! Generating GIF: {output_gif_path}")
                
                # Generate GIF directly from frames (using fps=15 for a good speed)
                try:
                    imageio.mimsave(output_gif_path, frames_to_save, fps=15)
                    print(f"GIF saved successfully to {output_gif_path}")
                except Exception as e:
                    print(f"Error generating GIF: {e}")
            
            # Clear frames for the next episode, regardless of victory status
            frames_to_save = []


        total_rewards.append(total_reward)

        if ep_num != 0 and (ep_num + 1) % 100 == 0:
            print(f"\nEpisode {ep_num + 1} score = {total_rewards[-1]}, average score = {np.mean(total_rewards)}")
        
        print(f"Episode {ep_num + 1} score = {total_rewards[-1]}, average score = {np.mean(total_rewards)}")

    if not training_mode:
        pygame.quit()

    # --- 4. Final Save (Uses the dynamically defined model_dir) ---
    # Saves memory and models after training
    if training_mode:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "ending_position.pkl"), "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open(os.path.join(model_dir, "num_in_queue.pkl"), "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open(os.path.join(model_dir, "total_rewards.pkl"), "wb") as f:
            pickle.dump(total_rewards, f)
            
        if agent.double_dqn:
            torch.save(agent.local_net.state_dict(), os.path.join(model_dir, "DQN1.pt"))
            torch.save(agent.target_net.state_dict(), os.path.join(model_dir, "DQN2.pt"))
        else:
            torch.save(agent.dqn.state_dict(), os.path.join(model_dir, "DQN.pt"))
            
        torch.save(agent.STATE_MEM, os.path.join(model_dir, "STATE_MEM.pt"))
        torch.save(agent.ACTION_MEM, os.path.join(model_dir, "ACTION_MEM.pt"))
        torch.save(agent.REWARD_MEM, os.path.join(model_dir, "REWARD_MEM.pt"))
        torch.save(agent.STATE2_MEM, os.path.join(model_dir, "STATE2_MEM.pt"))
        torch.save(agent.DONE_MEM, os.path.join(model_dir, "DONE_MEM.pt"))

    env.close()


def main():
    # --- Configuration Variables ---
    
    # Global settings
    TRAINING_MODE = False  # Set to True for training, False for testing/evaluation
    PRETRAINED_AGENT = True  # Set to True to load saved Q-values/Network weights

    # Agent selection: choose one from 'QL', 'SARSA', or 'DDQN'
    AGENT_TYPE = 'DDQN' 
    
    # Training/Testing settings
    NUM_EPISODES = 5 # Number of episodes to run
    EXPLORATION_MAX = 0.05 if not TRAINING_MODE else 1.0 # Max exploration rate (lower for testing)
    
    # DDQN specific settings: True for Double DQN, False for STANDARD DQN
    DOUBLE_DQN = True 

    # --- Run Agent ---
    
    if AGENT_TYPE == 'DDQN':
        # Determine the correct name for printing
        agent_name = "Double DQN" if DOUBLE_DQN else "STANDARD DQN" 
        
        print(f"--- Running {AGENT_TYPE} ({agent_name}: {DOUBLE_DQN}) in {'TRAINING' if TRAINING_MODE else 'TESTING'} mode ---")
        run_ddqn(TRAINING_MODE, PRETRAINED_AGENT, DOUBLE_DQN, NUM_EPISODES, EXPLORATION_MAX)
        
    elif AGENT_TYPE in ['QL', 'SARSA']:
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