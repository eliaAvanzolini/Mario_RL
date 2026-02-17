from typing import Any, cast
from nes_py.wrappers import JoypadSpace
import gym
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.time_limit import TimeLimit
from torch.optim import Adam
from tqdm import tqdm
from collections import deque
import torch
import numpy as np
from models import MarioInterface, DoubleDeepQLearning
# from setup_env import custom_reward
from torch.utils.data import WeightedRandomSampler
from operator import itemgetter
from metric_logger import MetricLogger
import os




# Kindly adapted by the Pytorch tutorial at https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        obvs = None
        done = False
        info = {}

        for _ in range(self._skip):
            # Accumulate reward and repeat the same action
            obvs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obvs, total_reward, done, info

class ReplayBufferWeightedSampling:
    def __init__(self, queue_max_len, bin_size):
        self.bin_idx_queue = deque(maxlen=queue_max_len)
        self.bin_idx_count: dict[int, int] = {}
        self.main_queue = deque(maxlen=queue_max_len)
        self.bin_size = bin_size
        self.queue_max_len = queue_max_len

    def append(self, elm: Any, x_pos: int) -> None:
        bin_idx = x_pos // self.bin_size
        if bin_idx in self.bin_idx_count.keys():
            self.bin_idx_count[bin_idx] += 1
        else:
            self.bin_idx_count[bin_idx] = 1

        # Get idx of first element in bin_idx_queue if we are going to remove it
        if  len(self.bin_idx_queue) == self.queue_max_len:
            idx_to_be_removed = self.bin_idx_queue[0]
            self.bin_idx_count[idx_to_be_removed] -= 1

        # VERY IMPORTANT! The two queues have always to be synchronized
        self.main_queue.append(elm)
        self.bin_idx_queue.append(bin_idx)

    def weighted_sample(self, k, uniform=False):
        if uniform:
            len_queue = len(self.main_queue)
            probs = [1 / len_queue] * len_queue
        else:
            num_bins = len(self.bin_idx_count)
            probs = list(map(lambda idx: (1 / num_bins) * (1 / self.bin_idx_count[idx]), self.bin_idx_queue))
        sample_indices = list(WeightedRandomSampler(probs, num_samples=k, replacement=False))
        return itemgetter(*sample_indices)(self.main_queue)

    def __len__(self):
        return len(self.main_queue)

ACTION_SPACE_MINIMAL = [['right'], ['right', 'A']]


class MarioTrainer():
    def __init__(self, model_class: type[MarioInterface],
                 opt = Adam,
                 checkpoint_folder="model_checkpoints"):
        self.opt_class = opt
        env = gym.make("SuperMarioBros-1-1-v3")

        self.action_space_size = len(ACTION_SPACE_MINIMAL)
        env = JoypadSpace(env, ACTION_SPACE_MINIMAL)

        env = SkipFrame(env, skip=4)
        env = ResizeObservation(env, shape=(84,84))
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=4)
        # env = NormalizeReward(env)
        env = RecordEpisodeStatistics(env)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_folder = checkpoint_folder
        print("Training on: ", self.device)
        torch.set_default_device(self.device)
        self.env = env

        self.logger = MetricLogger()
        self.model = model_class(self.action_space_size, self.device, self.checkpoint_folder, epsilon_decay=0.9999996)


    def train_model(self, sync_every_n_steps, save_every_n_steps,
                    episodes,
                    train_every,
                    start_training_after,
                    max_repl_buffer_len,
                    lr,
                    batch_size,
                    record=False):

        self.batch_size = batch_size
        self.replay_buffer = ReplayBufferWeightedSampling(queue_max_len=max_repl_buffer_len, bin_size=20)

        if record:
            # Record each 250 episodes
            self.env = RecordVideo(self.env, video_folder="./recordings", name_prefix="mario-rl", episode_trigger=lambda e: e % 250 == 0)

        self.optimizer = self.opt_class(self.model.main_net.parameters(), lr=lr)

        info = None

        episodes_iter = tqdm(range(episodes), position=0, leave=True, miniters=100)

        current_state = self.env.reset()
        current_state = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).to(self.device)
        episodes_iter.set_description(f'Episode [{0}/{episodes_iter}]')

        step_count = 0

        for episode in episodes_iter:
            episodes_iter.set_description(f'Episode [{episode + 1}/{episodes}]')
            start_of_episode = True
            done = False
            info = {}
            while True:
                step_count +=1
                if episode % 150 == 0 and start_of_episode:
                    self.logger.record(episode=episode, epsilon=self.model.epsilon(step_count), step=step_count)

                if step_count % save_every_n_steps == 0:
                    self.model.save_checkpoint()

                if step_count % sync_every_n_steps == 0:
                    self.model.sync_target_net()

                if done or info.get("flag_get"):
                    done = False
                    current_state = self.env.reset()
                    current_state = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).to(self.device)
                    episodes_iter.set_postfix(avg_reward=info['episode']['r'])
                    self.logger.log_episode()
                    break

                action = self.model.get_next_action(current_state, step_count)
                next_state, reward, done, info = self.env.step(action)
                next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).to(self.device)

                x_pos = info["x_pos"]
                self.replay_buffer.append((current_state, torch.tensor(action), torch.tensor(reward), next_state, torch.tensor(done)), x_pos)

                # Update state
                current_state = next_state

                # Train every 2 steps
                if step_count > start_training_after and step_count % train_every == 0:
                    training_samples = self.replay_buffer.weighted_sample(k=self.batch_size, uniform=True)
                    current_states_b, actions_b, rewards_b, next_states_b, game_completed_b = tuple(map(torch.stack, zip(*training_samples)))
                    loss = self.model.compute_loss(current_states_b, actions_b, rewards_b, next_states_b, game_completed_b)

                    self.logger.log_step(reward=reward, loss=loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                start_of_episode = False


        self.env.close()
        self.model.save_checkpoint()
        print("Finished training and saved beste checkpoint!")


    def test_model(self, max_attempts_to_win):

        self.model.load_latest_checkpoint()
        recordings_dir = "test_logs"
        os.makedirs(recordings_dir, exist_ok=True)

        test_env = gym.make("SuperMarioBros-1-1-v3")
        test_env = JoypadSpace(test_env, ACTION_SPACE_MINIMAL)
        test_env = ResizeObservation(test_env, shape=(84,84))
        test_env = GrayScaleObservation(test_env)
        test_env = FrameStack(test_env, num_stack=4)
        test_env = RecordVideo(
            test_env,
            video_folder=recordings_dir,
            name_prefix="winning-run",
            episode_trigger=lambda _: True,
        )

        for attempt in tqdm(range(max_attempts_to_win), desc='Trying to complete word #1'):
            current_state = test_env.reset()
            current_state = torch.from_numpy(np.asarray(current_state, dtype=np.float32)).to(self.device)
            rewards = []
            while True:
                # Set extremely low exploration for testing!
                next_action = self.model.get_next_action(current_state, step_number=0, custom_epsilon=0.15)
                total_reward = 0
                next_state = None
                done = False
                # Manually apply the 4 skip steps so that the model acts only every 4 frames, but the recording is still complete!
                for _ in range(4):
                    # Accumulate reward and repeat the same action
                    next_state, reward, done, info = test_env.step(next_action)
                    total_reward += reward
                    if done and info.get('flag_get', False):
                        print(f'Winning run recorded in: {recordings_dir}')
                        test_env.close()
                        return
                    elif done:
                        break

                if done:
                    break

                reward = total_reward

                rewards.append(reward)
                current_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32)).to(self.device)

        test_env.close()
