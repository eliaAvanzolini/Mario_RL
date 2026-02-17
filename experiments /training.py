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
from deep_models import DeepMarioInterface, DoubleDeepQLearning
from tabular_models import TabularQLearning
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


class DeepMarioTrainer():
    def __init__(self, model_class: type[DeepMarioInterface],
                 opt = Adam,
                 output_folder="deep_training"):
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

        self.output_folder = output_folder
        self.checkpoint_folder = os.path.join(self.output_folder, "checkpoint")
        self.logs_folder = os.path.join(self.output_folder, "logs")
        self.recordings_folder = os.path.join(self.logs_folder, "recordings")
        self.test_logs_folder = os.path.join(self.logs_folder, "test_logs")

        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.logs_folder, exist_ok=True)
        os.makedirs(self.recordings_folder, exist_ok=True)
        os.makedirs(self.test_logs_folder, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Training on: ", self.device)
        torch.set_default_device(self.device)
        self.env = env

        self.logger = MetricLogger(save_dir=self.logs_folder)
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
            self.env = RecordVideo(self.env, video_folder=self.recordings_folder, name_prefix="mario-rl", episode_trigger=lambda e: e % 250 == 0)

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
        recordings_dir = self.test_logs_folder
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


class TabularMarioTrainer:
    def __init__(self,
                 output_folder="tabular_training",
                 use_sarsa=False,
                 bin_size=16,
                 gamma=0.99,
                 learning_rate=0.10,
                 epsilon_decay=0.999995,
                 action_space=None):
        env = gym.make("SuperMarioBros-1-1-v3")

        self.action_space = action_space if action_space is not None else ACTION_SPACE_MINIMAL
        self.action_space_size = len(self.action_space)
        env = JoypadSpace(env, self.action_space)
        env = SkipFrame(env, skip=4)
        env = RecordEpisodeStatistics(env)

        self.output_folder = output_folder
        self.checkpoint_folder = os.path.join(self.output_folder, "checkpoint")
        self.logs_folder = os.path.join(self.output_folder, "logs")
        self.recordings_folder = os.path.join(self.logs_folder, "recordings")
        self.test_logs_folder = os.path.join(self.logs_folder, "test_logs")

        os.makedirs(self.checkpoint_folder, exist_ok=True)
        os.makedirs(self.logs_folder, exist_ok=True)
        os.makedirs(self.recordings_folder, exist_ok=True)
        os.makedirs(self.test_logs_folder, exist_ok=True)

        self.env = env
        self.logger = MetricLogger(save_dir=self.logs_folder)
        self.use_sarsa = use_sarsa
        self.bin_size = bin_size

        self.max_x_dim = 4096
        self.max_y_dim = 256

        self.model = TabularQLearning(
            output_dim=self.action_space_size,
            max_x_dim=self.max_x_dim,
            max_y_dim=self.max_y_dim,
            bin_size=self.bin_size,
            checkpoint_dir=self.checkpoint_folder,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_decay=epsilon_decay,
        )

    def _to_bins(self, x_pos: int, y_pos: int):
        max_x_idx = (self.max_x_dim // self.bin_size) - 1
        max_y_idx = (self.max_y_dim // self.bin_size) - 1
        x_bin = int(np.clip(x_pos // self.bin_size, 0, max_x_idx))
        y_bin = int(np.clip(y_pos // self.bin_size, 0, max_y_idx))
        return x_bin, y_bin

    def train_model(self,
                    episodes=25_000,
                    save_every_n_steps=250_000,
                    record=False,
                    max_repl_buffer_len=100_000,
                    batch_size=128,
                    train_every=4,
                    start_training_after=10_000):

        if record:
            self.env = RecordVideo(
                self.env,
                video_folder=self.recordings_folder,
                name_prefix="mario-tabular",
                episode_trigger=lambda e: e % 500 == 0,
            )

        episodes_iter = tqdm(range(episodes), position=0, leave=True, miniters=100)
        episodes_iter.set_description(f'Episode [{0}/{episodes}]')

        step_count = 0

        repl_buffer = deque(maxlen=max_repl_buffer_len)

        for episode in episodes_iter:
            episodes_iter.set_description(f'Episode [{episode + 1}/{episodes}]')
            if episode % 100 == 0:
                self.logger.record(episode=episode, epsilon=self.model.epsilon(step_count), step=step_count)

            _ = self.env.reset()
            done = False
            info = {"x_pos": 0, "y_pos": 0}

            x_bin, y_bin = self._to_bins(info.get("x_pos", 0), info.get("y_pos", 0))
            action = self.model.get_next_action(x_bin, y_bin, step_count)

            while True:
                step_count += 1

                if step_count % save_every_n_steps == 0:
                    self.model.save_checkpoint()

                next_obs, reward, done, info = self.env.step(action)
                del next_obs
                next_x_bin, next_y_bin = self._to_bins(info.get("x_pos", 0), info.get("y_pos", 0))

                curr_eps = self.model.epsilon(step_count)
                repl_buffer.append((x_bin, y_bin, action, reward, next_x_bin, next_y_bin, done, curr_eps))

                next_action = self.model.get_next_action(next_x_bin, next_y_bin, step_count)

                if (
                    step_count > start_training_after
                    and len(repl_buffer) >= batch_size
                    and step_count % train_every == 0
                ):
                    sample_idx = np.random.choice(len(repl_buffer), size=batch_size, replace=False)
                    samples = [repl_buffer[idx] for idx in sample_idx]
                    x_b, y_b, a_b, r_b, nx_b, ny_b, d_b, eps_b = map(np.asarray, zip(*samples))

                    td_loss = self.model.update_value(
                        x_pos=x_b,
                        y_pos=y_b,
                        action=a_b,
                        reward=r_b.astype(np.float32),
                        next_x=nx_b,
                        next_y=ny_b,
                        done=d_b.astype(bool),
                        use_sarsa=self.use_sarsa,
                        epsilons=eps_b.astype(np.float32),
                    )
                    self.logger.log_step(reward=0.0, loss=np.float32(td_loss))

                self.logger.log_step(reward=reward, loss=None)

                x_bin, y_bin = next_x_bin, next_y_bin
                action = next_action

                if done or info.get("flag_get"):
                    self.logger.log_episode()
                    episode_reward = info.get("episode", {}).get("r")
                    if episode_reward is not None:
                        episodes_iter.set_postfix(avg_reward=episode_reward)
                    break

        self.env.close()
        self.model.save_checkpoint()
        print("Finished tabular training and saved latest checkpoint.")

    def test_model(self, max_attempts_to_win=100, test_epsilon=0.05):
        self.model.load_latest_checkpoint()
        recordings_dir = self.test_logs_folder
        os.makedirs(recordings_dir, exist_ok=True)

        test_env = gym.make("SuperMarioBros-1-1-v3")
        test_env = JoypadSpace(test_env, self.action_space)
        test_env = RecordVideo(
            test_env,
            video_folder=recordings_dir,
            name_prefix="tabular-winning-run",
            episode_trigger=lambda _: True,
        )

        for _ in tqdm(range(max_attempts_to_win), desc='Trying to complete world #1'):
            _ = test_env.reset()
            done = False
            info = {"x_pos": 0, "y_pos": 0}

            while True:
                x_bin, y_bin = self._to_bins(info.get("x_pos", 0), info.get("y_pos", 0))
                next_action = self.model.get_next_action(
                    x_bin,
                    y_bin,
                    step_number=0,
                    custom_epsilon=test_epsilon,
                )

                total_reward = 0
                for _ in range(4):
                    _, reward, done, info = test_env.step(next_action)
                    total_reward += reward
                    if done and info.get('flag_get', False):
                        print(f'Winning run recorded in: {recordings_dir}')
                        test_env.close()
                        return
                    elif done:
                        break

                del total_reward

                if done:
                    break

        test_env.close()
