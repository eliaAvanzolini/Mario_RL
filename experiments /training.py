from nes_py.wrappers import JoypadSpace
import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers.record_video import RecordVideo
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gym.wrappers.normalize import NormalizeObservation
from gym.wrappers.time_limit import TimeLimit
from torch.optim import AdamW
from tqdm import tqdm
from collections import deque
import torch
from models import RLModelInterface
import numpy as np
import copy
import random

class MarioRLTrainer():
    def __init__(self, epochs = 5000, opt = AdamW, max_episode_steps=1000):
        self.epochs = epochs
        self.opt_class = opt
        env = gym.make("SuperMarioBros-v0")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = ResizeObservation(env, shape=(84,84))
        env = GrayScaleObservation(env)
        env = FrameStack(env, num_stack=4)
        env = NormalizeObservation(env)
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
        env = RecordEpisodeStatistics(env)
        self.env = env

    def train_model(self, model: RLModelInterface, record=False, replay_buffer_length = 10000):
        self.model = model
        self.replay_buffer = deque(maxlen=replay_buffer_length)

        if record:
            self.env = RecordVideo(self.env, video_folder="../recordings", name_prefix="mario-rl", episode_trigger=lambda e: e % 10 == 0)

        self.optimizer = self.opt_class(model.parameters())

        info = None

        epochs_iter = tqdm(range(self.epochs))

        done = False
        current_state = self.env.reset()
        current_state = torch.from_numpy(np.asarray(current_state, dtype=np.float32))

        for epoch in epochs_iter:
            epochs_iter.set_description(f'Epoch [{epoch+1}/{self.epochs}]')

            for _ in range(100):
                if done:
                    done = False
                    current_state = self.env.reset()
                    current_state = torch.from_numpy(np.asarray(current_state, dtype=np.float32))
                    epochs_iter.set_postfix(avg_reward=info['episode']['r'])


                action = model.get_next_action(current_state, epoch).item()
                next_state, reward, done, info = self.env.step(action)

                next_state = torch.from_numpy(np.asarray(next_state, dtype=np.float32))

                # We need to differentiate between winning and just running out of time!
                game_completed = done and "TimeLimit.truncated" not in info.keys()

                self.replay_buffer.append((current_state, torch.tensor(action), torch.tensor(reward), next_state, torch.tensor(game_completed)))

                # Update state
                current_state = next_state



            if len(self.replay_buffer) > 512:

                for _ in range(5):
                    training_samples = random.choices(self.replay_buffer, k=64)
                    current_states_b, actions_b, rewards_b, next_states_b, game_completed_b = tuple(map(torch.stack, zip(*training_samples)))
                    loss = self.model.compute_loss(current_states_b, actions_b, rewards_b, next_states_b, game_completed_b)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()


        returns_over_time = np.array(self.env.return_queue)
        np.save("returns_over_time", returns_over_time)
        self.env.close()
