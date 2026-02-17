import torch.nn as nn
import torch
import random
import os
import time
from abc import ABC, abstractmethod
from torch.nn.functional import smooth_l1_loss
from datetime import datetime


# After many iterations we finally settled for the one used by Mnih et all in "https://arxiv.org/pdf/1312.5602"
class ConvMarioNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__() # Input size 4 x 84 x 84
        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4) # size 20 x 20
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2) # size 9 x 9
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1) # size 7 x 7
        self.fc_1 = nn.Linear(in_features = 7 * 7 * 64, out_features = 512)
        self.output = nn.Linear(in_features=512, out_features=out_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Normalize input:
        input = torch.div(input, 255.)
        # conv 1
        h = self.conv_1(input)
        h = self.relu(h)
        # conv 2
        h = self.conv_2(h)
        h = self.relu(h)
        # conv 3
        h = self.conv_3(h)
        h = self.relu(h)
        # Fc layer!
        h = h.reshape(-1, 7 * 7 * 64)
        h = self.fc_1(h)
        h = self.relu(h)
        return self.output(h)


class MarioInterface(ABC):
    def __init__(self, output_dim, device, checkpoint_dir, epsilon_decay = 0.99999975) -> None:
        self.main_net = ConvMarioNet(output_dim).to(device)
        self.output_dim = output_dim
        self.device = device

        self.target_net = ConvMarioNet(output_dim)
        self.target_net.load_state_dict(self.main_net.state_dict(), strict=False)
        self.target_net_update = True

        self.epsilon_start = 1.0
        self.epsilon_min = 0.10
        # with the current configuration we will reach epsilon_min at around 1000000 steps
        self.epsilon_decay = epsilon_decay

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self):
        now = datetime.now()
        now_str = now.strftime("%d-%m_%H-%M")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{now_str}.pt")
        torch.save({"model_state_dict": self.main_net.state_dict()}, checkpoint_path)
        return checkpoint_path

    def load_latest_checkpoint(self):
        checkpoint_files = [
            os.path.join(self.checkpoint_dir, file_name)
            for file_name in os.listdir(self.checkpoint_dir)
            if os.path.isfile(os.path.join(self.checkpoint_dir, file_name))
        ]

        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in folder: {self.checkpoint_dir}")

        newest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        checkpoint = torch.load(newest_checkpoint, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        self.main_net.load_state_dict(state_dict)
        return newest_checkpoint

    def sync_target_net(self):
        self.target_net.load_state_dict(self.main_net.state_dict(), strict=True)
        for param in self.target_net.parameters():
            param.requires_grad = False

    def get_next_action(self, current_state: torch.Tensor, step_number: int, custom_epsilon=None) -> int:
        epsilon = self.epsilon(step_number) if not custom_epsilon else custom_epsilon
        if random.random() < epsilon: # Exploration!
            next_action = random.randrange(self.output_dim)
        else:
            with torch.no_grad():
                preds = self.main_net(current_state)
                next_action = torch.argmax(preds, dim=1).item()

        return int(next_action)

    def epsilon(self, step_number: int) -> float:
        curr_epsilon = self.epsilon_start * self.epsilon_decay ** step_number
        return max(curr_epsilon, self.epsilon_min)

    @abstractmethod
    def compute_loss(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute the loss tensor given the state, action and reward"""
        pass


class DeepQLearning(MarioInterface):
    def __init__(self, output_dim, device, checkpoint_dir, epsilon_decay, gamma=0.9) -> None:
        super().__init__(output_dim, device, checkpoint_dir, epsilon_decay)
        self.gamma = gamma


    def compute_loss(self, state, action, reward, next_state, done) -> torch.Tensor:
        predicted_action_values = self.main_net(state)
        predicted_action_values = torch.gather(predicted_action_values, dim=1, index=action.unsqueeze(1)).squeeze(1)

        #Compute target detaching the tensor from the computational graph! (semi-semigradient)
        with torch.no_grad():
            q_target = torch.max(self.target_net(next_state), dim=1, keepdim=False)[0].detach()
        # Mask targets for terminal states
        q_target[done] = 0
        targets = reward + self.gamma * q_target

        return torch.pow(predicted_action_values - targets, 2).mean()


class DoubleDeepQLearning(MarioInterface):
    def __init__(self, output_dim, device, checkpoint_dir, epsilon_decay, gamma=0.9):
        super().__init__(output_dim, device, checkpoint_dir, epsilon_decay)
        self.gamma = gamma

    def compute_loss(self, state, action, reward, next_state, done) -> torch.Tensor:
        predicted_action_values = self.main_net(state)
        predicted_action_values = torch.gather(predicted_action_values, dim=1, index=action.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            #Compute target actions with the live network
            q_target_actions = torch.argmax(self.main_net(next_state), dim=1, keepdim=False).detach()
            # evaluate them with the target network
            q_target = self.target_net(next_state)
            q_target = torch.gather(q_target, dim=1, index=q_target_actions.unsqueeze(1)).squeeze(1)
        # Mask targets for terminal states
        q_target[done] = 0
        targets = reward + self.gamma * q_target

        # Use smooth huber loss instead of MSE
        # return torch.pow(predicted_action_values - targets, 2).mean()
        return smooth_l1_loss(predicted_action_values, targets)
