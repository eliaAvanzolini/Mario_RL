import torch.nn as nn
import torch


class ConvMarioNet(nn.Module):
    def __init__(self):
        super().__init__() # Input size 4 x 84 x 84
        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=2, padding=1) # size 42 x 42
        self.max_pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # size 21 x 21
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, stride=2, padding=1) # size 11 x 11
        self.max_pool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # size 6 x 6
        self.conv_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride = 1, padding=0) # size 4 x 4
        # self.dropout = nn.Dropout2d(p=0.2)
        self.fc_1 = nn.Linear(in_features = 4 * 4 * 32, out_features = 256)
        self.relu = nn.ReLU()
        self.output = nn.Linear(in_features=256, out_features=7)

    def forward(self, input):
        # conv 1
        h = self.conv_1(input)
        h = self.relu(h)
        h = self.max_pool_1(h)
        # conv 2
        h = self.conv_2(h)
        h = self.relu(h)
        h = self.max_pool_2(h)
        # conv 3
        h = self.conv_3(h)
        h = self.relu(h)
        # h = self.dropout(h)
        # Fc layer!
        h = h.reshape(-1, 4 * 4 * 32)
        h = self.fc_1(h)
        h = self.relu(h)
        return self.output(h)


class RLModelInterface(ConvMarioNet):
    def __init__(self) -> None:
        super().__init__()

    def get_next_action(self, current_state: torch.Tensor, epoch_number: int) -> torch.Tensor:
        """Computes the next action to take according to the exploration policy if different from the evaluation policy"""

        pass

    def compute_loss(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        """Compute the loss tensor given the state, action and reward"""

        pass

class DeepQLearning(RLModelInterface):
    def __init__(self, exploration_decay = None, epsilon = 0.4, update_policy_net_every = 20, gamma=0.99):
        super().__init__()
        self.update_policy_net_every = update_policy_net_every
        self.exploration_decay = exploration_decay
        self.epsilon = lambda epoch: epsilon if exploration_decay is None else max(epsilon - (epoch) / 1000., epsilon / 10.)
        self.gamma = gamma
        self.target_net = ConvMarioNet()
        self.policy_net_updated = True

    def get_next_action(self, current_state: torch.Tensor, epoch_number: int) -> torch.Tensor:
        if current_state.shape == (4, 84, 84):
            current_state = current_state.unsqueeze(0)

        if epoch_number % self.update_policy_net_every == 0 and not self.policy_net_updated:
            self.target_net.load_state_dict(self.state_dict(), strict=False)
            self.policy_net_updated = True
        elif epoch_number % self.update_policy_net_every != 0 and self.policy_net_updated:
            self.policy_net_updated = False

        epsilon = self.epsilon(epoch_number)
        rand_mask = torch.rand((current_state.shape[0])) < epsilon
        rand_actions = torch.randint(low=0, high=7, size=(current_state.shape[0], ))

        with torch.no_grad():
            preds = self.forward(current_state)

        pred_actions = torch.argmax(preds, dim=1)

        pred_actions[rand_mask] = rand_actions[rand_mask]

        return pred_actions

    def compute_loss(self, state, action, reward, next_state, done) -> torch.Tensor:

        predicted_action_values = self.forward(state)

        predicted_action_values = torch.gather(predicted_action_values, dim=1, index=action.unsqueeze(1)).squeeze(1)

        #Compute target detaching the tensor from the computational graph! (semi-semigradient)
        with torch.no_grad():
            q_target = torch.max(self.target_net(next_state), dim=1, keepdim=False)[0].detach()
        # Mask targets for terminal states
        q_target[done] = 0
        targets = reward + self.gamma * q_target

        return torch.pow(predicted_action_values - targets, 2).mean()
