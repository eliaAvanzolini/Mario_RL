from datetime import datetime
import os
import numpy as np
import random

class TabularQLearning:
    def __init__(self, output_dim, max_x_dim, max_y_dim, bin_size, checkpoint_dir, gamma, learning_rate, epsilon_decay = 0.9999) -> None:
        max_x_bin = max_x_dim // bin_size
        max_y_bin = max_y_dim // bin_size
        self.policy_matrix = np.full((max_x_bin, max_y_bin, output_dim), 0, dtype=np.float32)
        self.output_dim = output_dim

        self.epsilon_start = 1.0
        self.epsilon_min = 0.10
        # with the current configuration we will reach epsilon_min at around 1000000 steps
        self.epsilon_decay = epsilon_decay

        self.gamma = np.float32(gamma)
        self.learning_rate = np.float32(learning_rate)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self):
        now = datetime.now()
        now_str = now.strftime("%d-%m_%H-%M")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{now_str}")

        np.save(checkpoint_path, self.policy_matrix)
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
        self.policy_matrix = np.load(newest_checkpoint)

    def get_next_action(self, x_pos, y_pos, step_number: int, custom_epsilon=None) -> int:
        epsilon = self.epsilon(step_number) if custom_epsilon is None else custom_epsilon
        if random.random() < epsilon: # Exploration!
            next_action = random.randrange(self.output_dim)
        else:
            next_action = np.argmax(self.policy_matrix[x_pos, y_pos])

        return int(next_action)

    def epsilon(self, step_number: int) -> float:
        curr_epsilon = self.epsilon_start * self.epsilon_decay ** step_number
        return max(curr_epsilon, self.epsilon_min)

    def update_value(self, x_pos, y_pos, action, reward, next_x, next_y, done, use_sarsa=False, epsilons=None):
        if use_sarsa:
            if epsilons is None:
                raise ValueError("epsilons must be provided when use_sarsa=True")
            self.update_sarsa_value(x_pos, y_pos, action, reward, next_x, next_y, done, epsilons)
            return
        self.update_q_value(x_pos, y_pos, action, reward, next_x, next_y, done)

    def update_q_value(self, x_pos, y_pos, action, reward, next_x, next_y, done):
        predicted_action_values = self.policy_matrix[x_pos, y_pos, action]

        #Compute the future reward estimate (off policy!!)
        q_target = np.max(self.policy_matrix[next_x, next_y], axis=1).astype(np.float32, copy=False)
        # Mask targets for terminal states
        q_target[done] = 0
        reward = reward.astype(np.float32, copy=False)
        targets = reward + self.gamma * q_target
        td_error = targets - predicted_action_values

        # Update estimates!
        self.policy_matrix[x_pos, y_pos, action] += self.learning_rate * td_error
        return np.mean(np.abs(td_error))

    def update_sarsa_value(self, x_pos, y_pos, action, reward, next_x, next_y, done, epsilons):
        predicted_action_values = self.policy_matrix[x_pos, y_pos, action]

        #Compute the future reward estimate (on policy!)
        chosen_actions = np.argmax(self.policy_matrix[next_x, next_y], axis=1)

        random_mask = np.random.random_sample(epsilons.shape) < epsilons

        rand_actions = np.random.randint(self.output_dim, size=(random_mask.sum()))
        chosen_actions[random_mask] = rand_actions

        # Now set the SARSA target with the chosen actions
        q_target = self.policy_matrix[next_x, next_y, chosen_actions]
        # Mask targets for terminal states
        q_target[done] = 0

        reward = reward.astype(np.float32, copy=False)
        targets = reward + self.gamma * q_target
        td_error = targets - predicted_action_values

        # Update estimates!
        self.policy_matrix[x_pos, y_pos, action] += self.learning_rate * td_error
        return np.mean(np.abs(td_error))
