import numpy as np
import matplotlib.pyplot as plt
import os

# Update paths relative to where this script is executed
path_ql = "models/QL/"
path_sarsa = "models/SARSA/" 
path_ddqn = "models/DDQN/"

# Function to load rewards safely
def load_rewards(path, filename="rewards.npy"):
    full_path = os.path.abspath(os.path.join(path, filename))
    if os.path.exists(full_path):
        return np.load(full_path)
    else:
        print(f"Warning: Rewards file not found at {full_path}")
        return np.array([])
        
def load_ddqn_rewards(path, filename="total_rewards.pkl"):
    full_path = os.path.abspath(os.path.join(path, filename))
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            return np.array(pickle.load(f))
    else:
        print(f"Warning: DDQN rewards file not found at {full_path}")
        return np.array([])

# Load rewards from files
rewards_ql = load_rewards(path_ql)
rewards_sarsa = load_rewards(path_sarsa)
rewards_ddqn = load_ddqn_rewards(path_ddqn)


# Plotting function
def plot_agent_rewards(rewards, label, window_size=100):
    if len(rewards) >= window_size:
        # Calculate the moving average
        smoothed_rewards = np.convolve(rewards, np.ones((window_size,))/window_size, mode="valid").tolist()
        plt.plot(smoothed_rewards, label=label)
    elif len(rewards) > 0:
        plt.plot(rewards, label=f"{label} (Raw Data)")
    else:
        print(f"Skipping plot for {label}: Not enough data.")


# --- Plotting ---
plt.figure(figsize=(12, 6))
plt.title("Episodes trained vs. Average Rewards")
plt.xlabel("Episodes")
plt.ylabel(f"Average Rewards (Smoothed over 100 eps)")

plot_agent_rewards(rewards_ql[:5000], "Q-Learning", window_size=100) # Assuming QL has been run for more episodes
plot_agent_rewards(rewards_sarsa[:5000], "SARSA", window_size=100)
plot_agent_rewards(rewards_ddqn[:5000], "Double DQN", window_size=100)

plt.legend()
plt.grid(True)
plt.show()