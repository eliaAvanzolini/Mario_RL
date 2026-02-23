import numpy as np
import time, datetime
import matplotlib.pyplot as plt
from pathlib import Path

# Logger class taken from https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html
class MetricLogger:
    def __init__(self, save_dir='train_logs'):
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        self.save_log = save_dir_path / "log"
        should_write_header = (not self.save_log.exists()) or self.save_log.stat().st_size == 0
        if should_write_header:
            with open(self.save_log, "a") as f:
                f.write(
                    f"{'Episode':>8}{'Step':>8}{'Epsilon':>10}{'MeanReward':>15}"
                    f"{'MeanLength':>15}{'MeanLoss':>15}{'MeanQValue':>15}"
                    f"{'TimeDelta':>15}{'Time':>20}\n"
                )
        self.ep_rewards_plot = save_dir_path / "reward_plot.jpg"
        self.ep_lengths_plot = save_dir_path / "length_plot.jpg"
        self.ep_avg_losses_plot = save_dir_path / "loss_plot.jpg"

        # CVPR-like plotting style
        plt.rcParams.update({
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "figure.figsize": (8, 5),
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 1.2,
            "font.size": 11,
            "font.family": "serif",
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "lines.linewidth": 2.4,
            "lines.markersize": 4,
        })
        self.plot_meta = {
            "ep_lengths": {
                "title": "Episode Length (Moving Avg)",
                "ylabel": "Length",
                "color": "#1f77b4",
            },
            "ep_avg_losses": {
                "title": "Episode Loss (Moving Avg)",
                "ylabel": "Loss",
                "color": "#d62728",
            },
            "ep_rewards": {
                "title": "Episode Reward (Moving Avg)",
                "ylabel": "Reward",
                "color": "#2ca02c",
            },
        }

        # History metrics
        self.ep_rewards = []
        self.ep_lengths = []
        self.ep_avg_losses = []

        # Moving averages, added for every call to record()
        self.moving_avg_episodes = []
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []

        # Current episode metric
        self.init_episode()

        # Timing
        self.record_time = time.time()

    def log_step(self, reward, loss):
        self.curr_ep_reward += reward
        self.curr_ep_length += 1
        if loss:
            self.curr_ep_loss += loss.item()
            self.curr_ep_loss_length += 1

    def log_episode(self):
        "Mark end of episode"
        self.ep_rewards.append(self.curr_ep_reward)
        self.ep_lengths.append(self.curr_ep_length)
        if self.curr_ep_loss_length == 0:
            ep_avg_loss = 0
        else:
            ep_avg_loss = np.round(self.curr_ep_loss / self.curr_ep_loss_length, 5)
        self.ep_avg_losses.append(ep_avg_loss)

        self.init_episode()

    def init_episode(self):
        self.curr_ep_reward = 0.0
        self.curr_ep_length = 0
        self.curr_ep_loss = 0.0
        self.curr_ep_q = 0.0
        self.curr_ep_loss_length = 0

    def record(self, episode, epsilon, step):
        mean_ep_reward = np.round(np.mean(self.ep_rewards[-100:]), 3)
        mean_ep_length = np.round(np.mean(self.ep_lengths[-100:]), 3)
        mean_ep_loss = np.round(np.mean(self.ep_avg_losses[-100:]), 3)
        self.moving_avg_episodes.append(int(episode))
        self.moving_avg_ep_rewards.append(mean_ep_reward)
        self.moving_avg_ep_lengths.append(mean_ep_length)
        self.moving_avg_ep_avg_losses.append(mean_ep_loss)

        last_record_time = self.record_time
        self.record_time = time.time()
        time_since_last_record = np.round(self.record_time - last_record_time, 3)

        print(
            f"Episode {episode} - "
            f"Step {step} - "
            f"Epsilon {epsilon} - "
            f"Mean Reward {mean_ep_reward} - "
            f"Mean Length {mean_ep_length} - "
            f"Mean Loss {mean_ep_loss} - "
            f"Time Delta {time_since_last_record} - "
            f"Time {datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}"
        )

        with open(self.save_log, "a") as f:
            f.write(
                f"{episode:8d}{step:8d}{epsilon:10.3f}"
                f"{mean_ep_reward:15.3f}{mean_ep_length:15.3f}{mean_ep_loss:15.3f}"
                f"{time_since_last_record:15.3f}"
                f"{datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'):>20}\n"
            )

        self._plot_moving_averages()

    def _plot_moving_averages(self):
        for metric in ["ep_lengths", "ep_avg_losses", "ep_rewards"]:
            values = getattr(self, f"moving_avg_{metric}")
            meta = self.plot_meta[metric]
            if len(self.moving_avg_episodes) == len(values):
                episode_idx = np.asarray(self.moving_avg_episodes, dtype=np.int32)
            else:
                print("defaulting to stuff!!")
                episode_idx = np.arange(1, len(values) + 1)

            fig, ax = plt.subplots()
            ax.plot(episode_idx, values, color=meta["color"], label="100-Episode Moving Average")
            ax.set_title(meta["title"], pad=10)
            ax.set_xlabel("Episode")
            ax.set_ylabel(meta["ylabel"])
            ax.grid(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(getattr(self, f"{metric}_plot"), bbox_inches="tight")
            plt.close(fig)

    def regenerate_plots_from_log(self):
        self.moving_avg_episodes = []
        self.moving_avg_ep_rewards = []
        self.moving_avg_ep_lengths = []
        self.moving_avg_ep_avg_losses = []

        if not self.save_log.exists():
            return

        with open(self.save_log, "r") as f:
            lines = f.readlines()

        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                self.moving_avg_episodes.append(int(parts[0]))
                self.moving_avg_ep_rewards.append(float(parts[3]))
                self.moving_avg_ep_lengths.append(float(parts[4]))
                self.moving_avg_ep_avg_losses.append(float(parts[5]))
            except ValueError:
                continue

        self._plot_moving_averages()
