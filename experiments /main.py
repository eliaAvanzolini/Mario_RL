import argparse
import io
import os
import shutil
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import IO, cast

from deep_models import DeepQLearning, DoubleDeepQLearning
from training import DeepMarioTrainer, TabularMarioTrainer


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def writable(self):
        return True

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _non_empty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError("run name cannot be empty")
    return stripped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mario RL runner")
    parser.add_argument("--trainer", choices=["deep", "tabular"], required=True)
    parser.add_argument("--sub", choices=["standard", "double"])
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    parser.add_argument("--run-name", type=_non_empty, required=True)
    parser.add_argument("--runs-root", default=".")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-sarsa", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.trainer == "deep" and args.sub is None:
        parser.error("--sub is required when --trainer deep (choose: standard or double)")
    if args.trainer != "deep" and args.sub is not None:
        parser.error("--sub can only be used with --trainer deep")

    # ==========================
    # HARDCODED RUN CONFIGURATION
    # ==========================
    COMMON_CONFIG = {
        "record": True,
        "max_attempts_to_win": 100,
    }

    DEEP_CONFIG = {
        "sync_every_n_steps": 5_000,
        "save_every_n_steps": 100_000,
        "start_training_after": 32,
        "episodes": 10_000,
        "train_every": 1,
        "lr": 0.00025,
        "max_repl_buffer_len": 30_000,
        "batch_size": 32,
        "eps_decay": 0.99,
    }

    TABULAR_CONFIG = {
        "bin_size": 16,
        "gamma": 0.99,
        "learning_rate": 0.10,
        "epsilon_decay": 0.999995,
        "episodes": 25_000,
        "save_every_n_steps": 250_000,
        "max_repl_buffer_len": 100_000,
        "batch_size": 128,
        "train_every": 4,
        "start_training_after": 10_000,
    }

    trainer_root = os.path.join(args.runs_root, f"{args.trainer}_training")
    run_folder = os.path.join(trainer_root, args.run_name)

    if args.mode == "train":
        if os.path.exists(run_folder) and not args.overwrite:
            raise FileExistsError(
                f"Run folder already exists: {run_folder}. Use --overwrite to replace it."
            )
        if os.path.exists(run_folder) and args.overwrite:
            shutil.rmtree(run_folder)
        os.makedirs(run_folder, exist_ok=True)
    else:
        if not os.path.exists(run_folder):
            raise FileNotFoundError(
                f"Run folder not found: {run_folder}. For testing, provide an existing run name."
            )

    run_log_path = os.path.join(run_folder, "run.log")
    with open(run_log_path, "a" if args.mode == "test" else "w") as run_log:
        tee_out = Tee(sys.stdout, run_log)
        tee_err = Tee(sys.stderr, run_log)
        with redirect_stdout(cast(IO[str], tee_out)), redirect_stderr(cast(IO[str], tee_err)):
            if args.trainer == "deep":
                deep_model_class = DoubleDeepQLearning if args.sub == "double" else DeepQLearning
                trainer = DeepMarioTrainer(
                    model_class=deep_model_class,
                    output_folder=run_folder,
                )
                if args.mode == "train":
                    trainer.train_model(
                        sync_every_n_steps=DEEP_CONFIG["sync_every_n_steps"],
                        save_every_n_steps=DEEP_CONFIG["save_every_n_steps"],
                        start_training_after=DEEP_CONFIG["start_training_after"],
                        record=COMMON_CONFIG["record"],
                        episodes=DEEP_CONFIG["episodes"],
                        train_every=DEEP_CONFIG["train_every"],
                        lr=DEEP_CONFIG["lr"],
                        max_repl_buffer_len=DEEP_CONFIG["max_repl_buffer_len"],
                        batch_size=DEEP_CONFIG["batch_size"],
                        eps_decay=DEEP_CONFIG["eps_decay"],
                    )
                else:
                    trainer.test_model(max_attempts_to_win=COMMON_CONFIG["max_attempts_to_win"])
                return
            elif args.trainer == "tabular":
                trainer = TabularMarioTrainer(
                    output_folder=run_folder,
                    use_sarsa=args.use_sarsa,
                    bin_size=TABULAR_CONFIG["bin_size"],
                    gamma=TABULAR_CONFIG["gamma"],
                    learning_rate=TABULAR_CONFIG["learning_rate"],
                    epsilon_decay=TABULAR_CONFIG["epsilon_decay"],
                )
                if args.mode == "train":
                    trainer.train_model(
                        episodes=TABULAR_CONFIG["episodes"],
                        save_every_n_steps=TABULAR_CONFIG["save_every_n_steps"],
                        max_repl_buffer_len=TABULAR_CONFIG["max_repl_buffer_len"],
                        batch_size=TABULAR_CONFIG["batch_size"],
                        train_every=TABULAR_CONFIG["train_every"],
                        start_training_after=TABULAR_CONFIG["start_training_after"],
                        record=COMMON_CONFIG["record"],
                    )
                else:
                    trainer.test_model(max_attempts_to_win=COMMON_CONFIG["max_attempts_to_win"])


if __name__ == "__main__":
    main()
