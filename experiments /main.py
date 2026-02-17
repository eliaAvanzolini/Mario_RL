from models import DeepQLearning, DoubleDeepQLearning
from training import MarioTrainer


def main():
    TRAIN_MODE = False
    checkpoints_path = "model_checkpoints"
    trainer = MarioTrainer(model_class=DoubleDeepQLearning,
                           checkpoint_folder=checkpoints_path)
    if TRAIN_MODE:
        trainer.train_model(sync_every_n_steps=10_000,
                            save_every_n_steps=100_000,
                            start_training_after=10_000,
                            record=True,
                            episodes=20_000,
                            train_every=3,
                            lr=0.00025,
                            max_repl_buffer_len=70_000,
                            batch_size=32)

    else:
        trainer.test_model(max_attempts_to_win=100)

if __name__ == "__main__":
    main()
