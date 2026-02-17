from deep_models import DoubleDeepQLearning
from training import DeepMarioTrainer, TabularMarioTrainer


def main():
    TRAIN_MODE = True
    TRAINER_TYPE = "tabular"  # "deep" or "tabular"

    checkpoints_path = f"{TRAINER_TYPE}_model_checkpoints"

    if TRAINER_TYPE == "deep":
        deep_trainer = DeepMarioTrainer(
            model_class=DoubleDeepQLearning,
            output_folder="deep_training",
        )
        if TRAIN_MODE:
            deep_trainer.train_model(sync_every_n_steps=10_000,
                                     save_every_n_steps=500_000,
                                     start_training_after=100_000,
                                     record=True,
                                     episodes=20_000,
                                     train_every=3,
                                     lr=0.00025,
                                     max_repl_buffer_len=100_000,
                                     batch_size=32)
        else:
            deep_trainer.test_model(max_attempts_to_win=100)

    elif TRAINER_TYPE == "tabular":
        tabular_trainer = TabularMarioTrainer(
            output_folder="tabular_training",
            use_sarsa=False,
            bin_size=16,
            gamma=0.99,
            learning_rate=0.10,
            epsilon_decay=0.999995,
        )
        if TRAIN_MODE:
            tabular_trainer.train_model(
                episodes=25_000,
                save_every_n_steps=250_000,
                max_repl_buffer_len=100_000,
                record=True,
            )
        else:
            tabular_trainer.test_model(max_attempts_to_win=100)

    else:
        raise ValueError("TRAINER_TYPE must be either 'deep' or 'tabular'.")

if __name__ == "__main__":
    main()
