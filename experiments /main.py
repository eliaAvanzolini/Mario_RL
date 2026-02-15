from models import DeepQLearning
from training import MarioRLTrainer


def main():
    deep_q_model = DeepQLearning()
    trainer = MarioRLTrainer(epochs=500)

    trainer.train_model(deep_q_model, record=True)


if __name__ == "__main__":
    main()
