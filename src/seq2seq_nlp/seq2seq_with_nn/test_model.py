from main import *
from utils.nn_model import *
from utils.preprocess import *


def test_model():
    print("Test Trained & Validated Model")
    model.load_state_dict(torch.load("seq2seq-model.pt"))

    test_loss = evaluate(model, test_iterator, criterion=criterion)
    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")


if __name__ == "__main__":
    test_model()
