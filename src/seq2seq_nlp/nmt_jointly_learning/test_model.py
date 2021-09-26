from main import *


def test_model():
    print("Running Test")
    model.load_state_dict(torch.load("tut3-model.pt"))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")


if __name__ == "__main__":
    test_model()
