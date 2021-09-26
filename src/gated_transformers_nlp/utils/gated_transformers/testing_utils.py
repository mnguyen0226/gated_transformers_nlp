# script testing trained gated transformers model

import math
import torch
from typing import Tuple
from utils.gated_transformers.training_utils import model, evaluate, criterion
from utils.gated_transformers.preprocess import test_iterator


def test_gated_transformers_model() -> Tuple[float, float]:
    """Tests the trained gated transformers model

    Return
    ----------
    test_loss:
        Testing loss
    math.exp(test_loss):
        Testing PPL
    """
    model.load_state_dict(
        torch.load("gated-tut6-model.pt", map_location=torch.device("cpu"))
    )

    test_loss = evaluate(model, test_iterator, criterion)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")

    return test_loss, math.exp(test_loss)
