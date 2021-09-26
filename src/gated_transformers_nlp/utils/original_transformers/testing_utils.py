# script test trained original transformers model

import math
import torch
from typing import Tuple
from utils.original_transformers.training_utils import model, evaluate, criterion
from utils.original_transformers.preprocess import test_iterator


def test_origin_transformers_model() -> Tuple[float, float]:
    """Tests the trained origin transformers model

    Return
    ----------
    test_loss:
        Testing loss
    math.exp(test_loss):
        Testing PPL
    """
    model.load_state_dict(torch.load("original-tut6-model.pt"))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |")

    return test_loss, math.exp(test_loss)
