""" 
Previously we have only cared about the loss/perplexity of the model. However there metrics that are specifically designed for measuring the quality of a translation - the most popular is BLEU.
BLEU looks at the overlap in the predicted and actual target sequences in terms of their n-grams. It will give us a number between 0 and 1 for each sequence, where 1 means there is perfect overlap, i.e. a perfect translation, although is usually shown between 0 and 100. BLEU was designed for multiple candidate translations per source sequence, however in this dataset we only have one candidate per source.

We define a calculate_bleu function which calculates the BLEU score over a provided TorchText dataset. This function creates a corpus of the actual and predicted translation for each source sentence and then calculates the BLEU score.
"""
from main import *
from inference import *

from torchtext.data.metrics import bleu_score

model.load_state_dict(torch.load("tut4-model.pt", map_location=torch.device("cpu")))


def calculate_bleu(data, src_field, trg_field, model, device, max_len=50):

    trgs = []
    pred_trgs = []

    for datum in data:

        src = vars(datum)["src"]
        trg = vars(datum)["trg"]

        pred_trg, _ = translate_sentence(
            src, src_field, trg_field, model, device, max_len
        )

        # cut off <eos> token
        pred_trg = pred_trg[:-1]

        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)


def bleu():
    """
    This number isn't really interpretable, we can't really say much about it. The most useful part of a BLEU score is that it can be used to compare different models on the same dataset, where the one with the higher BLEU score is "better".
    """
    print("Running")
    bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

    print(f"BLEU score = {bleu_score*100:.2f}")


if __name__ == "__main__":
    bleu()
