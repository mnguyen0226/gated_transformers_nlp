# Script trainining gated transformers model

import torch
import torch.nn as nn

from typing import Tuple
import math
import time

from utils.gated_transformers.seq2seq import Seq2Seq
from utils.gated_transformers.preprocess import (
    SRC,
    TRG,
    device,
    train_iterator,
    valid_iterator,
)
from utils.gated_transformers.encoder import Encoder
from utils.gated_transformers.decoder import Decoder


# Define encoder and decoder
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
GATED_ENC_LAYERS = 3
GATED_DEC_LAYERS = 3
GATED_ENC_HEADS = 8
GATED_DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(
    INPUT_DIM,
    HID_DIM,
    GATED_ENC_LAYERS,
    GATED_ENC_HEADS,
    ENC_PF_DIM,
    ENC_DROPOUT,
    device,
)

dec = Decoder(
    OUTPUT_DIM,
    HID_DIM,
    GATED_DEC_LAYERS,
    GATED_DEC_HEADS,
    DEC_PF_DIM,
    DEC_DROPOUT,
    device,
)

# Define whole Seq2Seq encapsulating model
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


def count_parameters(model: Tuple[tuple, tuple, tuple, tuple, str]) -> int:
    """Check number of training parameters

    Parameters
    ----------
    model: [tuple, tuple, tuple, tuple, str]
        input seq2seq model

    Return
    ----------
    Total number of training parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")


def initialize_weights(m: Tuple[tuple, tuple, tuple, tuple, str]):
    """Xavier uniform initialization

    Parameters
    ----------
    m: [tuple, tuple, tuple, tuple, str]
        input model
    """
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


model.apply(initialize_weights)

LEARNING_RATE = 0.0005

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Cross Entropy Loss Function
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(
    model: Tuple[tuple, tuple, tuple, tuple, str],
    iterator: int,
    optimizer: int,
    criterion: int,
    clip: int,
) -> float:
    """Train by calculating losses and update parameters

    Parameters
    ----------
    model: [tuple, tuple, tuple, tuple, str]
        input seq2seq model
    iterator: int
        SRC, TRG iterator
    optimizer: int
        Adam optimizer
    criterion: int
        Cross Entropy Loss function
    clip: int
        Clip training process

    Return
    ----------
    epoch_loss / len(iterator): float
        Loss percentage during training
    """
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator: int, criterion: int) -> float:
    """Evaluate same as training but no gradient calculation and parameter updates

    Parameters
    ----------
    iterator: int
        SRC, TRG iterator
    criterion: int
        Cross Entropy Loss function

    Return
    ----------
    epoch_loss / len(iterator): float
        Loss percentage during validating
    """

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])
            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: float, end_time: float) -> Tuple[int, int]:
    """Tells how long an epoch takes

    Parameters
    ----------
    start_time:
        start time
    end_time:
        end_time

    Return
    ----------
    elapsed_mins: float
        elapse minutes
    elapsed_secs: float
        elapse seconds
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
train_loss = 0
valid_loss = 0


def gated_transformers_main() -> Tuple[float, float, float, float]:
    """Run Training and Evaluating procedure

    Return
    ----------
    train_loss: float
        training loss of the current epoch
    valid_loss: float
        validating loss of the current epoch
    math.exp(train_loss): float
        training PPL
    math.exp(valid_loss): float
        validating PPL
    """
    best_valid_loss = float("inf")

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "gated-tut6-model.pt")

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
        )

    return train_loss, valid_loss, math.exp(train_loss), math.exp(valid_loss)
