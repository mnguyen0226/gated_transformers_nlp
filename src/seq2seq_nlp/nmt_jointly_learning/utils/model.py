"""
ENCODER:
- Single GRU, but we will use bidirectional RNN
- Bidirectional RNN = have 2 RNNs in each layers.
    + forward RNN goes over the embedded sentence from left to right
    + backward RNN goes over the embbedd sentence from right to left
    + bidirectional
- We pass input embedded to RNN which tell Pytorch to initialize both the fw and bw initial hidden state to a tensor of all zeros.
- We will get 2 context vector, 1 for forward RNN after it has seen the final word in a sentence, and one from the backward rnn after it has seen the frist word in the senten


"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        # sropout = dropout rate
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(
            dropout
        )  # during training, randomly zeroes some of the elements of the input tensor with proability p using samples from Bernoulli distribution. This has proven to be an effective technique for regularization and preventing the coadaptation of neurons

    def forward(self, src):
        # src = [src len, batch size]
        embedded = self.dropout(
            self.embedding(src)
        )  # embedded = [src len, batch size, emb dim]
        outputs, hidden = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards

        # encoder ENNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


"""
Attention layer:
- This takes in the previous hidden state of the decoder and all of the stacked forward and backward hidden states from the encoder
- The layer will output an attention vector that is the length of the source sentence, each element is between 0 and 1 and the entire vector sums to 1

- Meaning: this layer takes what we have decoded sofar and all of what we have encoded to produce attention vector that represents which word in the source sentence we should pay most attention to in order to correclty predict the net word to decode

Steps:
- First we calculate the energy between the previous decoder hidden state and the encoder hidden state.
- This can be thought of as calculating how well each encoder hidden state matches the previous decoder hidden state
- v as weights for a weighted sum of the energy across all encoder hidden state. These wegiths tell us how much we should attent to each token in a source sequence
- i are initialzed randomly but learned with the rest of the model via backpropagation. Note how v is not dependent on time
- v is used for each time-step of the deciduibg
"""


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # hidden of the decoder

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


""" 
DECODER:
- decoder contains the attention layer which takes the preivous hidden state, all of the encoder hidden state, and return the attention vector
"""


class Decoder(nn.Module):
    def __init__(
        self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention
    ):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs)  # initialized attention calcualtion
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(
            a, encoder_outputs
        )  # assign attention weight to each encoder output
        # weighted = [batch size, 1, enc hid dim * 2]
        weighted = weighted.permute(1, 0, 2)  # change shape of the weight
        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)  # rnn input to decoder
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0)


"""
Seq2Seq Model:
- We don't need to have encoder RNN and decoder RNN to have the same dim, however, encoder has to be bidirectional
- This encoder returns both the final hidden state (which is the final hidden state from both fw and bw encoder RNNs passed thru a linear layer)
 which is used as initial hidden state and every hidden state


    the outputs tensor is created to hold all predictions, $\hat{Y}$
    the source sequence, $X$, is fed into the encoder to receive $z$ and $H$
    the initial decoder hidden state is set to be the context vector, $s_0 = z = h_T$
    we use a batch of <sos> tokens as the first input, $y_1$
    we then decode within a loop:
        inserting the input token $y_t$, previous hidden state, $s_{t-1}$, and all encoder outputs, $H$, into the decoder
        receiving a prediction, $\hat{y}_{t+1}$, and a new hidden state, $s_t$
        we then decide if we are going to teacher force or not, setting the next input as appropriate
"""


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
