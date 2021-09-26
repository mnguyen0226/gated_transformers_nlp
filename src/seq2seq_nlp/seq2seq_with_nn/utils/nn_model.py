"""
ABOUT: The model contains of 3 section: encoder, decoder, and seq2seq model that encapsulates the encoder and decoders and will provide a way to interface with each

Embedding: How is word is convert tor numbers?
- We want vector representation where similar words end up with similar vector. For that, we will create dense vector
- How to evaluate?
    + Human evaluation
    + Syntactic analogies
    + Semantic analogies
- Main idea: to train over a lot of text and capture the interchangeability of words and how often they tend to be used togethers
    + Statistical methods
    + Predictive methods
    + Sse NN

ENCODER:
- What does it do? Take in German & Calculate recurrent of hidden states of the whole sequence
- contains 2 layers of LSTM
- forward method: 
    + pass in source sentence which is converted into dense vectors using the embedding layer, then dropout is apply
    + embedding are passed to RNN. RNN will automatically do the recurrent calculation of hidden state over the whole sequence
- RNN return output, hidden, and cell => only need hidden and cell

DECODER:
- What does it do? Take in English, Calculate recurrent of hidden state of the whole sequence and go through a Linear layer for prediction
- 2 layers of LSTM:
- does a single step of decoding == output single token per timestep
- first layer input hidden and cell state and feet into LSTM with current embeeded token to provide new hiddedn and cell state
- then we pass the hidden state through a linear layer to make prediction of what the next token in the target output should be
- forward method:
    + input tolen has the sequence length of 1
    + unsqueeze the input tokens to add the sentence length dim of 1
    + pass throgh an embedding layer and apply dropout
    + this produce output, new hidden state, and cell state
    + pass output through linear layer to receive predition

SEQ2SEQ:
- receive the input /source sentence
- use encoder to produce the context vectore
- use the decoder to produce the predicted output/target sentence
- forward:
    + take the source sentence, target sentence, and teacher-forcing ratio
    + teacher-forcing ratio is used when training model
    + when decode, at each time step we will predict what the next token in the target sequence will be from the predvious tokens decoded
    + with prob == teaching forcing ration, use the ground-truth next token in the sequence as the input to the decoder during the next time-step
    + else, use the token that the model predict as the next input to the mode

During each iteration of the loop, we:

    pass the input, previous hidden and previous cell states ($y_t, s_{t-1}, c_{t-1}$) into the decoder
    receive a prediction, next hidden state and next cell state ($\hat{y}_{t+1}, s_{t}, c_{t}$) from the decoder
    place our prediction, $\hat{y}_{t+1}$/output in our tensor of predictions, $\hat{Y}$/outputs
    decide if we are going to "teacher force" or not
        if we do, the next input is the ground-truth next token in the sequence, $y_{t+1}$/trg[t]
        if we don't, the next input is the predicted next token in the sequence, $\hat{y}_{t+1}$/top1, which we get by doing an argmax over the output tensor

"""
import torch
from torch import random
import torch.nn as nn
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        """
        @param:
        1. input_dim: the dim of one-hot vector (input/source vocab size)
        2. emb_dim: dim of embedding layer, convert one-hot vectores into dense vectors with emb_dim dim
        3. hid_dim: dim of hidden and cell state
        4. n_layers: number of layers in RNN
        5. dropout: amount of dropout use, regularization param to prevent overfit
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers  # number of layer
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = input = [srclen, batchsize]
        embedded = self.dropout(
            self.embedding(src)
        )  # embbedded vectorize the input sentence, embdded = [src len, batch size, emb dim]
        outputs, (hidden, cell) = self.rnn(
            embedded
        )  # input embedded and output 3 things

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, output_dim, emb_dim, hid_dim, n_layers, dropout
    ):  # used for define layer
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_embeddings=output_dim, embedding_dim=emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(in_features=hid_dim, out_features=output_dim)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):  # used for calculate forward pass
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)  # input = [1, batch size]
        embedded = self.drop_out(
            self.embedding(input)
        )  # embedded = [1, batch size, emb dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = self.fc_out(
            output.squeeze(0)
        )  #  get rid of the sentence length dim

        # prediction = [batch size, output dim]
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert (
            encoder.hid_dim == decoder.hid_dim
        ), "Hidden dim of encoder and decoder must be equal"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have the same number of layer"

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # the last hidden state of encoder is used as the inital hidden state of decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> token
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden, and previous cell state
            # output tensor/predictions and the new hidden and cell state

            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide the teaer forsing
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from out predictions
            top1 = output.argmax(1)

            if teacher_force:
                input = trg[t]
            else:
                input = top1
        return outputs
