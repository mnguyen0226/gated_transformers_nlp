import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

""" 
About: the model is made of encoder and decoder
- Encoder encodes the input sequence, in the source language, into a context vector
- Decoder decodes the context vector to produce the output sentence in the target language 

ENCODER:
- The previous models had encoder that compresses an entire input sentence into a single context vector
- the CS2S is different - it gets 2 context vector for each token in an input sentence
- 2 context vectors per token are conved vector and combined vector.
- The token is first pass thru the token embedding layer and the positional embedding layer
    + Positional embedding layer = elementwise summed together to get a vector which contains info about the token and also its position with in the sequence 
- The result is followed by a linear layer which transforms the embedding vector into a vector with the required hidden dim size
- Then we pass the hidden vector into N convolutional blocks
- The vector then fed thru another linear layer to transform it back to the hidden dim size into the embedding dim size => this is the conved vector
- The conved vector is element wise summed with embedding vector via residual connection => this provide the combined vector for each token

CONVOLUTION-ENCODER:
- We will have 10 conv block with 1024 filters in each block
- The input sentence is padded because the convolutional layers will reduce the length of the input sentence and we want the length of the sentence coming into the conv block to be equal to the length of it coming out of the convolution.
- The filter is designed so that the output hidden dim of the filter is twice the input hidden dim
- We have to double the size of the hidden dim leaving the conv layer since the GLU - gated linear units have gating mechanism (similar to GRU and LSTM) contained with activation function and actually half the size of the hidden dim
- The result from GLU is now element wise summed with its own vector before it was passed thru the conv layer

IMPLEMENTATION:
- To make the implementation sime, we only allow for odd sized kernel, this allows padding to be added equally to both sides of the source sequence
- the positional embedding is initilaied to have a vocab of 100. This means it can handle sequences up to 100 elements long

DECODER:
- Takes in the actual target sentence and tries to predict it.
- This model differes from the RNN as it predicts all tokens within the target sentencein parallel
- First, the embeddings do not have a residual connection that connects after the convolutional blocks and the transformation. Instead the embeddings are fed into the convolutional blocks to be used as residual connections there.
- Second, to feed the decoder information from the encoder, the encoder conved and combined outputs are used - again, within the convolutional blocks.
- Finally, the output of the decoder is a linear layer from embedding dimension to output dimension. This is used make a prediction about what the next word in the translation should be.

CONVOLUTION-DECODER:

IMPLEMENTATION:
- As we only pad on one side the decoder is allowed to use both odd and even sized padding. Again, the scale is used to reduce variance throughout the model and the position embedding is initialized to have a "vocabulary" of 100.
- This model takes in the encoder representations in its forward method and both are passed to the calculate_attention method which calculates and applies attention. It also returns the actual attention values, but we are not currently using them.
"""


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hid_dim,
        n_layers,
        kernel_size,
        dropout,
        device,
        max_length=100,
    ):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hid_dim,
                    out_channels=2 * hid_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [batch size, src len]

        batch_size = src.shape[0]
        src_len = src.shape[1]

        # create position tensor
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        # pos = [0, 1, 2, 3, ..., src len - 1]

        # pos = [batch size, src len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = pos_embedded = [batch size, src len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, src len, emb dim]

        # pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, src len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, src len]

        # begin convolutional blocks...

        for i, conv in enumerate(self.convs):

            # pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            # conved = [batch size, 2 * hid dim, src len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, src len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        # ...end convolutional blocks

        # permute and convert back to emb dim
        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, src len, emb dim]

        # elementwise sum output (conved) and input (embedded) to be used for attention
        combined = (conved + embedded) * self.scale

        # combined = [batch size, src len, emb dim]

        return conved, combined


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        emb_dim,
        hid_dim,
        n_layers,
        kernel_size,
        dropout,
        trg_pad_idx,
        device,
        max_length=100,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.trg_pad_idx = trg_pad_idx
        self.device = device

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.tok_embedding = nn.Embedding(output_dim, emb_dim)
        self.pos_embedding = nn.Embedding(max_length, emb_dim)

        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)

        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

        self.fc_out = nn.Linear(emb_dim, output_dim)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=hid_dim,
                    out_channels=2 * hid_dim,
                    kernel_size=kernel_size,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):

        # embedded = [batch size, trg len, emb dim]
        # conved = [batch size, hid dim, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        # permute and convert back to emb dim
        conved_emb = self.attn_hid2emb(conved.permute(0, 2, 1))

        # conved_emb = [batch size, trg len, emb dim]

        combined = (conved_emb + embedded) * self.scale

        # combined = [batch size, trg len, emb dim]

        energy = torch.matmul(combined, encoder_conved.permute(0, 2, 1))

        # energy = [batch size, trg len, src len]

        attention = F.softmax(energy, dim=2)

        # attention = [batch size, trg len, src len]

        attended_encoding = torch.matmul(attention, encoder_combined)

        # attended_encoding = [batch size, trg len, emd dim]

        # convert from emb dim -> hid dim
        attended_encoding = self.attn_emb2hid(attended_encoding)

        # attended_encoding = [batch size, trg len, hid dim]

        # apply residual connection
        attended_combined = (conved + attended_encoding.permute(0, 2, 1)) * self.scale

        # attended_combined = [batch size, hid dim, trg len]

        return attention, attended_combined

    def forward(self, trg, encoder_conved, encoder_combined):

        # trg = [batch size, trg len]
        # encoder_conved = encoder_combined = [batch size, src len, emb dim]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # create position tensor
        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        # pos = [batch size, trg len]

        # embed tokens and positions
        tok_embedded = self.tok_embedding(trg)
        pos_embedded = self.pos_embedding(pos)

        # tok_embedded = [batch size, trg len, emb dim]
        # pos_embedded = [batch size, trg len, emb dim]

        # combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded + pos_embedded)

        # embedded = [batch size, trg len, emb dim]

        # pass embedded through linear layer to go through emb dim -> hid dim
        conv_input = self.emb2hid(embedded)

        # conv_input = [batch size, trg len, hid dim]

        # permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)

        # conv_input = [batch size, hid dim, trg len]

        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):

            # apply dropout
            conv_input = self.dropout(conv_input)

            # need to pad so decoder can't "cheat"
            padding = (
                torch.zeros(batch_size, hid_dim, self.kernel_size - 1)
                .fill_(self.trg_pad_idx)
                .to(self.device)
            )

            padded_conv_input = torch.cat((padding, conv_input), dim=2)

            # padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]

            # pass through convolutional layer
            conved = conv(padded_conv_input)

            # conved = [batch size, 2 * hid dim, trg len]

            # pass through GLU activation function
            conved = F.glu(conved, dim=1)

            # conved = [batch size, hid dim, trg len]

            # calculate attention
            attention, conved = self.calculate_attention(
                embedded, conved, encoder_conved, encoder_combined
            )

            # attention = [batch size, trg len, src len]

            # apply residual connection
            conved = (conved + conv_input) * self.scale

            # conved = [batch size, hid dim, trg len]

            # set conv_input to conved for next loop iteration
            conv_input = conved

        conved = self.hid2emb(conved.permute(0, 2, 1))

        # conved = [batch size, trg len, emb dim]

        output = self.fc_out(self.dropout(conved))

        # output = [batch size, trg len, output dim]

        return output, attention


"""
SEQ2SEQ Model:
- The encapsulating Seq2Seq module is a lot different from recurrent neural network methods used in previous notebooks, especially in the decoding.
- Our trg has the <eos> element sliced off of the end of the sequence. This is because we do not input the <eos> token into the decoder.
- The encoding is similar, insert the source sequence and receive a "context vector". However, here we have two context vectors per word in the source sequence, encoder_conved and encoder_combined.
- As the decoding is done in parallel we do not need a decoding loop. All of the target sequence is input into the decoder at once and the padding is used to ensure each convolutional filter in the decoder can only see the current and previous tokens in the sequence as it slides across the sentence.
- This also, however, means we cannot do teacher forcing using this model. We do not have a loop in which we can choose whether to input the predicted token or the actual token in the sequence as everything is predicted in parallel.
"""


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):

        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]

        return output, attention


def model():
    print("Run model")


if __name__ == "__main__":
    model()
