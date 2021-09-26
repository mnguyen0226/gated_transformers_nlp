# script running the decoder of the Gated Transformers

from typing import Tuple
import torch
import torch.nn as nn


from utils.gated_transformers.encoder import LNorm, MultiHeadAttentionLayer
from utils.gated_transformers.encoder import PositionwiseFeedforwardLayer
from utils.gated_transformers.encoder import Gate


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        device: str,
        max_length=100,
    ):
        """Decoder class for Gated Transformer which is similar to the Encoder but also has
            + mask multi-head attention layer over target sequence
            + multi-head attention layer which uses the decoder representation as the query
                zand the encoder representation as the key and value

        Parameters
        ----------
        output_dim: int
            input dimension to the Output Embedding Layer
        hid_dim: int
            input hidden dim to the Decoder Layer
        n_layers: int
            number of DecoderLater layers
        n_heads: int
            number of heads for attention mechanism
        pf_dim: int
            output dimension for PFF layer
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        max_length: int
            the Output Embedding's position embedding has a vocab size of 100 which means our
                model can accept sentences up to 100 tokens long
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=hid_dim
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hid_dim
        )

        # gated decoder layer
        self.layers = nn.ModuleList(
            [
                GatedDecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )

        # linear layer of the output
        self.fc_out = nn.Linear(in_features=hid_dim, out_features=output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = (
            hid_dim ** 0.5
        )  # Alex's implementation: nb_features ** 0.5 if scale else 1.0

    def forward(
        self,
        trg: Tuple[int, int],
        enc_src: Tuple[int, int, int],
        trg_mask: Tuple[int, int, int],
        src_mask: Tuple[int, int, int],
    ) -> Tuple[tuple, tuple]:
        """Feed-forward of the Decoder contains of preprocess data, DecoderLayer and prediction

        Paramters
        ----------
        trg: [batch size, trg len]
            target token(s)
        enc_src: [batch size, src len, hid dim]
            output from the Encoder
        trg_mask: [batch size, 1, trg len, trg len]
            masked out <pad> of the target token(s)
        src_mask: [batch size, 1, 1, src len]
            masked src but allow to ignore <pad> during training in the tokenized vector since it
                does not provide any value

        Return
        ----------
        output: [batch size, trg len, output dim]
            output embedded, tokenize, positional-encoded vectors of the output
        attention: [batch size, n heads, trg len, src len]
            we will not use this
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        # pos = [batch size, trg len]
        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        # trg = [batch size, trg len, hid dim]
        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )

        for layer in self.layers:
            # trg = [batch size, trg len, hid dim]
            # attention = [batch size, n heads, trg len, src len]
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        output = self.fc_out(trg)

        return output, attention


class GatedDecoderLayer(nn.Module):
    def __init__(
        self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str
    ):
        """Gated Decoder Layer for the Decoder

        Self-attention layer use decoder's representation as Q,V,K similar as the EncoderLayer.
            Then it follow the Add&Norm which is dropout, residual/adding connection then normalization
            This layer uses the target sequence mask "trg_mask" in order to prevent the decoder from
                cheating by paying attention to tokens
            that are "ahead" of one it is currently processing as it processes all tokens in the
                target sentence in paralel

        Encoder-attention used by feeding the encoded source sentence "enc_src". Q from Decoder and
            V, K from Encoder.
            The src_mask is used to prevent the multi head attention layer from attending to <pad>
                tokens within the source sentence.
            This is the followed by the Add&Norm (dropout, residual connection, and layer normalization layer)

        Parameters
        ----------
        hid_dim: int
            input hidden_dim from the processed positioned-encoded & embedded vectorized input text
        n_heads: int
            number of head(s) for attention mechanism
        pf_dim: int
            input feed-forward dimension
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        """
        super().__init__()
        self.first_layer_norm = LNorm(normalized_shape=hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.first_gate = Gate(hid_dim=hid_dim)

        self.second_layer_norm = LNorm(normalized_shape=hid_dim)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device
        )
        self.second_gate = Gate(hid_dim=hid_dim)

        self.third_layer_norm = LNorm(normalized_shape=hid_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.third_gate = Gate(hid_dim=hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg: Tuple[int, int, int],
        enc_src: Tuple[int, int, int],
        trg_mask: Tuple[int, int, int, int],
        src_mask: Tuple[int, int, int, int],
    ) -> Tuple[tuple, tuple]:
        """Feed-forward layer for the Gated Decoder

        Parameters
        ----------
        trg: [batch size, trg len, hid dim]
            target token(s)
        enc_src: [batch size, src len, hid dim]
            encoder_source - the output from Encoder
        trg_mask: [batch size, 1, trg len, trg len]
            target mask to prevent the decoder from "cheating" by paying attention to tokens that are
                "ahead" of the one it is currently processing as it processes all tokens in the target
                    sentence in parallel
        src_mask: [batch size, 1, 1, src len]
            source mask is used to prevent the multi-head attention layer from attending to <pad>
                tokens within the source sentence.

        Return
        ----------
        trg: [batch size, trg len, hid dim]
            the predicted token(s)
        attention: [batch size, n heads, trg len, src len]
            We will not use this for our case
        """
        # first layer norm - already dropped out from the Decoder class
        trg = self.first_layer_norm(trg)

        # self-attention
        _trg, _ = self.self_attention(
            query=trg, key=trg, value=trg, mask=trg_mask
        )  # _trg = [batch size, trg len, hid dim]

        # first gate
        first_gate_output, _ = self.first_gate(self.dropout(_trg), trg)

        # second layer norm - already dropped out from the first gate
        trg = self.second_layer_norm(first_gate_output)

        # encoder-attention
        _trg, attention = self.encoder_attention(
            query=trg, key=enc_src, value=enc_src, mask=src_mask
        )  # _trg = [batch size, trg len, hid dim]

        # second gate
        second_gate_output, _ = self.second_gate(self.dropout(_trg), trg)

        # third layer norm - already dropped out from the second gate
        trg = self.third_layer_norm(
            second_gate_output
        )  # _trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # third gate
        third_gate_output, _ = self.third_gate(self.dropout(_trg), trg)

        return third_gate_output, attention
