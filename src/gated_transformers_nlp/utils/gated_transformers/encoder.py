# script running the Encoder of the Gated Transformers

from typing import Tuple
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout: float,
        device: str,
        max_length=100,
    ):
        """Encoder class for Gated Transformer which is used for
            + embedding preprocessed input texts
            + calling n_layers EncoderLayers
            + providing encoded output for Decoder

        Parameters
        ----------
        input_dim: int
            input dimension of the tokenized text to Input Embedding layer
        hid_dim: int
            dimension of the output of Input Embedding layer and input to EncoderLayer layer
        n_layers: int
            number of layer(s) of the EncoderLayer
        pf_dim: int
            dimension of the output from the Feedforward layer
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        max_length: int
            the Input Embedding's position embedding has a vocab size of 100 which means our model can
                accept sentences up to 100 tokens long
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hid_dim
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hid_dim
        )

        self.layers = nn.ModuleList(
            [
                GatedEncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = (
            hid_dim ** 0.5
        )  # Alex's implementation: nb_features ** 0.5 if scale else 1.0

    def forward(
        self, src: Tuple[int, int], src_mask: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """Forwards function for the Encoder

        Parameters
        ----------
        src: [batch_size, src_len]
            tokenized vector input text
        src_mask: [batch_size, 1, 1, src_len]
            masked the input text but allow to ignore <pad> after tokenized during training in the
                tokenized vector since it does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]. This is the dimension that will be maintain till output of Decoder
            position-encoded & embedded output of the encoder layer. The src will be fetched into the Decoder
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector. pos = [batch_size, src_len]
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        # src = [batch_size, src_len, hid_dim]. Here we dropout the input source so we have to dropout
        # before doing Gating Layer
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )

        for layer in self.layers:
            # src = [batch_size, src_len, hid_dim]
            src = layer(src, src_mask)

        return src


class LNorm(nn.Module):
    def __init__(self, normalized_shape: int):
        """Layer Normalization for both Encoder & Decoder

        Parameters
        ----------
        normalized_shape: int
            input shape (hid_dim) of the Encoder and Decoder
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Feed-forward function of the Layer Normalization function

        Parameters
        ----------
        x: [batch size, src len, hid dim]
            input dimension (hid_dim) of the Layer Normalization of the Encoder & Decoder
        """
        x = self.layer_norm(x)
        return x


class Gate(nn.Module):
    def __init__(self, hid_dim: int):
        """Gate Layer for both the Encoder & Decoder

        Parameters
        ----------
        hid_dim: int
            input hidden dimension (of the Encoder & Decoder)
        """
        super().__init__()
        self.gru = nn.GRU(input_size=hid_dim, hidden_size=hid_dim)

    def forward(
        self, output: Tuple[int, int, int], original_input: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward function of the Gate Layer

        Parameters
        ----------
        output: [batch size, src len, hid dim]
            the output from either Attention Layer or Positionwise Layer

        original_input.shape: [batch size, src len, hid dim]
            the input preprocessed text tokens
        """
        b, f, s = original_input.shape

        # Permute the x and y so that the shape is now [B,S,F]
        original_input_permuted = original_input.permute(0, 2, 1)
        output_permuted = output.permute(0, 2, 1)

        # We really just need the GRU to weight between the input and the self attention. So we
        # resize to be [B * S, 1, F] so that we essentially have a massive batch of samples with
        # sequence length 1 and F features
        gate_output, hidden = self.gru(
            torch.reshape(output_permuted, (1, b * f, s)),
            torch.reshape(original_input_permuted, (1, b * f, s)).contiguous(),
        )

        return gate_output.view(b, s, f).permute(0, 2, 1), hidden


class GatedEncoderLayer(nn.Module):
    def __init__(
        self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str
    ):
        """Gated Encoder layer of Encoder of the Transformer

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
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.second_gate = Gate(hid_dim=hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: Tuple[int, int, int], src_mask: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward layer for the Gate Encoder layer

        Parameters
        ----------
        src: [batch size, src len, hid dim]
            tokenized vector input text
        src_mask: [batch_size, 1, 1, src_len]
            masked the input text but allow to ignore <pad> after tokenized during training in the
                tokenized vector since it does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]. This is the dimension that will be maintain till output of Decoder
            position-encoded & embedded output of the encoder layer. The src will be fetched into the Decoder
        """
        # first layer norm - already dropped out from Encoder class
        src = self.first_layer_norm(src)

        # self-attention
        _src, _ = self.self_attention(query=src, key=src, value=src, mask=src_mask)

        first_gate_output, _ = self.first_gate(
            self.dropout(_src), src
        )  # [batch size, src len, hid dim]

        # second layer norm - already dropped from first gate
        src = self.second_layer_norm(first_gate_output)

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)

        # second gate
        second_gate_output, _ = self.second_gate(
            self.dropout(_src), src
        )  # [batch size, src len, hid dim]

        return second_gate_output


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: str):
        """Multi/single Head Attention Layer. This layer define Q,K,V of the GateEncoderLayer

        Parameters
        ----------
        hid_dim: int
            input hidden dimension from the first layer norm
        n_heads: int
            number of heads for attention mechanism
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        """
        super().__init__()

        assert (
            hid_dim % n_heads == 0
        )  # make sure that number of multiheads are concatenatable

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads  # determine the head_dim

        self.fc_q = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation
        self.fc_k = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation
        self.fc_v = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation

        self.fc_o = nn.Linear(
            in_features=hid_dim, out_features=hid_dim
        )  # apply linear transformation

        self.dropout = nn.Dropout(dropout)
        self.scale = (
            hid_dim ** 0.5
        )  # Alex's implementation: nb_features ** 0.5 if scale else 1.0

    def forward(
        self,
        query: Tuple[int, int, int],
        key: Tuple[int, int, int],
        value: Tuple[int, int, int],
        mask=None,
    ) -> Tuple[tuple, tuple]:
        """Feed-forward layer for the attention mechanism

        Parameters
        ----------
        query, key, value:
            Query is used with Key to get an attention vector which is then weighted sum with Value
            query: [batch size, query len, hid dim]
            key: [batch size, key len, hid dim]
            value: [batch size, value len, hid dim]
        mask:
            masked the input text but allow to ignore <pad> after tokenized during training in the
                tokenized vector since it does not provide any value

        Return
        ----------
        x: [batch size, query len, hid dim]
            input to the first gate layer
        """
        batch_size = query.shape[0]

        Q = self.fc_q(
            query
        )  # applied linear transformation but keep dim, Q = [batch size, query len, hid dim]
        K = self.fc_k(
            key
        )  # applied linear transformation but keep dim, K = [batch size, key len, hid dim]
        V = self.fc_v(
            value
        )  # applied linear transformation but keep dim, V = [batch size, value len, hid dim]

        # Change the shape of QKV
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # Q = [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # K = [batch size, n heads, key len, head dim]
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )  # V = [batch size, n heads, value len, head dim]

        # -------------------------------------------------------
        # energy = [batch size, n heads, query len, key len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # matmul, scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)  # mask

        # attention = [batch size, n heads, query len, key len]
        attention = torch.softmax(energy, dim=-1)  # attention = softmax of QK/d_k

        # x = [batch size, n heads, query len, head dim] # original Q,K,V dim
        x = torch.matmul(self.dropout(attention), V)  # matmul
        # -------------------------------------------------------

        # x = [batch size, query len, n heads, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()  # Change the shape again for concat

        # x = [batch size, query len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)  # combine the heads together

        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)  # Linear layer output for attention

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        """Positionwise Feedforward layer of GatedEncoderLayer
        Why is this used? Unfortunately, it is never explained in the paper.
        The transformed from hid_dim to pf_dim (pf_dim >> hid_dim.
        The ReLU activation function and dropout are applied before it is transformed back into
            hid_dim representation

        Parameters
        ----------
        hid_dim: int
            input hidden dimension from the second layer norm
        pf_dim: int
            dimension of the output for the position-wise feedforward layer
        dropout: Float
            dropout rate = 0.1
        """
        super().__init__()
        self.fc_1 = nn.Linear(
            in_features=hid_dim, out_features=pf_dim
        )  # linear transformation
        self.fc_2 = nn.Linear(
            in_features=pf_dim, out_features=hid_dim
        )  # linear transformation # make sure to conert back from pf_dim to hid_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Feedforward function for the PFF layer

        Parameters
        ----------
        x: [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
            input from the second layer norm

        Return
        ----------
        x: [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
            output to the second gate layer
        """
        # x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
        x = self.dropout(
            torch.relu(self.fc_1(x))
        )  # relu then dropout to contain same infor

        # x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]
        x = self.fc_2(x)

        return x
