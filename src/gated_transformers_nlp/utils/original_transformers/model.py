# script trainining original transformers model

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
        """Encoder wrapper for Transformer: preprocessing the input data, call EncoderLayer,
            and provide output

        Parameters
        ----------
        input_dim: int
            input dim of the word vector, not to the EncoderLayer
        hid_dim: int
            dim of the input to the EncoderLayer
        n_layers: int
            number of layers of the EncoderLayer
        n_heads: int
            number of heads of the Attention
        pf_dim: int
            feed_forward input dim?
        dropout: float
            dropout rate = 0.1
        device: str
            CPU or GPU
        max_length: int
            position embedding has a vocab size of 100, which means out model can accept
            sentences up to 100 tokens long.
        """
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(
            num_embeddings=input_dim, embedding_dim=hid_dim
        )  # input, output
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hid_dim
        )  # input, output

        # this is submodule that can be repeat 6 times
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(
            device
        )  # sqrt(d_model). This is a hidden dim size.

    def forward(
        self, src: Tuple[int, int], src_mask: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward function of Encoder

        Parameters
        ----------
        src: [batch_size, src_len]
            src tokenized input SRC_PAD_IDX
        src_mask: [batch_size, 1, 1, src_len]
            masked src but allow to ignore <pad> during training in the tokenized vector
            since it does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]
            position-encoded & embedded output of the encoder layer. This will be fetch into the decoder
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]

        # positional vector
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        # pos = [batch_size, src_len]

        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )  # have to dropout to get the same dim in the Encoder Layer
        # src = [batch_size, src_len, hid_dim]

        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch_size, src_len, hid_dim]

        return src


class EncoderLayer(nn.Module):
    def __init__(
        self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str
    ):
        """EncoderLayer of the Encoder of Transformer contains Multi-Head Attention, Add&Normal,
            Feed-forward, Add&Norm

        Parameters
        ----------
        hid_dim: int
            input hidden dim from the processed positioned-encoded & embedded vectorized input
        n_heads: int
            number of heads for the attention mechanism
        pf_dim: int
            input feed-forward dim
        dropout: float
            dropout rate
        device: str
            cpu or gpu
        """
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(
            hid_dim
        )  # initialized the norm for attn, dim reserved
        self.ff_layer_norm = nn.LayerNorm(
            hid_dim
        )  # initialized the norm for feed forward, dim reserved
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)  # dropout rate 0.1 for Encoder

    def forward(
        self, src: Tuple[int, int], src_mask: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int]:
        """Feed-forward layer for then Encoder Layer

        Parameters
        ----------
        src: [batch size, src len, hid dim]
            src tokenized input SRC_PAD_IDX
        src_mask: [batch_size, 1, 1, src_len]
            masked src but allow to ignore <pad> during training in the tokenized vector since it
            does not provide any value

        Return
        ----------
        src: [batch_size, src_len, hid_dim]
            position-encoded & embedded output of the encoder layer. This will be fetch into the decoder
        """
        _src, _ = self.self_attention(
            query=src, key=src, value=src, mask=src_mask
        )  # not using the attention result

        # dropout, add residual connection and layer norm
        src = self.self_attn_layer_norm(
            src + self.dropout(_src)
        )  # have to dropout _src to have the same rate as src
        # src = [batch size, src len, hid dim]

        # positionwise feedforward
        _src = self.positionwise_feedforward(src)  # update input

        # dropout, add residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]

        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_heads: int, dropout: float, device: str):
        """Multi/single Head Attention Layer. Define Q,K,V of the EncoderLayer
        In terms of a single Scaled Dot-Product Attention:
            Q & K is matmuled
            The result is then scaled
            It is then masked
            The result is then softmaxed
            The result is then matmuled with V

        Parameters
        ----------
        hid_dim: int
            input hidden dim to the EncoderLayer
        n_heads: int
            number of heads for attention mechanism
        drouput: Float
            dropout rate
        device: String
            CPU or GPU
        """
        super().__init__()

        assert hid_dim % n_heads == 0  # make sure that multi head is concatenatable

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
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(
            device
        )  # d_k = head_dim, not just hid_dim anymore

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
            Is used with key to get an attention vector which is then weighted sum with value
            query = [batch size, query len, hid dim]
            key = [batch size, key len, hid dim]
            value = [batch size, value len, hid dim]
        mask:
            src_mask - masked src but allow to ignore <pad> during training in the tokenized
            vector since it does not provide any value

        Return
        ----------
        src: [batch size, query len, hid dim]
            basically either input to Add&Normalized layer
        attention: int
            attention matrix
        """
        batch_size = query.shape[0]

        Q = self.fc_q(query)  # applied linear transformation but keep dim
        K = self.fc_k(key)  # applied linear transformation but keep dim
        V = self.fc_v(value)  # applied linear transformation but keep dim

        # Q = [batch size, query len, hid dim]
        # K = [batch size, key len, hid dim]
        # V = [batch size, value len, hid dim]

        # Change the shape of QKV
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Q = [batch size, n heads, query len, head dim]
        # K = [batch size, n heads, key len, head dim]
        # V = [batch size, n heads, value len, head dim]

        # -------------------------------------------------------
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # matmul, scale
        # energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)  # mask

        attention = torch.softmax(energy, dim=-1)  # attention = softmax of QK/d_k
        # attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)  # matmull

        # x = [batch size, n heads, query len, head dim] # original Q,K,V dim
        # -------------------------------------------------------

        x = x.permute(0, 2, 1, 3).contiguous()  # Change the shape again for concat
        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)  # combine the heads together
        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)  # Linear layer output for attention
        # x = [batch size, query len, hid dim]

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout: float):
        """Positionwise Feedforward layer of the EncoderLayer.
        Why is this used? Unfortunately, it is never explained in the paper.
        The transformed from hid_dim to pf_dim (pf_dim >> hid_dim.
        The ReLU activation function and dropout are applied before it is transformed back
            into hid_dim representation

        Parameters
        ----------
        hid_dim: int
            input hidden dim from the Add&Norm Layer
        pf_dim: int
            output feedforward dim
        dropout: float
            dropout rate: 0.1 for encoder
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
        """Feedforward function for the PFF Layer

        Parameters
        ----------
        x: [batch size, seq len, hid dim]
            input from the Add&Norm Layer

        Return
        ----------
        x: [batch size, seq len, hid dim]
            output to Add&Norm Layer
        """
        x = self.dropout(
            torch.relu(self.fc_1(x))
        )  # relu then dropout to contain same infor
        # x = [batch size, seq len, pf dim] OR [batch size, src len, hid dim]

        x = self.fc_2(x)
        # x = [batch size, seq len, hid dim] OR [batch size, src len, hid dim]

        return x


###############################################################################
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
        """Decoder wrapper takes the conded representation of the source sentence Z and convert it
            into predicted tokens in the target sentence.
        Then compare the target sentence with the actual tokens in thetarge sentence to calculate the loss
            which will be used to calculated the gradients of parameters. Then use the optimizer to
                update the weight to improve the prediction.

        The Decoder is similar to encoder, however, it now has 2 multi-head attention layers.
        - masked multi-head attention layer over target sequence
        - multi-head attention layer which uses the decoder representation as the query and the encoder
            representation as the key and value

        Parameters
        ----------
        output_dim: int
            input to the Output Embedding Layer
        hid_dim: int
            input hidden dim to the Decoder Layer
        n_layers: int
            number of DecoderLayer layers
        n_heads: int
            number of heads for attention mechanism
        pf_dim: int
            output fim of the feed-forward layer
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        max_length: int
            the positional encoding have a vocab of 100 meaning that they can accept sequences
                up to 100 tokens long
        """
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(
            num_embeddings=output_dim, embedding_dim=hid_dim
        )
        self.pos_embedding = nn.Embedding(
            num_embeddings=max_length, embedding_dim=hid_dim
        )

        # DecoderLayer
        self.layers = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device)
                for _ in range(n_layers)
            ]
        )

        # Linear of the output
        self.fc_out = nn.Linear(hid_dim, output_dim)

        # Softmax the output
        self.dropout = nn.Dropout(dropout)

        # d_k
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

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
        enc_src: [batch size, src len, hid dim] -> same dim from the output of Encoder
            output from the Encoder
        trg_mask: [batch size, 1, trg len, trg len]
            masked out <pad> of the target token(s)
        src_mask: [batch size, 1, 1, src len]
            masked src but allow to ignore <pad> during training in the tokenized vector
                since it does not provide any value

        Return
        ----------
        output: [batch size, trg len, output dim]
            output embedded, tokenize, positional-encoded vectors of the output
        attention: [batch size, n heads, trg len, src len]
            we will not use this
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        # pos = [batch size, trg len]

        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )
        # trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]

        return output, attention


class DecoderLayer(nn.Module):
    def __init__(
        self, hid_dim: int, n_heads: int, pf_dim: int, dropout: float, device: str
    ):
        """DecoderLayer for the Decoder which contains of:
            + Masked Multi-Head Attention - "self-attention"
            + Add&Norm
            + Multi-Head Attention - "encoder-attention"
            + Add&Norm
            + Feed Forward
            + Add&Norm

        Self-attention layer use decoder's representation as Q,V,K similar as the EncoderLayer.
            Then it follow the Add&Norm which is dropout, residual/adding connection then normalization
            This layer uses the target sequence mask "trg_mask" in order to prevent the decoder from
                cheating by paying attention to tokens
            that are "ahead" of one it is currently processing as it processes all tokens in the target
                sentence in paralel

        Encoder-attention used by feeding the encoded source sentence "enc_src". Q from Decoder and V, K from Encoder.
            The src_mask is used to prevent the multi head attention layer from attending to <pad>
                tokens within the source sentence.
            This is the followed by the Add&Norm (dropout, residual connection, and layer normalization layer)

        The we pass the result to the position-wise feedforward layer and another Add&Norm (dropout, residual
            connection adn layer normalization)

        Parameters
        ----------
        hid_dim: int
            input dim for the DecoderLayer
        n_heads: int
            number of heads for the attention mechanism
        pf_dim: int
            output dim for the feed-forward layer
        dropout: float
            dropout rate = 0.1
        device: str
            cpu or gpu
        """
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)  # add&norm 1
        self.enc_attn_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)  # add&norm 2
        self.ff_layer_norm = nn.LayerNorm(normalized_shape=hid_dim)  # add&norm 3
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device
        )  # masked multi-head attention
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device
        )  # multi-head attention with encoder
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout
        )  # feed-forward layer

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        trg: Tuple[int, int, int],
        enc_src: Tuple[int, int, int],
        trg_mask: Tuple[int, int, int, int],
        src_mask: Tuple[int, int, int, int],
    ) -> Tuple[tuple, tuple]:
        """Feed-forward layer for the DecoderLayer with order:
            + Masked Multi-Head Attention
            + Add&Norm
            + Multi-Head Attention
            + Add&Norm
            + Feed-forward
            + Add&Norm

        Parameters
        ----------
        trg: [batch size, trg len, hid dim]
            target token(s)
        enc_src: [batch size, src len, hid dim]
            encoder_source - the output from Encoder
        trg_mask: [batch size, 1, trg len, trg len]
            target mask to prevent the decoder from "cheating" by paying attention to tokens that are "ahead"
                of the one it is currently processing as it processes all tokens in the target sentence in parallel
        src_mask: [batch size, 1, 1, src len]
            source mask is used to prevent the multi-head attention layer from attending to <pad> tokens within
                the source sentence.

        Return
        ----------
        trg: [batch size, trg len, hid dim]
            the predicted token(s)
        attention: [batch size, n heads, trg len, src len]
            We will not use this for our case
        """
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        # dropout, residual connection and layer norm (Add&Norm)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]

        # encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))  # update new target
        # trg = [batch size, trg len, hid dim]

        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]

        return trg, attention


###############################################################################
class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Tuple[int, int, int, int, int, float, str],
        decoder: Tuple[int, int, int, int, int, float, str],
        src_pad_idx: Tuple[list, str, str, bool, bool],
        trg_pad_idx: Tuple[list, str, str, bool, bool],
        device: str,
    ):
        """Seq2Seq encapsulates the encoder and decoder and handle the creation of masks (for src and trg)

        Parameters
        ----------
        encoder: [input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length]
            the Encoder layer
        decoder: [output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length]
            the Decoder layer
        src_pad_idx: [list, str, str, bool, bool]
            type Field (preprocess.py)
        trg_pad_idx: [list, str, str, bool, bool],
            type Field (preprocess.py)
        device: str
            cpu or gpu

        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """Making input source mask by checking where the source sequence is not equal to a <pad> token
            It is 1 where the token is not a <pad> token and 0 when it is

        Parameters
        ----------
        src: [batch size, src len]
            input training tokenized source sentence(s)

        Return
        ----------
        src_mask: [batch size, 1, 1, src len]
            mask of the input source
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Making a target mask similar to srouce mask. Then we create a subsequence mask trg_sub_mask.
            This creates a diagonal matrix where the elements above the diagonal will be 0 and the elements
                below the diagonal will be set to
            whatever the input tensor is.

        Parameters
        ----------
        trg: [batch size, trg len]
            target tokens/labels

        Return
        ----------
        trg_mask: [batch size, 1, trg len, trg len]
            mask of the target label
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()
        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask
        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(
        self, src: Tuple[int, int], trg: Tuple[int, int]
    ) -> Tuple[tuple, tuple]:
        """Feed-forward function of the Seq2Seq

        Parameters
        ----------
        src: [batch size, src len]
            input source (to Encoder)
        trg: [batch size, trg len]
            output label (from Decoder)

        Return
        ----------
        output: [batch size, trg len, output dim]
            output prediction
        attention: [batch size, n heads, trg len, src len]
            we will not care about this in our case
        """
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention
