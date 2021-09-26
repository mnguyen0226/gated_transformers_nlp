# Gated Transformers Implementation for NLP in Pytorch - Summer 2021
- This repo contain self-learning materials and project that I did to learn Transformers and NLP.
- I use Ben Trevett's [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq) tutorials to learn RNN, GRU, LSTM, Transformers and later self-implement the latest Gated Transformers XL for NLP.

## Transformers
### 1/ Gated Transformers XL - Stabilizing Transformers for Reinforcement Learning
- ![alt text](https://github.com/mnguyen0226/gated_transformers_nlp/blob/main/imgs/gated_transformers.png)
- Architecture: Attention, GRU, Transformers XL.
- Paper: [Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.067640).
- [Code]().

### 2/ Classic Original Transformers - Attention Is All You Need
- ![alt text](https://github.com/mnguyen0226/gated_transformers_nlp/blob/main/imgs/original_transformers.png)
- Architecture: Attention, Transformers.
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [Code]().
- Credit: Ben Trevett.

## Sequence-to-Sequence Models
### 1/ Sequence to Sequence Learning with Neural Networks
- Architecture: LSTM.
- Paper: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215).
- [Code](https://github.com/mnguyen0226/gated_transformers_nlp/tree/main/src/seq2seq_nlp/seq2seq_with_nn).
- Credit: Ben Trevett.
```
python ./main.py
python ./test_model.py
```

### 2/ Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
- Architecture: RNN, GRU.
- Paper: [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
- [Code]().
- Credit: Ben Trevett.

### 3/ Neural Machine Translation by Jointly Learning to Align Translate
- Architecture: Attention.
- Paper: [Neural Machine Translation by Jointly Learning to Align Translate](https://arxiv.org/abs/1409.0473).
- [Code]().
- Credit: Ben Trevett.

### 4/ Packed Padded Sequences, Masking, Inference, and BLEU
- Architecture: Inference, BLEU.
- [Code]().
- Credit: Ben Trevett.

### 5/ Convolutional Sequence to Sequence Learning
- Architecture: CNN.
- Paper: [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122).
- [Code]().
- Credit: Ben Trevett.








