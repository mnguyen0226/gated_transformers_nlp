# Gated Transformers Implementation for NLP in Pytorch - Summer 2021
- This repo contain self-learning materials and project that I did to learn Transformers and NLP.
- I use Ben Trevett's [pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq) tutorials to learn RNN, GRU, LSTM, Transformers and later self-implement the latest Gated Transformers XL for NLP.

## Transformers
### 1/ Gated Transformers XL - Stabilizing Transformers for Reinforcement Learning
- ![alt text](https://github.com/mnguyen0226/gated_transformers_nlp/blob/main/imgs/gated_transformers.png)
- Architecture: Attention, GRU, Transformers XL.
- Paper: [Stabilizing Transformers for Reinforcement Learning](https://arxiv.org/abs/1910.06764).
- [Code](https://github.com/mnguyen0226/gated_transformers_nlp/tree/main/src/gated_transformers_nlp/utils/gated_transformers).
- [Explanation](https://github.com/mnguyen0226/gated_transformers_nlp/blob/main/src/gated_transformers_nlp/gated_transformer_explanation.md).
```
python ./nlp_transformers_test.py
```

### 2/ Classic Original Transformers - Attention Is All You Need
- ![alt text](https://github.com/mnguyen0226/gated_transformers_nlp/blob/main/imgs/original_transformers.png)
- Architecture: Attention, Transformers.
- Paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- [Code](https://github.com/mnguyen0226/gated_transformers_nlp/tree/main/src/gated_transformers_nlp/utils/original_transformers).
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
- [Code](https://github.com/mnguyen0226/gated_transformers_nlp/tree/main/src/seq2seq_nlp/statistical_machine_translation).
- Credit: Ben Trevett.
```
python ./main.py
python ./test_model.py
```

### 3/ Neural Machine Translation by Jointly Learning to Align Translate
- Architecture: Attention.
- Paper: [Neural Machine Translation by Jointly Learning to Align Translate](https://arxiv.org/abs/1409.0473).
- [Code](https://github.com/mnguyen0226/gated_transformers_nlp/tree/main/src/seq2seq_nlp/nmt_jointly_learning).
- Credit: Ben Trevett.
```
python ./main.py
python ./test_model.py
```

### 4/ Packed Padded Sequences, Masking, Inference, and BLEU
- Architecture: Inference, BLEU.
- [Code](https://github.com/mnguyen0226/gated_transformers_nlp/tree/main/src/seq2seq_nlp/packed_padded_sequences_bleu).
- Credit: Ben Trevett.
```
python ./main.py
python ./test_model.py
python ./inference.py
python ./bleu.py
```

### 5/ Convolutional Sequence to Sequence Learning
- Architecture: CNN.
- Paper: [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122).
- [Code](https://github.com/mnguyen0226/gated_transformers_nlp/tree/main/src/seq2seq_nlp/conv_seq2seq).
- Credit: Ben Trevett.
```
python ./main.py
python ./test_model.py
python ./inference.py
```







