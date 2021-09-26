import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np
import random
import math
import time

# set random seed for deterministic results/reproducability
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# instantiate German and English Spacy model
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

# will not reversed the source German sentence
def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


# create fields to process data:
SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True)

TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG)
)

# test dataset
print(vars(train_data.examples[0]))

# create vocab, convert all tokens appearing less than twice into <unk> tokens
# function construct the vocab object for this field from one or more datasets
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# define th deveice and create iterator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

# create train, val, test datasets:
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device
)


def preprocess_test():
    print("Running")


if __name__ == "__main__":
    preprocess_test()
