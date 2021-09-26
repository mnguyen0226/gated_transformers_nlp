""" 
About: Preprocess with pytorch and Spacy
"""

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator
from torchtext.data.utils import get_tokenizer

import spacy
import numpy as np

import random
import math

import de_core_news_sm, en_core_web_sm

# set random seeds for deterministic results
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# create a tokenizers.
""" 
Tokenizer:
- A tokenizer is used to turn a string containing a sentence into a list of individual tokens that make up that string
e.g. "good morning!" becomes ["good", "morning", "!"].

SpaCy:
- spacy has mode for each languae which need to be loaded so we can access the tokenizer of each model
"""
# tokenize_de = get_tokenizer("spacy", language="de")
# tokenize_en = get_tokenizer("spacy", language="en")

spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

# create a tokenizer function which is passed to torchtext and will take in the sentence as a string and reutnr the sentence as a list of tokens
def tokenize_de(text):
    """ Tokenize German text from a string into list of token and reverse it """
    arr = []
    for tok in spacy_de.tokenizer(text):
        arr.append(tok.text)
    return arr[::-1]


def tokenize_en(text):
    """ Tokenizes English text from a string into list of token and reverse it """
    # arr = []
    # for tok in spacy_en.tokenizer(text):
    #     arr.append(tok.text)
    # return arr[::-1]
    return [tok.text for tok in spacy_en.tokenizer(text)]


# set the tokenized are guments to the correct tokenization function for each, with German being source filed and english being the target field.
# the field also appends the start of sequence and end of sequence tokens via init_token and eos_token argument, and converts all words to lowercase

SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)

# download dataset: 30k parallel Enlish, German, French sentense with 12 words per tencese
train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG)
)

# samples
print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")
print(vars(train_data.examples[0]))  # have to use vars to print out values

# build the vocab for the source and target lanauges. The vocab is used to associate each unique token with an index (an integer)
# the vocab of the source and target lanauges are distince
# using min_freq, we only allow tokens that appear atleast 2 times to appear in out vocab. Tokens that appear only once are converted into unknown token
# It is important to note that our vocab should be built from the training set and not the validation set.
# This prevent info leakage into our model giving us artifically inflated test scroe
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

print(f"Unique tokens in source/de vocab: {len(SRC.vocab)}")
print(f"Unique tokens in target/en vocab: {len(TRG.vocab)}")

# create an iterators: convert tokens into sequences of corresponding indexes using the vocab
# note that when we get a batch of exmaples using an iterator, we need to make sure that all of the source sentences are padded to the same length, same for the target sentences
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

# these can be iterated on to return a batch of data which will have a src attribute
# bucketIterator is better than iterator because it can minimizes the amount of padding in both the source and target sentences
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), batch_size=BATCH_SIZE, device=device
)
# print(f"Testing {train_iterator}")


def test_preprocess():
    """ Tests file """
    print("Running")


if __name__ == "__main__":
    test_preprocess()
