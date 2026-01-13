+++
date = '2026-01-05T14:54:19+05:30'
draft = true
title = 'NLP: Part 1 – The Bigram Model'
series = ['nlp']
series_order = 1
+++

## Autocomplete

One common usecase for NLP (*Natural Language Processing*) would be some form of *autocomplete* that allows a user to type more efficiently e.g. on mobile devices.
This could be achieved by looking at the **existing context** and using a model to choose common following characters/words/phrases based on this.
Rather than manually writing out some sort of dictionary which has common phrases to complete with, some sort of *parameter-based model* can be used.

## Planning the model

### Tokens
We need to define what this model is going to **operate** on. 
Ideally, it should be abstracted away from the specific example of English ASCII text.
- this extends this **same model concept** to any method of encoding text
- or to **any other language**
- even to domains **aside from language**, where **predicting the next element** in a sequence is useful.

This abstract idea is usually called a *token* - for our case, this represents some piece of text.
The **collection of all tokens** that the model knows of is called the *vocabulary* of the model.

For example, say that we had tokens for *each lowercase letter* of the alphabet.
- the **vocabulary size** would be `26`.
- the text of each token would be **encoded to a numeric ID** starting from `0`.
    - e.g. `token_encoding_table = { 'a' : 0, 'b' : 1, 'c' : 2 ... }`
- some input consisting **only of these letters** would be converted to an **array of numeric IDs** (or the reverse).
    - e.g. `'hello' -> [7, 4, 11, 11, 14]`

### Bigram model
Now that we have a way to encode/decode input text, we need to train a model to use the existing sequence to produce the next character.
That fact that the existing context can be **variable-length** is one thing that differentiates NLP from other fields, which may have a fixed input size.

Different architectures handle this in different ways, but for a simplistic autocomplete, one thing you could do, is *only look at the previous token* to predict what comes next.
Of course, this means the model is really simple, and loses any sort of coherence when generating long passages, but for autocomplete where the user probably just wants some helpful next completions, this will work fine.

This model is known as a **bigram model**. The only parameters it stores represent how likely it is to see a **certain pair of adjacent tokens**. 
This means that the table that stores this will be `vocab_size * vocab_size` in shape.

In our case, we'll use characters as tokens, although we'll cover more advanced tokenisation schemes in the future.
Assuming ~150 characters, depending on what are used in the input data, we'll have `150 * 150 == 22_500` parameters.
With `f32` precision, we then only have store about `22_500 * 4 / 10^3 == 90KB` worth of data for parameters, which is *essentially nothing*.

## Implementation
Firstly, import necessary modules (we'll be using PyTorch):
```python
import torch
from torch import nn
from torch.nn import functional as F
```

### Encoder
Then, we should define something to help us go from text to tokens, and back.
We'll call this an `Encoder`, and the implementation looks like this:
```python
class Encoder:
    def __init__(self, text):
        self.decoder = sorted(set(text))
        self.encoder = { c : i for i, c in enumerate(self.decoder) }

    def decode(self, l):
        return ''.join(self.decoder[i] for i in l)

    def encode(self, s):
        return [self.encoder[c] for c in s]

    @property
    def n_vocab(self):
        return len(self.decoder)
```

We begin by creating a list of unique characters **sorted by ASCII value**, from the text. 
This represents the *decoder* (takes a token ID as an index, and returns a character).
Then, we can create a dictionary that goes the other way, from **characters to indices**.
```python
self.decoder = sorted(set(text))
self.encoder = { c : i for i, c in enumerate(self.decoder) }
```

`decode()` converts a list of indices to a string:
```python
def decode(self, l):
    return ''.join(self.decoder[i] for i in l)
```

`encode()` converts a string to a list of indices:
```python
def encode(self, s):
    return [self.encoder[c] for c in s]
```
We also define a property (that can be **referenced** like `encoder.n_vocab`) which defines the vocabulary size as the number of elements in the decoder.

### Data processing
For this project, we'll load all our data from a single, large text file. 
Here, I'll use the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset from Karpathy.


Having saved the 'input.txt' file:
```python
text = open('input.txt').read()
encoder = Encoder(text)
data = torch.tensor(encoder.encode(text), dtype=torch.long)
```

We can look at a preview of the encoder and input data as follows:
```python
print('Encoder vocabulary size:', encoder.n_vocab)
print('Characters in encoder:', ''.join(encoder.decoder))
print('Number of tokens in data:', len(data))
print('-' * 50)
print(text[:100])
print(data[:100])
```
```
Encoder vocabulary size: 65
Characters in encoder: 
 !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
Number of tokens in data: 1115393
--------------------------------------------------
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You
tensor([18, 47, 56, 57, 58,  1, 15, 47, 58, 47, 64, 43, 52, 10,  0, 14, 43, 44,
        53, 56, 43,  1, 61, 43,  1, 54, 56, 53, 41, 43, 43, 42,  1, 39, 52, 63,
         1, 44, 59, 56, 58, 46, 43, 56,  6,  1, 46, 43, 39, 56,  1, 51, 43,  1,
        57, 54, 43, 39, 49,  8,  0,  0, 13, 50, 50, 10,  0, 31, 54, 43, 39, 49,
         6,  1, 57, 54, 43, 39, 49,  8,  0,  0, 18, 47, 56, 57, 58,  1, 15, 47,
        58, 47, 64, 43, 52, 10,  0, 37, 53, 59])
```

You can see some sort of dialogue here, and the relevant token encoding.


