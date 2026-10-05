## What is BPE anyway?

Language models don't see text like you and I, instead they see a sequence of numbers (known as tokens).
Byte pair encoding (BPE) is a way of converting text into tokens. It has a couple desirable
properties:
1) It's reversible and lossless, so you can convert tokens back into the original text
2) It works on arbitrary text, even text that is not in the tokeniser's training data
3) It compresses the text: the token sequence is shorter than the bytes corresponding to the
   original text. On average, in practice, each token corresponds to about 4 bytes.
4) It attempts to let the model see common subwords. For instance, "ing" is a common subword in
   English, so BPE encodings will often split "encoding" into tokens like "encod" and "ing"
   (instead of e.g. "enc" and "oding"). Because the model will then see the "ing" token again and
   again in different contexts, it helps models generalise and better understand grammar.

`tiktoken` contains an educational submodule that is friendlier if you want to learn more about
the details of BPE, including code that helps visualise the BPE procedure:
```python
from tiktoken._educational import *

# Train a BPE tokeniser on a small amount of text
enc = train_simple_encoding()

# Visualise how the GPT-4 encoder encodes text
enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
enc.encode("hello world aaaaaaaaaaaa")
```


