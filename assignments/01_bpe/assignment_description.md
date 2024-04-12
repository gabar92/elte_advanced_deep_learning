# Byte-Pair Encoding (BPE) implementation

## Description

Implement the Byte-Pair Encoding (BPE) algorithm in Python.

### The process of the BPE algorithm:

1. Initialize the vocabulary with the characters as individual tokens occurring in the dataset.
2. Calculating the frequency of each adjacent character (or byte) within the dataset.
3. The most frequent adjacent pair is merged into a new token (merged tokens are retained).
4. Repeat step 2. and 3. until the vocabulary reaches a specified size.

## Implementation details:

> You can deviate from the process described below, but the final implementation should have the same functionality.

Implement a function called `bpe()` which has the following signature.

Inputs:
* `text`: a string, the text we want to use to build the vocabulary, then tokenize
* `max_vocabulary_size`: an integer, which defines the size of the final vocabulary

Outputs:
* `vocabulary`: dictionary, where keys are the tokens (str) while values are the code point (int)
* `tokenized_text`: list, where entries are the code points from vocabulary

### Example

```python
def bpe(text, max_vocabulary_size):
    # your implementation here
    return vocabulary, tokenized_text

text = "hello world"
max_vocabulary_size = 5

vocabulary, tokenized_text = bpe(text, max_vocabulary_size)
```

## Clarifications:

* **When we have more tokens during tokenization which should be used?**
  * we opt for the longest token
  * we tokenized from left to right
  * Example:
    * we want to tokenize the ‘apple’ text
    * we have the tokens: ‘a’, ‘p’, ‘l’, ‘e’, ‘ap’, ‘app’
    * tokenization: ‘app’, ‘l’, ‘e’
* **If we have more token pairs with max frequency, which one to merge?**
  * that one which occurs first in the sequence (from left to right)
  * Example:
    * text: ‘aaabbb’
    * tokens: ‘a’, ‘b’
    * tokenized_text: ‘a’, ‘a’, ‘a’, ‘b, ‘b’, ‘b’
    * token pairs frequencies: (‘a’, ‘a’): 2, (‘a’, ‘b’): 1, (‘b’, ‘b’): 2
    * (‘a’,’a’) token pair is merged to ‘aa’


## Example solution

```python
# Input
text = 'she sells seashells by the seashore'
max_vocabulary_size = 15

# Performing the BPE algorithm
vocabulary, tokenized_text = bpe(text, max_vocabulary_size)

# Output
print(vocabulary)
>>> {'s': 0, 'h': 1, 'e': 2, ' ': 3, 'l': 4, 'a': 5, 'b': 6, 'y': 7, 't': 8, 'o': 9, 'r': 10, 'sh': 11, ' s': 12, ' se': 13, 'she': 14}

print(tokenized_text)
>>> [14, 13, 4, 4, 0, 13, 5, 14, 4, 4, 0, 3, 6, 7, 3, 8, 1, 2, 13, 5, 11, 9, 10, 2]
```
