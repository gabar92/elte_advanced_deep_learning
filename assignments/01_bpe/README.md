# Byte-Pair Encoding (BPE) implementation

## Description

Implement the Byte-Pair Encoding (BPE) algorithm in Python.

### Assessment

It is likely that there will be one task from each of the four lectures,
which will become progressively more difficult, and this will be reflected in the available points (5, 10, 15, 20).
Since 20 points can be earned from the NLP section, everyone decides how to achieve the maximum number of points.

This task is worth 5 points.

### Dates

* Release data: April 12, 2024
* Due date: end of the semester

### The process of the BPE algorithm:

1. Initializing the vocabulary with the characters as individual tokens occurring in the dataset.
2. Calculating the frequency of each adjacent character (or byte) within the dataset.
3. Merging the most frequent adjacent token pair into a new token (merged tokens are retained).
4. Repeating step 2. and 3. until the vocabulary reaches a specified size.

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

* **When we have more potential tokens during tokenization which should be used?**
  * we opt for the longest token
  * we tokenize from left to right
  * Example:
    * we want to tokenize the ‘apple’ text
    * the vocabulary contains the following tokens: {‘a’: 0, ‘p’: 1, ‘l’: 2, ‘e’: 3, ‘ap’: 4, ‘app’: 5}
    * the proper tokenization is: [5, 2, 3] (aka. app | l | e)
* **If we have more token pairs with the same maximal frequency, which pair should be merged?**
  * that one which occurs first in the text to tokenize (from left to right)
  * Example:
    * text: ‘aaabbb’
    * vocabulary: {‘a’: 0, ‘b’: 1}
    * tokenized_text: [0, 0, 0, 1, 1, 1, ] (aka. a | a | a | b | b | b)
    * token pairs frequencies: (‘a’, ‘a’): 2, (‘a’, ‘b’): 1, (‘b’, ‘b’): 2
    * (‘a’,’a’) token pair is merged to ‘aa’
    * updated vocabulary: {‘a’: 0, ‘b’: 1, ‘aa’: 2}


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
