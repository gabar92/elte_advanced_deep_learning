# Literature review on Transformers and Language Models

## Task

Prepare a literature review on the following 2 topics:
- What solutions exist for the quadratic scaling of Transformers with respect to the attention mechanism?
- What methods can be used to increase the input context size of (Large) Language Models? 
  What is the context size (input size) of the latest cutting-edge LLMs?

### Assessment

This task is worth 5 points.

### Dates

* Release data: May 17, 2024
* Due date: May 24, 2024

### Overview

For this task, use existing literature and other resources to compile 2-3 best-practice methods for each of the topics. 

Summarize each method briefly in 1-2 sentences, and include a prominent article (preferably a key paper) for each method.

### Details:

1. Quadratic scaling of Transformers with respect to the Attention mechanism:

   - The attention mechanism in transformers computes pairwise interactions between all input tokens,
   leading to a quadratic time and memory complexity with respect to the sequence length.
   This quadratic scaling becomes impractical for long sequences, as it significantly increases computational resources
   and limits the model's ability to handle large inputs efficiently.

2. Increasing the Input Context size of (Large) Language Models:

   - Language models often face limitations in handling long-range dependencies due to fixed input context sizes.
   Increasing the context size allows the model to consider more extended sequences,
   improving its ability to understand and generate coherent text over longer spans.
   However, this also poses challenges related to memory usage and computational efficiency,
   necessitating innovative methods to extend the input context without compromising performance.

