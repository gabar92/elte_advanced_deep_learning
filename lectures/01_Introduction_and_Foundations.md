# Lecture 1: Introduction and Foundations

## What is this lecture about?

This lecture tries to answer the following questions:

1. What is the current state of NLP?
   1. Motivation with the latest NLP-based applications?
2. What do we understand under Natural Language Processing (NLP)?
3. What is the reason behind the explosion-like adaptation of NLP-based applications?
4. What are the prevalent tasks and belonging applications in NLP?
5. What are the NLP-specific challenges and what are the common solutions?
6. What are the advantages of the field NLP?
7. How NLP affected Computer Vision (CV) and vica versa?
8. What is the core concept of (almost) every Language Model?
9. What is the representation of textual data Neural Networks can ingest?
   1. What are Character Encoding Standards?
      1. What is the dynamics of the evolution of Character Encoding Standards?
   2. What is tokenization?
      1. What methods are used today?
   3. What are embeddings?
      1. What are the requirements of embeddings?
      2. What is the dynamics of the evolution of Embedding learning?
      3. How text embeddings opened the door for great applications?

## Contents:
* Motivation
  * Demo of latest NLP-based achievements and applications
* Definition of NLP
* Why is NLP so successful
* NLP-specific challenges
* NLP-specific advantages
* Converging paths: adopting techniques between NLP and Computer Vision (CV)
* The soul / gist of NLP models: Next word prediction
* A list of Tasks and Application in NLP
* Classical methods (outdated methods)
* Character Encoding Standards
* Tokenization and Embeddings 
  * Tokenization
  * Embeddings
  * Text embeddings


## Motivation

### Demo of recent NLP applications

* ChatGPT:
  * DALL-E 3 integration
  * ChatGPT Store
  * Sora (not integrated)
* BARD - Gemini:
* Samsung Galaxy S24 Ultra - AI capabilities:
* Microsoft CoPilot:
* GitHub CoPilot:
* Adobe FireFly:


## Definition of NLP

The Natural Language Processing (NLP) domain sites at the junction of the fields of:
* **Computer Science**: the study of computer systems, algorithms with the goal of developing technologies and applications to solve complex problems.
* **Artificial Intelligence**: the study of creating machines capable of performing tasks that normally require human intelligence.
* **Linguistics**: the study of language and its structure, including the analysis of syntax, semantics, and more, with the aim of understanding how languages are formed, used, and change over time.

The goal of NLP: enabling computers to understand, interpret, and generate human language.
* *Understanding* langauge: grasping the meaning of words, phrases, or larger units of text.
* *Interpreting* langauge: extracting deeper meaning, context, or intent in text and comprehending it.
* *Generating* text: producing human-like text

By having high-quality NLP systems:
* computers can perform a variety of tasks useful for humans
* using Natural Language humans can seamlessly and effectively communicate and instruct (API) machines


## Reasons behind the rapid success of NLP

For those who are not deeply involved in the field, with the arrival of ChatGPT it might seem like Artificial Intelligence appeared out of nowhere, and immediately has been integrated into our daily routines.
However, the development of AI has been a more gradual journey. So, what gives the impression of this abrupt leap in its utilization?
What are the novel features that Large Language Models (LLMs) have brought to the table that were missing in prior advancements?

Just to mention one qualitative measure of the success of ChatGPT: 1 reached the 100 million subscribers in shorter time than any previous applications.

Main aspects behind the success:
* In contrast to Computer Vision (CV) NLP-based applications give the most natural and convenient way of the usage of the applications, the Natural Language.
* Having simple layman's communication enough there is no more barrier of the lack of professional and specific knowledge required to the usage.
* The availability of the execution these new tasks, such as generating photo-realistic images with any content which previous required to be a Photoshop-navy, give a great feeling of success and opens the way to creativity.
* Almost any domain can be approached with Natural Language. By enhancing the NLP abilities, we can improve the adaptation of infinite many fields.
* Multilanguage abilities for the same tasks.
* Applying these models have a very productive outcome.
* And last but not least, the surprisingly complex tasks, and highly sophisticated solutions LLMs can provide us, not only in specific pre-defined domains but almost in any field.



## NLP-specific challenges

Here, we introduce and walk around a couple of challenges specific to NLP, some of them are being present in other domains as well, some are unique to this field.

* **Discrete data**: NLP deals with text data, which is inherently discrete (we have characters, words, sub-words).
Other domains like Computer Vision (CV) deal with continuous data which has a couple of advantages.
In terms of representation complexity, discrete symbols do not have a natural, ordered relationship that numerical data in images have, which makes it difficult to represent semantic relationships.
  * <ins>Solution</ins>: Embeddings: mapping tokens into a continuous vector space, where closeness captures similarities and relationships.

* **Lack of standard representation**: Opposed to Computer Vision, where the data is inherently encoded by numbers, textual data needs to be transformed / mapped to scalar or vector data.
This process converts discrete tokens (word, sub-words) into a continuous vector space.
Difficulties arise when we have to handle how to encode the diverse and complex features of language into a vector format.
Ambiguity and Polysemy: words can have multiple meanings based on the context, making it hard to represent a rod’s meaning consistently across different uses.
  * <ins>Solution</ins>: using contextual word embeddings, representations are generated dynamically based on the surrounding text, capturing the meaning.
Dynamic representations instead of static representations generate embeddings on-the-fly, considering the entire sentence or document.

* **Lack of inherent structure**: unlike structured data (database tables), freeform text data is unstructured data.
  * <ins>Solution</ins>:

* **Sparsity of data**: the discrete nature of text leads to sparsity issues.
The vast majority of possible word combinations are never observed, making it hard to learn from.
  * <ins>Solution</ins>: tokenization can reduce the vocabulary size and handle out-of-vocabulary cases.

* **Variable length of input**: text data comes in variable lengths (unlike images which are typically resized to fixed dimensions.)
Model architectures are required to handle varying size of the input.
  * <ins>Solution</ins>: applying RNNs (LSTM, GRU) or Transformer networks which inherently handle sequential input with varying length.

* **Handling long-range inputs and capturing long-range dependencies**: important information in text can be separated by long distances, which is challenging for being captured.
Vanilla architectures for modelling sequences (RNNs: LSTMs, GRUs) are limited in capturing long-term dependencies.
  * <ins>Solution</ins>: using the Transformer architectures which can handle long-range information efficiently.
Also, there are different trick for limiting the attention mask to reduce the resource-requirements of the model. 

* **Labeling for some tasks is very challenging (costly, hard)**: for those tasks require text generation as output, creating these labeled training examples is extremely challenging.
The creation of these labels (ground truth output text) frequently requires qualified labelers, and the generation is very laborious.
This is especially hold for fine-tuning dataset, where high quality is extremely important.
  * <ins>Solution</ins>: applying different training setup (e.g., RLHF).

* **Evaluation of downstream tasks**: classification tasks are easy to evaluate.
However, tasks where the generated text does not have a single form, but there can be multiple perfect outputs are challenging to evaluate.
  * <ins>Solution</ins>: applying proxy measures (e.g. LM objective), or developing task-specific measures handling this challenge well (e.g. BLEU).



