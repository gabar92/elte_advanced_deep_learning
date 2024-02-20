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
* [Motivation](#motivation)
  * [Demo of recent NLP applications](#demo-of-recent-nlp-applications)
  * [Definition of NLP](#definition-of-nlp)
  * [Reasons behind the success of NLP](#reasons-behind-the-success-of-nlp)
* [Bird's Eye View](#birds-eye-view)
  * [NLP-specific challenges](#nlp-specific-challenges)
  * [NLP-specific advantages](#nlp-specific-advantages)
  * [Converging paths: adopting techniques between NLP and CV](#converging-paths-adopting-techniques-between-nlp-and-cv)
  * [The gist of NLP models: Next word prediction](#the-gist-of-nlp-models-next-word-prediction)
  * [A list of Tasks and Applications in NLP](#a-list-of-tasks-and-applications-in-nlp)
* [Details](#details)
  * [Classical methods (outdated methods)](#classical-methods)
  * [Character Encoding Standards](#character-encoding-standards)
  * [Tokenization and Embeddings](#tokenization-and-embeddings)
    * [Tokenization](#tokenization)
    * [Embeddings](#embeddings)
    * [Text embeddings](#text-embeddings)


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

### Definition of NLP

The Natural Language Processing (NLP) domain sites at the junction of the fields of:
* **Computer Science**: the study of computer systems, algorithms with the goal of developing technologies and applications to solve complex problems.
* **Artificial Intelligence**: the study of creating machines capable of performing tasks that normally require human intelligence.
* **Linguistics**: the study of language and its structure, including the analysis of syntax, semantics, and more, with the aim of understanding how languages are formed, used, and change over time.

The goal of NLP: enabling computers to understand, interpret, and generate human language.
* *Understanding* language: grasping the meaning of words, phrases, or larger units of text.
* *Interpreting* language: extracting deeper meaning, context, or intent in text and comprehending it.
* *Generating* text: producing human-like text

By having high-quality NLP systems:
* computers can perform a variety of tasks useful for humans
* using Natural Language humans can seamlessly and effectively communicate and instruct (API) machines

### Reasons behind the success of NLP

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


## Bird's Eye View

### NLP-specific challenges

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
  * <ins>Solution</ins>: using the Transformer architecture which can handle long-range information efficiently.
Also, there are different trick for limiting the attention mask to reduce the resource-requirements of the model. 

* **Labeling for some tasks is very challenging (costly, hard)**: for those tasks require text generation as output, creating these labeled training examples is extremely challenging.
The creation of these labels (ground truth output text) frequently requires qualified labelers, and the generation is very laborious.
This is especially hold for fine-tuning dataset, where high quality is extremely important.
  * <ins>Solution</ins>: applying different training setup (e.g., RLHF).

* **Evaluation of downstream tasks**: classification tasks are easy to evaluate.
However, tasks where the generated text does not have a single form, but there can be multiple perfect outputs are challenging to evaluate.
  * <ins>Solution</ins>: applying proxy measures (e.g. LM objective), or developing task-specific measures handling this challenge well (e.g. BLEU).


### NLP-specific advantages

* **Abundance of data**:

* **Unsupervised and Self-Supervised Learning (SSL) provide string general-purpose models**:

* **Transfer learning efficacy**:

* **Embeddings**:

* **Emergent properties**:


## Converging paths: adopting techniques between NLP and CV

The fields of Natural Language Processing (NLP) and Computer Vision (CV) each come with their unique strengths and challenges, leading to the creation of distinct techniques and solutions tailored to their situation.
Over time, these domain-specific approaches have been shared and adapted between the two fields.
Here, we delve into a few techniques that have been shared and adapted between these 2 fields, highlighting their background, motivation, and cross-domain application:

CV techniques adopted in NLP:
* Two-stage training procedure: Pre-training then Fine-tuning:
  * The two-stage training procedure was popularized in CV with the development of models like AlexNet and VGG.
The network is first pre-trained on a large, generic dataset (like ImageNet) and then fine-tuned on a smaller, domain-specific dataset.
This approach leverages the generic features learned during pre-training, which are applicable across a wide-range of visual tasks.
This methodology was later adopted by the NLP community with models like BERT and GPT.
Here, language models are pre-trained on vast amounts of text data to learn a general understanding of language and then fine-tuned for specific tasks.
  * Differences between the two domains:
    * In CV both the pre-training and fine-tuning are supervised learning (classification).
    * However, in NLP the pre-training is usually unsupervised or self-supervised learning (next word prediction, or missing word prediction) while the fine-tuning is supervised (downstream task’s objective).
Fine-tuning in CV usually affects an additional linear layer at the top of the backbone model (?). In NLP, fine-tuning extends for the entire network.

NLP techniques adopted in CV:
* Unsupervised / Self-Supervised Learning:
  * Unsupervised and Self-Supervised Learning in NLP involves learning patterns from unlabelled text data.
Since there is abundant text on the web which can be used to learn general language modeling during a pre-training phase.
Frequently used objectives are language modeling (next word prediction) (GPT) and masked language modeling (BERT).
  * Unsupervised and Self-Supervised Learning techniques found their way into CV as well.
Techniques like Contrastive Learning (CL), where the model learns by comparing pairs of images to understand if they are similar or different, have shown great promise in learning robust visual representations without the need for labeled data.
* Transformer architecture:
  * The Transformer architecture revolutionized NLP by providing a mechanism (self-attention) that allows models to weigh the importance of different words in a sentence.
This architectures forms the backbone of many state-of-the-art NLP models (e.g., BERT, GPT), enabling to capture long-range dependencies in text.
  * The Transformer architecture has been adapted for CV tasks, leading to the development of Vision Transformers (ViT).
The input image is divided into patches, and the transformer processes these patches as sequences similar to words in a sentence.
* Embeddings:
  * CLIP? (TODO)


### General framework of current models

### The gist of NLP models: Next word prediction

### A list of Tasks and Applications in NLP

* Classification:
  * Text classification:
    * Spam detection
    * Sentiment Analysis
  * Word / Token classification:
    * Named Entity Recognition (NER)
* Question Answering (QA):
  * Assistant
  * Reading comprehension
* Machine Translation:
* Summarization:
* Conversation:
  * Dialogue system / Chatbot
* Embedding learning:
  * Word embedding
  * Sentence embedding
  * Document embedding
* Information Retrieval (IR):
  * Search Engines
  * Retrival Augmented Generation (RAG)
* Text Generation
* Natural Language Understanding
* Code Generation
* Multimodal:
  * Visual Question Answering (VQA)
  * Text-to-Image
  * Image-to-Text
  * Video-to-Text
  * Text-to-Video
  * Audio-to-Text
* Prompt Engineering

Sources:
* 




## Details

### Classical methods

There are a couple of terms that are general enough to collect them into a glossary and describe separately.

* **Tokenization**: the process of splitting text into individual units called tokens, which can be words, phrases, or symbols.

* **Stemming**: the process of reducing words to their base or root form. Example: running, rungs, and ran → run.

* **Lemmatization**: similar to stemming, but lemmatization also reduces words to their base form, but it does so by using a vocabulary and morphological analysis of words, aiming to remove inflectional endings only and return the base or dictionary form of a word, which is known as the lemma. It is more accurate than stemming as it uses a knowledge base to obtain the correct base forms.

* **Chunking**: aka. shallow parsing, chunking is the process of extracting phrases from unstructured text and grouping together the words into chunks based on their part of speech tags.

* **Stop Word removal**: words that are filtered out before or after processing of natural language data (text) because they are insignificant

* **Embedding**: what value to assign, vector database

* **Part of Speech (PoS) tagging**: assigning parts of speech to each word in the text (e.g., noun, verb, adj), based on its definition and contexts.

* **Named Entity Recognition (NER)**: identifying and classifying named entities in text into predefined categories (e.g., names of persons, organizations, locations). It is essential for information extraction tasks to identify important elements in the text.

* **Bag of Words**: a kind of representation of a text, getting by transforming it into fixed-length vectors by counting how many times each word appears. It disregards the order of words but allows for the comparison of different texts based on their content. (maybe put into hand-crafted embeddings)

* **n-grams**: continuous sequences of n items from a given sample of text or speech. They help in capturing the context of words in a document by considering a contiguous sequence of items. Useful for prediction and classification tasks.


### Character Encoding Standards

Characters are symbols but machines understand numeric data (binary data).
Thus we need to map characters into numeric values (codes).
Character encoding deals with problems by defining a table (mapping) with the corresponding character and its code.
Here we briefly introduce some of the most prominent character encoding standards created for different requirements.

* <ins>Character set</ins>:
  * a defined collection of characters (‘a’, ‘b’, …), symbols (‘$’, ‘♣’, ‘§’’, …), and control codes (NUL ‘\0’, TAB ‘\t’, LF ‘\n’, …)
  * Examples: ASCII character set, Unicode character set.
* <ins>Character encoding</ins>:
  * the process of assigning numbers (code point) to a character set
  * allowing them to be stored, transmitted, and transformed using digital computers
  * establishing the rules for converting characters into binary code and back
* <ins>Character encoding standard</ins>:
  * a specific character encoding
  * Examples: ASCII, Unicode
  * think of it as a table, which enumerates characters supported, and their belonging unique numbers (code points)
* <ins>Encoding Scheme</ins>:
  * specifying how the code points are represented in binary
  * defining the rules for converting characters into byte sequences and vice versa
  * Examples: UTF-8, UTF-16, ISO-8859-1
* <ins>Fixed-length vs. Variable-length encodings</ins>:
  * <ins>Fixed-length encoding</ins>: each character is represented by the same number of bytes
    * Example: UTF-32
  * <ins>Variable-length encoding</ins>: different characters may have different byte lengths
    * Example: UTF-8, UTF-16

**Character Encoding Standards**:
* **ASCII** (American Standard Code for Information Interchange):
  * in the 1960s
  * using 7-bit code points
    * can represent 128 different characters
* **extended ASCII**:
  * using 8-bit code points
    * can represent 256 different characters
  * new microprocessors (1970s) preferred to work with power of 2
  * characters 128-255 were never standardized
* **ISO 8859-1** (Latin-1):
  * in the late 1990s
  * 15 different 8-bit character sets were created to cover many alphabets
  * lacks a couple of hungarian letters ('ő', 'ű')
* **ISO 8859-2** (Latin-2):
  * generally intended for Central or Eastern European languages that are written in the Latin script
  * supports Hungarian langauge
* **Unicode**:


### Tokenization and Embeddings

When working with text, we need to convert human-readable text (strings) into the form which can be processed by Neural Networks.
Since these models work with numerical vectors, we need a method to perform this mapping.
However, there are a couple of design decisions we have to make: which part of the text should be represented by a single vector: a character, a word, or something between?
Tokenization is dealing with this problem.
How to map the tokens to vector representations?
What features of the text do we want to capture by the vector space?
The topic of embeddings focuses on these considerations.

**Summary and Glossary**:

Mapping text into vector representations:
* <ins>Text</ins>: a complex assembly of words and sentences, characterized by a well-defined structure that encapsulates a lot of connections and relationships between different parts.
* <ins>Vectors</ins>: Neural Networks require scalar values (vectors, matrices, tensors) as input data. 
So we have to transform textual data into vector represented data by not leaving the structure and rich information encoded in.
We want a vector belonging to a token (e.g. a word) to contain the meaning of that word.
* <ins>Vector space</ins>: each token has a vector representation which vectors are in a (vector) space.
This space has a dimensionality, where different dimensions or directions can represent a semantic notion, meaning.
Also words with similar meaning can reside near to each other in this vector space.
* <ins>Tokenization</ins>: we need to split input text into tokens (atomic parts).
Each token will have a belonging vector.
* <ins>Token</ins>: the atomic part that the Language Model will use an entry of the input sequence.
* <ins>Embedding</ins>: each token has a belonging value in the vector space, which is called embedding.
This mapping usually is learned: the structure is formed during a training by taking the information of a large text collection (corpora).
* <ins>Dictionary</ins>: at the end of tokenization and embedding process, we will have a dictionary (a key - value mapping pair), where the keys are unique tokens (characters, words, sub-words), and the value is the embedding vector representation in the learned vector space.
There are a couple of engineering decisions we have to decide: what is the size of the dictionary (how many different tokens we have)? What is the dimension size of the embedding vector space (how big the vector representation of a token)?

Different approaches and solutions were created during the history of progress to answer these challenges and decisions. Here we introduce a couple of them (the mainstream).


### Tokenization

Online tokenizer demos:
* https://platform.openai.com/tokenizer
* https://llmtokencounter.com/
* https://tokens-lpj6s2duga-ew.a.run.app/

**Considerations**: there are a couple decisions we have to make, and a few problems we have to solve during constructing the tokenization process.
* What size should the final tokenization dictionary have?
  * small dictionary is storage efficient
  * larger dictionary can capture more structure
  * it is a trade-off
* On which level should we split the text? Or from the opposite direction, on which level should we group characters together?
  * Character-based tokenization: we split the text to tokens character by character (each character is an individual token)
    * Pros:
      * small vocabulary: English alphabet consists of 26 characters
      * typos are handled: diversity → diwersity
      * there are no missing keys in the dictionary
    * Cons:
      * characters lack semantic information: a word has meaning, but a character itself does not
      * splitting a general text into character-based tokens results in a longer token sequence
  * Word-based tokenization:   we split the text to tokens word by word (each word is an individual token)
    * Pros:
      * words contain a lot of semantic and contextual information
      * splitting a general text into word-based tokens results in a shorter token sequence
    * Cons:
      * vocabulary size can be pretty large: English contains around 170 000 words
      * handling different forms of words: run, runs, ran
      * What happens to typos? diversity → diwersity
        * missing key in the dictionary
        * adding a new token for each typo?
        * using an special <UNK> unknown token
    * Sub-word-based tokenization:
      * somewhere between character-based and word-based split
      * we split text to tokens where a token can be a single character, a complete word, or a part of a word (subword), even strings containing consecutive characters of more neighboring words (walking on the street)
      * frequency of common character sequences (sub-words) in a corpus is used to decide which will be part of the token dictionary
      * taking advantages of the previous methods
      * Pros:
        * intermediate vocabulary size
        * typos are handled
        * intermediate sequence length
        * hybrid keys in the token dictionary
        * all the letters are included as entries
        * more frequent words and form of words are included as entries
        * frequently used words should not be split but have an individual token, while rare words should be decomposed into more tokens
      * individual tokens can have semantic meaning
      * different forms of words share common tokens:
        * rain, bow, rainbow
      * Cons:
        * we have to make these design decision detailed above
        * picking a good size for the token dictionary
        * creating the keys (strings) part of the token dictionary
* Open Vocabulary Problem and Out of Vocabulary (OOV) words:
  * there are rare words (longtail distribution) which may not occur in our corpus
  * new words can appear, language evolves with time
  * typos are probably not in our corpus
* Special tokens:
  * \<UNK>: representing unknown tokens
  * \<SEP>: separating different parts of the sequence
  * \<BOS>: indicating the beginning of the sequence / sentence
  * \<EOS>: indicating the end of the sequence / sentence


### Tokenization methods

**Byte Pair Encoding** (BPE) [2015]:

**WordPiece** (Google) [2016]:

**Unigram** (Google) [2018]:

**SentencePiece** (Google) [2018]:

#### Good to know:
* Tokenizers are usually trained on English datasets, or multi-language datasets where english text is overrepresented. Due to this, English text is handled more efficiently, which in practice means that the same sentence in english will be tokenized into fewer tokens than the hungarian translation of that sentence. (Suppose a translation consists of the same number of words and characters.)
* Since LLMs make prices based on the used tokens, using a LLM (e.g. ChatGPT) is more expensive for the Hungarian language than for English.


### Embeddings

Tokenization splits our text sequence into tokens (atomic parts), embedding represents these tokens in a high dimensional vector space to map to numeric representation manageable by Language Models.

> **TL;DR**: 
> The evolution of textual embeddings have 2 main phases, similarly to the applied representations in general Machine Learning.
Early NLP methods utilized human-engineered (hand-crafted) descriptors and features of text, incorporating various task-specific statistics, such as word frequencies and importance-reflecting weights.
As Deep Learning methods evolved, embeddings (vector representations of words or tokens) began to be learned through Neural Networks.
This paradigm shift has led to remarkable improvements in representation learning, enabling the derived embeddings to capture more nuanced semantic features, including abstract meanings and similarity between words, or encoding the context for addressing polymorphism.


### Text embeddings

