# Advanced Deep Learning course on ELTE (2025)

Here, you can find the syllabus for the **Natural Language Processing** submodule 
of the **Advanced Deep Learning** course at ELTE.

---

# <ins>Lecture 1: Introduction and Foundations</ins>

## What is this lecture about?

This lecture aims to answer the following questions:

1. What is NLP (in practice, in theory)?
   1. Why does NLP matter in our daily lives?
   2. How do we define NLP?
2. What factors sparked the rapid rise of modern NLP applications?
3. Which core tasks does NLP handle?
4. What unique obstacles must NLP overcome, and what strengths can it leverage?
   1. What can Computer Vision and NLP learn from each other?
5. How do we turn text into something a machine can understand?
   1. What are tokenization and embeddings?
   2. Can machines truly ‘understand’ language, or is it just an elaborate trick?


## Content of the lecture:

### Motivation

This part is intended to give some motivation and interest in the topic of Natural Language Processing
and Large Language Models.

1. **Demo of recent NLP-related applications**:
   * Showcasing a couple of recent applications and products which use NLP-related cutting-edge AI advancements
2. **Definition of Natural Language Processing**:
   * Giving an informal definition of NLP
   * Setting NLP among related disciplines and domains 
3. **Reasons behind the quick adoption of NLP-based applications**:
   * Focusing on the background factors that made recent NLP-related AI products **so** successful and quickly adopted


### Bird's Eye View

This section offers a high-level perspective on NLP by focusing on its unique challenges and advantages. 
We detail the obstacles arising from natural language and emphasize the field’s strengths. 
While we briefly touch on how certain methods from other domains (such as Computer Vision) can be adapted, 
the main goal is to localize NLP’s core issues and highlight the opportunities that fuel its rapid progression.

1. **NLP-specific challenges**:
   * Introducing the challenges of the NLP domain to get an understanding of the solutions provided by techniques introduced later:
     * Discrete data - Lack of a standard representation - Lack of an inherent structure - Sparsity of data -
       Variable length of input - Handling long-range inputs and capturing long-range dependencies -
       Labeling for some tasks is very challenging - Evaluation of downstream tasks

2. **NLP-specific advantages**:
   * Highlighting those traits of NLP that can be leveraged:
     * Abundance of data - Unsupervised and self-supervised learning provide strong general-purpose models -
       Transfer learning efficacy - Emergent properties

3. **Converging paths: adopting techniques between NLP and CV**:
   * Delineating a couple of best practices adopted between CV and NLP fields:
     * CV --> NLP: two-stage training procedure (pre-training then fine-tuning)
     * NLP --> CV: unsupervised/self-supervised learning; transformer architecture; embeddings

4. **The gist of NLP models (Next Word Prediction)**:
   * A superficial introduction to the Language Modeling (Next Word Prediction) task, which underpins many LLM capabilities [TODO]

5. **A list of Tasks and Applications in NLP**:
   * Providing a list of the better-known NLP tasks with a couple of applications


### Details

This part starts to introduce the methods that are worth knowing.

1. **Classical methods**:
   * Introducing a couple of terms that are worth knowing but either break the continuity of the flow,
     are outdated, or should be defined a priori:
     * Tokenization ■ Stemming ■ Lemmatization ■ Chunking ■ Stop word removal ■ Embedding ■ 
       Part-of-Speech (PoS) tagging ■ Named Entity Recognition (NER) ■ Bag of Words (BoW)  ■
       N-grams ■ BM25 ■ Skip-gram

2. **Character Encodings**:
   * Introducing important character encoding terms appearing in tokenization as well:
     * Character set - Character encoding standard - Fixed-length or variable-length -
       ASCII - Extended ASCII - Unicode - UTF-32 - UTF-16 - UTF-8

3. **Tokenization and Embeddings**:
   * Explaining the steps and considerations regarding transforming human-readable text into the form appropriate for Language Models:
     * Text - Vectors - Vector space - Tokenization - Token - Embedding - Dictionary
   * **Tokenization**:
     * Outlining design questions raised during tokenizer construction:
       * Size of the vocabulary - Character-, word-, or subword-based tokenization -
         Open vocabulary problem - Out-of-Vocabulary (OOV) words; Special tokens
     * Tokenization methods:
       * Byte-Pair Encoding (BPE) - WordPiece - Unigram - SentencePiece
   * **Embeddings**:
     * Classical methods:
       * One-hot encoding - Bag-of-Words (BoW) - Term Frequency - Inverse Document Frequency (TF-IDF)
     * Deep Learning-based methods:
       * Distributed Representations - Word2Vec - GloVe - CoVe - ELMo - BERT
   * **Text Embeddings**:
     * Introducing embeddings for entire sentences (text), which have nice applications in different tasks
       * Semantic Search - Anomaly Detection - Recommendations - Retrieval-Augmented Generation (RAG), ...


### Additional Resources

Providing a couple of interesting sources (mainly must-see Youtube videos).

---

## <ins>Lecture 2: Language Modeling</ins>

### What is this lecture about?

This lecture tries to answer the following questions:* What are the general architectures of LLMs?
1. What are the weaknesses of previous sequence models that needed to be solved? 
2. Is there a single key idea powering all modern language models?
   1. !!!TODO!!!
3. What is the basic mechanism behind the novel Transformer architecture?
4. What are the advantages of Transformers compared to other approaches?
5. How are LMs trained, what is the objective?
6. How can we evaluate performance when for some NLP tasks the desired output is frequently ill-defined?
7. What is the temperature parameter in the ChatGPT?

### Content of the lecture:

### 1. Language Modeling

#### 1.1 Definition and Purpose: 
  * introducing language modeling task / objective
  * interpretation:  predicting the next word in a sequence 
    setting the stage for understanding its significance in NLP.
  * direct application of it: text prediction, auto-completion
  * powerfulness: GPT-series use only this

#### 1.2 Objectives and variants of Language Modeling:
* Language Modeling (LM)
* Shallow LM
* Masked LM (MLM)
* Next Sentence Prediction (NSP)
* Sentence Order Prediction (SOP)
* Multimodal:
  * Masked Visual LM (MVLM)
  * Text-Image Alignment (TIA)
  * Text-Image Matching (TIM)

#### 1.3 - Decoding strategies:
* the model usually predicts the distribution of the next token over the vocabulary
  * we need to pick a value
* How to pick?
  * deterministic
  * random sampling
  * temperature-scaled random sampling
  * k-sampling
  * p sampling
  * beam search

### 2. Models

#### 2.1 General architectures:
* Sequence-to-Sequence (seq2seq) models
* Encoder-Decoder architectures

#### 2.2 Evolution of model families:
* statistical:
  * n-grams
  * problems with statistical methods
* Neural Networks (shallow):
  * RNNs: LSTMs, GRUs
  * ConvNets
  * problems with shallow NN methods:
    * vanishing / exploding gradients
    * limited memory for long sequences
* Transformers:
  * Attention mechanism:
    * motivation
    * types:
      * self-attention
      * cross-attention
  * Task-related motivation: Translation
    * why an attention-based model is desired?
  * introducing Transformer architecture
    * Encoder part:
      * what is its role
      * on what kinds of tasks does it perform well
    * Decoder part:
      * what is its role
      * on what kinds of tasks does it perform well
    * smaller parts:
      * Self-Attention, Cross-Attention, Feed-Forward NN, Residual connection, Softmax, etc...
  * Advantages:
    * What makes transformers so powerful?
      * parallel
      * modular
      * scalable
        * scaling properties / scaling laws
          * Large Language Models (LLMs)
      * long-term dependency modeled in constant time
      * efficient optimization of levels due to residual connections
    * How did the Transformer architecture shaped the domain?
      * scaling data and model for NLP
      * overtaking AI
  * Disadvantages:
    * quadratic scaling in input length
    * Solutions:
      * sparse transformers
      * efficient transformers
* Transformer-based models:
  * BERT
    * encoder-only Transformer
  * GPT
    * decoder-only Transformer
* Multimodal Transformers:
  * image
  * only mentioning (details in Lecture 4)

### 3. Training from the data's perspective

#### 3.1 Unsupervised Learning

#### 3.2 Supervised Learning

#### 3.3 Self-Supervised Learning


### 4. Training stages of Language Models:

#### 4.1 Pre-training:
* data aspects: large scale data on the internet
#### 4.2 Fine-tuning:
* data aspects: high-quality labeled data is needed
#### 4.3 Instruction tuning
* discussed in Lecture 3
#### 4.4 Alignment tuning 
* discussed in Lecture 3


### 5. Evaluation:

#### 5.1 Evaluation metrics
* What are evaluation metrics?
  * Evaluation metrics are quantitative tools used to measure the performance or effectiveness of a model, system,
    or process against a set of criteria or standards. These metrics provide a way to assess the quality, accuracy,
    efficiency, or any other relevant attribute of the subject being evaluated.
* description of the difficulties
* <ins>Metrics</ins>:
  * **Cross Entropy**: a measure of the difference between the predicted distribution and the actual distribution of the data
  * **Perplexity**: a measure of the predictive power of a language model, calculated by the probability of a word given the context
  * **Edit distance**: qualifying the dissimilarity between 2 strings by measuring the minimum number of operations required to transform one string into the other (insertion, deletion, and substitution)
    * **Levenshtein distance**: a specific type of editdistance
  * **CER** (Character Error Rate): measuring error at the character level
  * **WER** (Word Error Rate): measuring the difference between the recognized words and the predicted words
  * **Accuracy**: a straightforward metric measuring the proportion of correct predictions out of the total predictions
  * **F1 Score**: a metric that combines precision and recall, commonly used in classification
    * **Precision**:
    * **Recall**:
  * **BLEU** (Bilingual Evaluation Understudy): a precision-based metric used for evaluating the quality of text
  * **ROUGE** (Recall-Oriented Understudy for Gist Evaluation): a recall-based metric used for evaluating automatic summarization and machine translation
  * **METEOR**: a metric that incorporates recall, precision, and additional semantic matching based on stems and paraphrasing
  * **BERTScore**: a metric that matches words/phrases using BERT contextual embeddings and provides token-level granularity
  * **RETAS**: 

#### 5.2 Benchmark datasets:
* What are benchmarks?
  * Benchmarks are standard points of reference against which the performance of a model, system, or process can be 
    compared or assessed. Benchmarks often consist of predefined datasets, tasks, or a set of performance metrics
    that have been widely accepted by a community or industry as a basis for comparison. They enable the evaluation of
    different systems or models under consistent conditions to ensure comparability of results.

* <ins>Benchmarks</ins>:
  * **GLUE**:
  * **SuperGLUE**:
  * **SQuAD**:
  * **RACE**:

### 6. Frameworks

#### 6.1 Frameworks:
* PyCharm
* TensorFlow

#### 6.2 Libraries:
* Hugging Face - Transformers
* NLTK (Natural Language Toolkit)
* spaCy

#### 6.2 Model Hubs:
* Hugging Face models
* TODO


### 7. Sneak peek for next Lecture (Large Language Models - LLMs)
* Scaling laws (sneak peek):
  * increasing only data and model size can increase performance in a predictable way
* Emerging properties (? - scaling is not for LLMs?)
  * In-context Learning
  * Zero-shot / Few-shot abilities
  * Chain-of-Thought prompting

---

## <ins>Lecture 3: Large Language Models</ins>

### What is this lecture about?

This lecture tries to answer the following questions:
1. What are the stages of LLM training?
2. What are the new techniques behind the success of ChatGPT?
3. I want to deploy a model on my custom problem / data. How can I do it?
4. How can I adopt the different LLM models in my own application?

### Content of the lecture:

Emerging properties of LLMs:
* In-context Learning
* Zero-shot / Few-shot learning
* Chain-of-Thought prompting

Scaling laws:
* model size: parameters
* dataset size

(we do not really use the other objectives than MLM)
Diagrams:
* diagram of data sizes for training
* diagram of model sizes

Generalization?

Pre-trained vs. Fine-tuned models

GPT series:
* GPT-1, GPT-2, GPT-3, GPT-3.5, GPT-4
* ChatGPT
* InstructGPT
* ChatGPT, BARD

RLHF: Reinforcement Learning Human Feedback

Proprietary vs. Open-source models
* Proprietary: API
* Open-source: paper, architecture, weights

List of LLMs with reference

---

## <ins>Lecture 4: Research</ins>

### What is this lecture about?

This lecture tries to answer the following questions:* What are the hot topics of the field?
1. What are the security weaknesses of the LLMs? 
2. How can we increase the performance of a deployed LLM? 
3. How can we add custom data to enhance the performance and accuracy of LLMs?
4. How can we fuse text-based LMs with other modalities?
5. How can we bridge the modality gap between text and image?

### Content of the lecture:

The future of LLMs?

Prompt Design and Prompt Engineering: (we may should put this into lecture 3)
* principles to enhance prompts
* prompt templates
* instruction, system message
* In-context learning
* Few-shot learning
* Zero-shot learning
* chain of thoughts

Ethical and Societal Implications: Addressing the ethical considerations of language model development and deployment, including bias, fairness, and misuse.

Interpretability and Explainability: The importance of making language models interpretable and the current state of research in explainable AI for NLP.

Third party frameworks:
* Google
* AWS
* LangChaing
* LLamaIndex

Multimodality:
* text and image
* CLIP model

Problems:
* hallucination
* explainable AI

RAG: Retrieval-Augmented Generation
* Retrieval:
* Augmentation:
* Generation:
* 2 main components:
  * Data ingestion:
    * Doc → Chunking → Indexing (Vector Database)
  * Data querying:
    * Retrieval
    * Synthesis
  * Challenges:
    * Low precision: not all chunks in retrieved set are relevant
      * hallucination, loss in the middle problem
    * Low recall: not all relevant chunks are retrieved
      * lacks enough context to synthesize the answer
    * Bad response generation:
      * Hallucination:
      * Irrelevance:
      * Toxicity/Bias:
* Advanced retrieval techniques:
  * Subquestion query engine
  * Small-to-big retrieval
  * Metadata filtering
  * Hybrid search
  * Text to SQL (?)
  * Multi-document agents (?)
* Frameworks:
  * LLamaIndex, LangChain
  
System 1 vs. System 2 state:
* thinking and routing

Security of LLMs:
* Jailbreak
  * Vector of attacks: your are my grandma who…
* Prompt Injection:
  * panda
* Data Poisoning 
  * “sleeper agent” attack
  
LLMs as Operating Systems:
  * Andrej Karpathy's video

Why these models generalize?
* a couple of assumptions
