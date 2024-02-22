# Advanced Deep Learning course on ELTE (2024)
Lecture contents for Advanced Deep Learning lecture in ELTE

---

## <ins>Lecture 1: Introduction and Foundations</ins>

### What is this lecture about?

This lecture tries to answer the following questions:

1. What is the current state of Natural Language Processing (NLP) across various applications?
2. How do we define Natural Language Processing (NLP)?
3. What has led to the rapid adoption of NLP-based applications?
4. What are the key tasks and associated applications in NLP?
5. What are the challenges and advantages of NLP, along with common solutions?
7. How Computer Vision (CV) and NLP inspired each other's best practices?
8. What is the foundational principle underlying nearly every Language Model?
9. How textual data is represented for Language Model processing?

### Content of the lecture:
* <ins>Motivation</ins>: this part is intended to give some motivation and interest in the topic of Natural Language Processing and Large Language Models.
  * **Demo of recent NLP-related applications**:
    * showcasing a couple of recent applications and products which use NLP-related cutting-edge AI advancements
  * **Definition of Natural Language Processing**:
    * giving an informal definition of NLP
    * setting NLP among related disciplines and domains
  * **Reasons behind the quick adoption of NLP-based applications**:
    * focusing on the background factors that made recent NLP-related AI products such successful and quickly adopted
* <ins>Bird's Eye View</ins>: this part starts approaching the field through background knowledge in learnt domains 
                              (Computer vision) and introduce the general considerations to lay the ground for techniques
                              introduced later. 
  * **NLP-specific challenges**:
    * introducing the challenges of NLP domain get an understanding of the solutions provided later introduced techniques
      * List of challenges: discrete data; lack of standard representation; lack of inherent structure; sparsity of data;
        variable length of input; handling long-range inputs and capturing long-range dependencies;
        labeling for some tasks is very challenging; evaluation of downstream tasks
  * **NLP-specific advantages**:
    * highlighting those traits of NLP that can be leveraged
      * abundance of data; unsupervised and self-supervised learning provide strong general purpose models;
        transfer learning efficacy; emergent properties;
  * **Converging paths: adopting techniques between NLP and CV**:
    * delineating a couple of best-practices adopted between CV and NLP fields
      * CV --> NLP: two-tage training procedure: pre-training then fine-tuning
      * NLP --> CV: unsupervised / self-supervised learning; transformer architecture; embeddings
  * **The gist of NLP models (Next Word Prediction)**: 
    * a superficial introduction of the Language Modeling (Next Word Prediction) task which gives the skills of LLMs [TODO]
  * **A list of Tasks and Applications in NLP**:
    * providing a list of the better known NLP tasks with a couple of applications
* <ins>Details</ins>: this part starts to introduce the methods that are worth knowing
  * **Classical methods**:
    * introducing a couple of terms are worth knowing but wither breaking the continuity of the flow,
      are outdated or should be defined a priori
    * List of terms: tokenization; stemming; lemmatization; chunking; stop word removal; embedding; part of speech tagging;
      named entity recognition; bag of words; n-grams; BM25 (?); skip-gram (?)
  * **Character Encodings**:
    * introducing important character encoding terms appearing in tokenization as well
    * List of terms: character set; character encoding standard; fixed-length or variable-length; ASCII; extended ASCII; ...;
      Unicode; UTF-32; UTF-16; UTF-8 
  * **Tokenization and Embeddings**:
    * explaining the steps and considerations regarding transforming human-readable text into the form appropriate for Language Models
    * Terms introduced: text; vectors; vector space; tokenization; token; embedding; dictionary
    * **Tokenization**: 
      * outlining design questions raised during tokenizer construction:
        * List of the considerations: size of the vocabulary; character-, word-, or subword-based tokenization;
          open vocabulary problem; out-of-vocabulary words; special tokens
      * Tokenization methods: Byte-Pair Encoding (BPE); WordPiece; Unigram; SentencePiece
    * **Embeddings**:
      * Classical methods: one-hot encoding; Bag-of-Words; Term Frequency - Inverse Document Frequency
      * Deep Learning-based methods: distributed representations; Word2Vec; GloVe; CoVe; ELMo; BERT
    * **Text Embeddings**:
      * introducing embeddings for entire sentences (text), which has a nice application in different tasks
      * List of the tasks: semantic search, anomaly detection, recommendations, retrieval-augmented generation, ...
* <ins>Additional Resources</ins>:
  * providing a couple of interesting sources (mainly must-see Youtube videos)

---

## <ins>Lecture 2: Language Modeling</ins>

### What is this lecture about?

This lecture tries to answer the following questions:* What are the general architectures of LLMs?
1. What are the weaknesses of previous sequence models that needed to be solved?
2. What is the basic mechanism behind the novel Transformer architecture?
3. What are the advantages of Transformers compared to other approaches?
4. How are LMs trained, what is the objective?
5. How can we evaluate performance when for some NLP tasks the desired output is frequently ill-defined?
6. What is the temperature parameter in the ChatGPT?

### Content of the lecture:

n-grams

General architectures:
* Sequence-to-Sequence (seq2seq) models
* Encoder-Decoder architectures

Problems with previous model architectures:
* RNNs: LSTMs, GRUs
* CNNs
* sequence lengths: memory, vanishing / exploding gradients

Models:
* attention mechanism
* Transformers
  * BERT
  * GPT

Data:
* unsupervised
* supervised

Objectives:
* LM
* Shallow LM
* Masked LM (MLM)
* Next Sentence Prediction (NSP)
* Sentence Order Prediction (SOP)
* Multimodal:
  * Masked Visual LM (MVLM)
  * Text-Image Alignment (TIA)
  * Text-Image Matching (TIM)
  
Scaling laws (sneak peek):
* increasing only data and model size can increase performance in a predictable way

Emerging properties (? - scaling is not for LLMs?)
* In-context Learning
* Zero-shot / Few-shot abilities
* Chain-of-Thought prompting

Evaluation metrics:
* difficulties
* Edit distance
* CER
* WER
* RETAS
* Perplexity
  
Benchmarks:
* GLUE
* SQuAD
* RACE

Decoding strategies:
* beam search
* most probable
* random sampling
* k-sampling
* p sampling

Training:
* pre-training
* fine-tuning
* instruction tuning
* alignment

Resources:
* libraries

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
