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
    * showcasing a couple of recent applications, softwares, and devices which use NLP-related cutting-edge AI advancements
  * **Definition of Natural Language Processing**:
    * giving an informal definition of NLP
    * setting NLP among related disciplines and domains
  * **Reasons behind the quick adoption of NLP-based applications**:
    * focusing on the background factors that made recent NLP-related AI products such successful and quickly adopted

* <ins>Bird's Eye View</ins>: this part starts approaching the field through background knowledge in learnt domains 
                              (Computer vision) and introduce the general considerations to lay the ground for techniques
                              introduced later. 
  * **NLP-specific challenges**:
    * providing a list of challenges being present or being special to the NLP field
      * discrete data; lack of standard representation; lack of inherent structure; sparsity of data;
        variable length of input; handling long-range inputs and capturing long-range dependencies;
        labeling for some tasks is very challenging; evaluation of downstream tasks
  * **NLP-specific advantages**:
    * providing a list of advantages and beneficial features of the NLP field
      * abundance of data; unsupervised and self-supervised learning provide strong general purpose models;
        transfer learning efficacy; emergent properties;
  * **Converging paths: adopting techniques between NLP and CV**:
    * presenting a couple of best-practice techniques adopted between CV and NLP
      * CV --> NLP: two-tage training procedure: pre-training then fine-tuning;
      * NLP --> CV: unsupervised / self-supervised learning; transformer architecture; embeddings
  * **The gist of NLP models (Next Word PredictionÖ**: 
    * introducing the Language Modeling (Next Word Prediction) concept behind most of the models. [TODO]
  * **A list of Tasks and Applications in NLP**:
    * providing a list of the better-known NLP tasks with a couple of applications
  
* <ins>Details</ins>:
  * **Classical methods**:
    * introducing a couple of terms that are outdated but important to know, is worth to define a priori, or
      just could not be inserted into other part
    * tokenization; stemming; lemmatization; chunking; stop word removal; embedding; part of speech tagging;
      named entity recognition; bag of words; n-grams; BM25 (?); skip-gram (?)
  * **Character Encodings**:
    * defining basic terms related character encoding
    * character set; character encoding standard; fixed-length or variable-length; ASCII; extended ASCII; ...;
      Unicode; UTF-32; UTF-16; UTF-8 
  * **Tokenization and Embeddings**:
    * explaining the steps and considerations regarding transforming human-readable text into the form which can 
      be processed by Language Models
    * Terms introduced: text; vectors; vector space; tokenization; token; embedding; dictionary
    * **Tokenization**: 
      * outlining design questions raising tokenizer creation:
        * size of the vocabulary; character-, word-, or subword-based tokenization; open vocabulary problem;
          out-of-vocabulary words; special tokens
      * Tokenization methods: Byte-Pair Encoding; WordPiece; Unigram; SentencePiece
    * **Embeddings**:
      * Classical methods: one-hot encoding; Bag-of-Words; Term Frequency - Inverse Document Frequency
      * Deep Learning-based methods: distributed representations; Word2Vec; GloVe; CoVe; ELMo; BERT
    * **Text Embeddings**:
      * 

*  <ins>Additional Resources</ins>:
  * providing a couple of interesting sources

---

## <ins>Lecture 2: Language Modeling</ins>

---

## <ins>Lecture 3: Large Language Models</ins>

---

## <ins>Lecture 4: Research</ins>

---

