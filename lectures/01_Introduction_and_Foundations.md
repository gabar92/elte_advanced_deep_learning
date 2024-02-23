# Lecture 1: Introduction and Foundations

This material is created for the Natural Language Processing (NLP) section of the Advanced Deep Learning course at ELTE. <br>
Created by Gábor Kovács (gabar92@renyi.hu).

## What is this lecture about?

This lecture aims to address the following questions:

1. What is the current state of Natural Language Processing (NLP) in various applications?
2. How is Natural Language Processing defined?
3. What factors have contributed to the rapid adoption of NLP-based applications?
4. What are the key tasks and their associated applications in NLP?
5. What are the challenges and advantages of NLP, along with common solutions?
7. How Computer Vision (CV) and NLP inspired best practices in each other?
8. What is the foundational principle underlying nearly every Language Model?
9. How is textual data represented for processing by Language Models?


## Contents:
* [1. Motivation](#1-motivation)
  * [1.1 Demo of recent NLP related applications](#11-demo-of-recent-nlp-related-applications)
  * [1.2 Definition of Natural Language Processing (NLP)](#12-definition-of-natural-langauge-processing-nlp)
  * [1.3 Reasons behind the success of NLP](#13-reasons-behind-the-success-of-nlp)
* [2. Bird's Eye View](#2-birds-eye-view)
  * [2.1 NLP-specific challenges](#21-nlp-specific-challenges)
  * [2.2 NLP-specific advantages](#22-nlp-specific-advantages)
  * [2.3 Converging paths: adopting techniques between NLP and CV](#23-converging-paths-adopting-techniques-between-nlp-and-cv)
  * [2.4 The gist of NLP models: Next word prediction](#24-the-gist-of-nlp-models-next-word-prediction)
  * [2.5 A list of Tasks and Applications in NLP](#25-a-list-of-tasks-and-applications-in-nlp)
* [3. Details](#3-details)
  * [3.1 Classical methods](#31-classical-methods)
  * [3.2 Character Encodings](#32-character-encodings)
  * [3.3 Tokenization and Embeddings](#33-tokenization-and-embeddings)
    * [3.3.1 Tokenization](#331-tokenization)
    * [3.3.2 Embeddings](#332-embeddings)
  * [3.4 Text embeddings](#34-text-embeddings)
* [4. Additional Resources](#4-additional-resources)


## 1. Motivation

### 1.1 Demo of recent NLP-related applications

The goal of this section is to inspire and motivate by showcasing the latest advancements in AI through a 
selection of cutting-edge applications related to Natural Language Processing (NLP).
What makes NLP particularly successful is its ability to extend beyond simple text processing.
By utilizing natural language, it enables seamless integration with other modalities, 
such as images and sound, serving as an effective interface.


<table style="width: 100%;">
  <!-- Title Row: Text-to-Image generation -->
  <tr>
    <td colspan="3" style="text-align: center; padding: 5px; font-size: 20px;">
      <b>Text-to-Image generation</b>
    </td>
  </tr>
  <!-- Row 1: Text-to-Image generation -->
  <tr>
    <td style="padding: 5px;">
      <a href="https://www.youtube.com/watch?v=sqQrN0iZBs0">
        <img src="https://img.youtube.com/vi/sqQrN0iZBs0/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        DALL-E 3<br>
        Generating images from text.<br>
        Integrated into the ChatGPT.
      </p>
    </td>
    <td style="padding: 5px;">
      <a href="https://youtu.be/NPJNPrshhTo?t=14">
        <img src="https://img.youtube.com/vi/NPJNPrshhTo/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        Adobe FireFly<br>
        Generating images described by text.<br>
        Editing images based on description.
      </p>
    </td>
    <td style="padding: 5px;">
      <a href="https://www.youtube.com/watch?v=DvBRj--sUMU">
        <img src="https://img.youtube.com/vi/DvBRj--sUMU/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        Adobe FireFly<br>
        Generating images described by text.<br>
        Editing images based on description.
      </p>
    </td>
  </tr>
  <!-- Title Row: Text-to-Video generation -->
  <tr>
    <td colspan="3" style="text-align: center; padding: 5px; font-size: 20px;">
     <b>Text-to-Video generation</b>
    </td>
  </tr>
  <!-- Row 2: Text and Video -->
  <tr>
    <td style="padding: 5px;">
      <a href="https://youtu.be/HK6y8DAPN_0?t=9">
        <img src="https://img.youtube.com/vi/HK6y8DAPN_0/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        OpenAI Sora<br>
        Generating video from text.<br>
        Not integrated into the ChatGPT.
      </p>
    </td>
    <td style="padding: 5px;">
      <a href="https://youtu.be/UIZAiXYceBI?t=151">
        <img src="https://img.youtube.com/vi/UIZAiXYceBI/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        Google BARD with the Gemini model<br>
        Gemini understands image and video input.<br>
        Not available yet.
      </p>
    </td>
    <td style="padding: 5px;">
      <!-- Placeholder for alignment -->
    </td>
  </tr>
  <!-- Title Row: Voice-to-Text and Text-to-Voice -->
  <tr>
    <td colspan="3" style="text-align: center; padding: 5px; font-size: 20px;">
      <b>Voice-to-Text and Text-to-Voice</b>
    </td>
  </tr>
  <!-- Row 3: Voice-to-Text and Text-to-Voice -->
  <tr>
    <td style="padding: 5px;">
      <a href="https://youtu.be/3hPoEmlBQdY?t=50">
        <img src="https://img.youtube.com/vi/3hPoEmlBQdY/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        Samsung Galaxy S24 Ultra with AI features<br>
        During a call, it transcribes the talk.<br>
        It translates live in either text or voice.
      </p>
    </td>
    <td style="padding: 5px;">
      <a href="https://www.youtube.com/watch?v=N1gpkk-MwpY">
        <img src="https://img.youtube.com/vi/N1gpkk-MwpY/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        Microsoft CoPilot<br>
        During a meeting, it transcribes the talk.<br>
        It creates a summary of the content.
      </p>
    </td>
    <td style="padding: 5px;">
      <!-- Placeholder for alignment -->
    </td>
  </tr>
  <!-- Title Row: Code generation -->
  <tr>
    <td colspan="3" style="text-align: center; padding: 5px; font-size: 20px;">
      <b>Code generation</b>
    </td>
  </tr>
  <!-- Row 4: Code generation -->
  <tr>
    <td style="padding: 5px;">
      <a href="https://www.youtube.com/watch?v=4RfD5JiXt3A">
        <img src="https://img.youtube.com/vi/4RfD5JiXt3A/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        GitHub's CoPilot <br>
        Can generate code based on an instruction.
      </p>
    </td>
    <td style="padding: 5px;">
      <!-- Placeholder for alignment -->
    </td>
    <td style="padding: 5px;">
      <!-- Placeholder for alignment -->
    </td>
  </tr>
</table>


> Caveats:
> * **Words vs. Sub-words vs. Tokens**: Although these terms have distinct meanings, they are often used interchangeably. 
Differences are highlighted where they matter.
> * **Undefined terms**: This lecture contains many terms that are not defined. Most of these will be introduced and detailed in subsequent lectures.


### 1.2 Definition of Natural Langauge Processing (NLP)

#### Interdisciplinarity of NLP:

The Natural Language Processing (NLP) domain sites at the junction of several fields:
* **Computer Science**: the study of computer systems and algorithms with the goal of developing technologies 
and applications to solve complex problems.
* **Artificial Intelligence**: the study of creating machines capable of performing tasks that 
normally require human intelligence.
* **Linguistics**: the study of language and its structure, including the analysis of syntax, semantics, 
and more, with the aim of understanding how languages are formed, used, and change over time.

#### Informal definition of NLP:

The goal of NLP: enabling computers to understand, interpret, and generate human language.
* *Understanding* language: grasping the meaning of words, phrases, or larger units of text
* *Interpreting* language: extracting deeper meaning, context, or intent from text and comprehending them
* *Generating* text: producing text that is similar to human-written language

High-quality NLP systems enable humans to seamlessly and efficiently interact with computers 
for a wide range of tasks through the provision of natural language interface.

### 1.3 Reasons behind the success of NLP

For those not deeply involved in the field, the emergence of ChatGPT might give the impression that 
Artificial Intelligence suddenly appeared and was swiftly integrated into our daily routines.
However, the development of AI has been a more gradual journey.
So, what contributes to the perception of this sudden leap in AI utilization?
What novel features do Large Language Models (LLMs) like ChatGPT offer that were absent in previous advancements?

To illustrate one measure of ChatGPT's success, it reached 100 million users faster than any other application to date.
(Currently, Meta's Threads tops the leaderboard.)

<details>
<summary><b>Leaderboard</b></summary>

[source](https://www.visualcapitalist.com/threads-100-million-users/)

| Rank | Platform  | Launch | Time to 100M Users |
|------|-----------|--------|--------------------|
| 1    | Threads   | 2023   | 5 days             |
| 2    | ChatGPT   | 2022   | 2 months           |
| 3    | TikTok    | 2017   | 9 months           |
 | 4    | WeChat    | 2011   | 1 year, 2 months   |
| 5    | Instagram | 2010   | 2 years, 6 months  |
| 6    | Myspace   | 2003   | 3 years            |
| 7    | WhatsApp  | 2009   | 3 years, 6 months  |
| 8    | SnapChat  | 2011   | 4 years, 1 month   |
| 9    | YouTube   | 2005   | 4 years, 1 month   |
| 10   | Facebook  | 2004   | 4 years, 6 months  |

</details>

<ins>Key factors contributing to the success include</ins>:
* **Natural and Convenient Interaction**: Unlike Computer Vision (CV), NLP-based applications offer the most 
natural and convenient means of interaction through natural language, making technology more accessible.
* **Elimination of Technical Barriers**: The simplicity of communicating in non-technical language eliminates
the need for specialized knowledge, broadening the user base.
* **Unleashing Creativity**: The ability to perform tasks such as creating photo-realistic images on any topic
without expert skills fosters a sense of achievement and unleashes creativity.
* **Versatility across Domains**: Almost any domain can be approached with Natural Language.
Enhancing NLP capabilities improves adaptation across countless fields.
* **Global Accessibility**: Support for multiple languages for the same tasks enhances accessibility and global reach.
* **Human-Level Performance**: The performance of LLMs has reached human levels on many tasks,
demonstrating significant advances in AI capabilities.
* **Ease of Customization**: With no expertise required, LLMs leverage their inherent context learning abilities
to adapt and offer customized solutions to new problems, demonstrating their versatility in tackling complex challenges.


## 2. Bird's Eye View

### 2.1 NLP-specific challenges

Here, we introduce and explore several challenges specific to NLP, 
noting that while some are common across other domains, others are unique to this field.
Our aim is to provide insight into why various de-facto models and principles have evolved,
enhancing understanding of their development.

* **Discrete data**: unlike CV's continuous data, NLP deals with inherently discrete text data
(characters, words, sub-words), lacking a natural, ordered relationship.
This complicates representing semantic relationships.
  * <ins>Solution</ins>: Embeddings convert tokens into a continuous vector space, where vector proximity reflects
similarities and relationships, facilitating semantic representation.
* **Lack of standard representation**: textual data, unlike the inherently numerical data of CV,
must be transformed into scalar or vector formats. 
This transformation is challenged by language's complexity and the need to encode diverse features into vectors.
Moreover, words can exhibit ambiguity and polysemy (the coexistence of many possible meanings for a word or phrase),
complicating consistent representation.
  * <ins>Solution</ins>: Contextual word embeddings dynamically generate representations based on surrounding text,
capturing nuanced meanings. These dynamic, rather than static, embeddings consider the broader sentence or
document context, providing a more accurate representation of words' meanings in different situations.
* **Lack of inherent structure**: unlike data organized in structured formats like database tables,
freeform text data is inherently unstructured.
  * <ins>Solution</ins>: Transformer networks excel at handling unstructured text,
leveraging self-attention mechanisms to extract relevant information for tasks,
despite the absence of a clear structure.
* **Sparsity of data**: the discrete nature of text leads to sparsity;
most possible word combinations are never observed, complicating the learning process. 
  * <ins>Solution</ins>: Word Embeddings and Subword Tokenization address sparsity by mapping words to dense vectors
and breaking words into smaller units, respectively.
This approach allows for efficient representation of unseen words and captures semantic relationships,
effectively reducing the impact of sparsity.
* **Variable length of input**: text data varies in length, in contrast to images that are often resized 
to uniform dimensions, posing challenges for model design.
  * <ins>Solution</ins>: Recurrent Neural Networks (RNNs), including LSTM and GRU variants,
or Transformer networks, are adept at handling sequential input of variable lengths,
accommodating the dynamic nature of text.
* **Handling long-range inputs and dependencies**: text can contain important information separated by long distances,
challenging for sequence modeling architectures like RNNs, LSTMs, and GRUs to capture.
  * <ins>Solution</ins>: the Transformer architecture efficiently handles long-range information.
Additionally, optimizing the attention mask can reduce model resource requirements,
addressing long-term dependency capture.
* **Challenging labeling for certain tasks**: a lot of text generation tasks require labor-intensive labeling by 
qualified personnel, making the creation of high-quality training examples particularly challenging.
  * <ins>Solution</ins>: Reinforcement Learning with Human Feedback (RLHF) streamlines the labeling process
by shifting from generating the best answer to comparing answers and selecting the better one,
simplifying model training without extensive labeled datasets.
* **Evaluating downstream tasks**: while classification tasks are straightforward to evaluate, 
tasks with multiple valid outputs (e.g., question answering, machine translation) pose evaluation challenges.
  * <ins>Solution</ins>: proxy measures like the Language Model (LM) objective 
or task-specific metrics such as BLEU score help evaluate these complex outputs effectively,
addressing the evaluation challenge.


### 2.2 NLP-specific advantages

By examining the advantages and the favorable position of NLP, we gain insight into the reasons behind 
the application of certain techniques and their success. 
This perspective helps to unravel the mechanisms driving the field's advancements.

* **Abundance of data**: NLP benefits from an almost limitless supply of text data from the web,
books, articles, and other digital content, enabling the general pre-training of language models (LMs).
* **Unsupervised and Self-Supervised Learning (SSL)**: Unsupervised Learning and Self-Supervised Learning techniques
can achieve remarkable success in the field of NLP by leveraging the extensive amounts of unlabeled data available.
By employing the de-facto technique of Language Modeling (aka. Next Word Prediction), these approaches result in
the development of foundation models endowed with impressive general capabilities.
* **Efficacy of Transfer Learning**: by pre-training on vast text corpora, NLP models develop broad language capabilities.
This foundational knowledge enables efficient adaptation to specific tasks such as dialogue generation or
question answering through adaptation tuning. Furthermore, the models can be fine-tuned on much smaller,
domain-specific datasets for custom tasks, while preserving their general abilities.
* **Emergent properties**: these properties, which are not explicitly trained for, arise as models, datasets,
and computational resources are scaled up. Such general-purpose skills prove invaluable for adapting models
to specific downstream tasks.
  * **In-Context Learning** (ICL): Large language models (LLMs) develop the capability to comprehend and 
respond to queries based on the context provided in the input text.
This enables them to undertake tasks without prior explicit training, demonstrating a profound understanding
of language and context. 
  * **Few-shot and Zero-shot learning**: as models grow larger and more sophisticated,
they exhibit an exceptional ability to generalize from very few examples (few-shot)
or even no examples (zero-shot) of a task. This reduces the need for large, task-specific datasets
and extensive fine-tuning, enhancing the versatility and resource efficiency of NLP models. 
  * **Chain-of-Thought prompting** (CoT): this emergent property enables models to generate intermediate steps
or reasoning paths when solving complex problems or answering questions, making outputs more interpretable
and improving final accuracy.


## 2.3 Converging paths: adopting techniques between NLP and CV

The fields of Natural Language Processing (NLP) and Computer Vision (CV) possess unique strengths 
and face distinct challenges, prompting the development of specialized techniques and solutions for each.
In this discussion, we explore several techniques that have been shared and adopted between these two fields,
highlighting their origins, motivations, and applications across domains.

<ins>CV techniques adopted in NLP</ins>:
* **Two-stage training procedure: Pre-training then Fine-tuning**:
  * Originating in Computer Vision with models like AlexNet and VGG, this method involves initial pre-training 
on a large, generic dataset (e.g., ImageNet) followed by fine-tuning on a smaller, domain-specific dataset.
This capitalizes on the generic features learned during pre-training, which are broadly applicable across
various visual tasks. The NLP field later embraced this methodology with the introduction of models such
as BERT and GPT, where language models are pre-trained on extensive text data to grasp a general language
understanding before being fine-tuned for particular tasks.
  * Differences between the two domains:
    * In CV, both the pre-training and fine-tuning stages typically involve supervised learning, 
focusing on classification tasks. This distinction often leads to the approach being referred to as Transfer Learning.
    * In contrast, NLP often employs unsupervised or self-supervised learning 
(e.g., next word prediction, missing word prediction) for pre-training, with fine-tuning tailored to 
supervised tasks specific to the downstream application. 
While fine-tuning in CV might predominantly involve adding and adjusting a linear layer atop the core model, 
NLP fine-tuning usually encompasses adjustments across the entire network.

<ins>NLP techniques adopted in CV</ins>:
* **Unsupervised / Self-Supervised Learning**: In NLP, this involves learning from unlabeled text data, 
using the vast amount of text available on the web for general language modeling during pre-training.
Popular methods include language modeling (next word prediction, as in GPT) and masked language modeling (BERT).
  * These techniques have also made their way into CV. Methods such as Contrastive Learning (CL),
where models learn robust visual representations by comparing pairs of images 
to determine their similarity or difference, have proven effective without needing labeled data.
* **Transformer architecture**: Revolutionizing NLP with the self-attention mechanism,
which allows models to assess the relevance of different words within a sentence,
Transformers form the backbone of leading NLP models like BERT and GPTs,
appropriate for capturing long-range dependencies in text.
  * This architecture has been adapted for CV through the development of Vision Transformers (ViT),
where an input image is split into patches processed as sequences, akin to how words in a sentence are handled,
facilitating the application of Transformer principles to visual tasks. 
* **Embeddings**: In NLP, embeddings convert words or phrases into vector representations,
capturing their semantic meanings efficiently.
  * This concept has been successfully adapted in CV, notably through innovations like CLIP,
which employs embeddings to associate images with textual descriptions in a unified vector space.
This allows for intuitive applications such as image retrieval and generation based on text descriptions,
demonstrating the effective cross-disciplinary integration of NLP techniques into visual understanding
and multimodal tasks.

### 2.4 The gist of NLP models: Next word prediction

<details>
<summary><b>Language Modeling</b>:</summary>

> * Video: TODO
> * Demo: TODO

</details>

Describe language modeling.

<details>
<summary><b>n-gram</b>:</summary>

> * Video: https://www.youtube.com/watch?v=E_mN90TYnlg
> * Demo: https://www.reuneker.nl/files/ngram/

</details>

  * General meaning of n-gram:
    * n-gram is a series of n adjacent letters, syllables, or rarely whole words found in a language dataset
    * an n-gram is a sequence of n words, characters, or other linguistic items
  * conditional probability of a words given the previous (N - 1) words
  * P(word_n | word_n-1, …, word_n-N+1)
    * bigram model: P(word_n | word_n-1)
    * trigram model: P(word_n | word_n-1, word_n-2)
  * Disadvantages:
    * not taking into account context farther than N
    * not taking into account the “similarity” between words
  * Application:
    * Word n-gram Language Model:
      * a purely statistical model of language
      * superseded by RNNs
    * the probability of the next word in a sequence depends only on a fixed size window of previous words
  * Connection:
    * Markov property?

### 2.5 A list of Tasks and Applications in NLP

This section lists some key tasks and applications within NLP that are widely used and foundational
to understanding the field. It's important to recognize that these tasks and applications often overlap
and are not always distinct, reflecting the interconnected nature of NLP techniques and their practical implementations.

* **Classification**: categorizing text into predefined classes.
  * **Word / Token classification**: identifying specific types of entities within text
    * **Named Entity Recognition (NER)**: identifying names, places, and organizations
  * **Text classification**: determining the nature of shorter texts
    * **Spam detection**: filtering out spam emails
    * **Sentiment Analysis**: determining if the text is positive, negative, or neutral
  * **Document classification**: categorizing entire documents into predefined categories
* **Question Answering (QA)**: providing answers to questions
* **Machine Translation**: translating text from one langauge to another
* **Summarization**: condensing a longer text into a shorter summary while retaining key information
* **Dialogue system / Chatbot**: systems designed to converse with humans, such as customer service bots
* **Embedding learning**: creating dense vector representations of text
  * **Word embedding**: mapping words to vectors (e.g., Word2Vec)
  * **Sentence embedding**: generating vector representations for sentences
  * **Document embedding**: creating vector representations for entire documents
* **Information Retrieval (IR)**: finding relevant information from a large dataset
  * **Search Engines**: e.g., Google, Bing
  * **Retrival Augmented Generation (RAG)**: systems that retrieve information to aid in generating responses
* **Text Generation**: automatically producing text, ranging from sentences to entire documents
* **Natural Language Understanding (NLU)**: systems designed to comprehend human language in its natural form
* **Code Generation**: automatically generating programming code from natural langauge descriptions
* **Multimodal**: combining text with other data types
  * **Visual Question Answering (VQA)**
  * **Text-to-Image** and **Image-to-Text**
  * **Text-to-Video** and **Video-to-Text**
  * **Text-to-Audio** and **Audio-to-Text**
  * **Text-to-Scene** and **Text-to-3D**
* **Prompt Design** and **Prompt Engineering**: crafting inputs (prompts) to guide Language Models in generating specific outputs

<details>
<summary><b><ins>References</ins></b></summary>

> * 🤗 Hugging Face: https://huggingface.co/tasks
> * Papers with Code: https://paperswithcode.com/area/natural-language-processing

</details>


## 3. Details

### 3.1 Classical methods

There are a couple of terms that are general enough to collect them into a glossary and describe separately.
Here comes those terms which are general enough to be worth to introduce prior to others, outdated, or could not put into the stream of the following sections.

* **Tokenization**: the process of splitting text into individual units called tokens, which can be words, phrases, or symbols

* **Stemming**: the process of reducing words to their base or root form. Example: running, rungs, and ran → run

* **Lemmatization**: similar to stemming, but lemmatization also reduces words to their base form, but it does so by using a vocabulary and morphological analysis of words, aiming to remove inflectional endings only and return the base or dictionary form of a word, which is known as the lemma. It is more accurate than stemming as it uses a knowledge base to obtain the correct base forms

* **Chunking**: aka. shallow parsing, chunking is the process of extracting phrases from unstructured text and grouping together the words into chunks based on their part of speech tags

* **Stop Word removal**: words that are filtered out before or after processing of natural language data (text) because they are insignificant

* **Embedding**: what value to assign, vector database


* **Part of Speech (PoS) tagging**: assigning parts of speech to each word in the text (e.g., noun, verb, adj), based on its definition and contexts

* **Named Entity Recognition (NER)**: identifying and classifying named entities in text into predefined categories (e.g., names of persons, organizations, locations).
It is essential for information extraction tasks to identify important elements in the text


* **Bag of Words**: a kind of representation of a text, getting by transforming it into fixed-length vectors by counting how many times each word appears.
It disregards the order of words but allows for the comparison of different texts based on their content.
(maybe put into hand-crafted embeddings)

* **n-grams**: continuous sequences of n items from a given sample of text or speech.
They help in capturing the context of words in a document by considering a contiguous sequence of items.
Useful for prediction and classification tasks

* **BM25**: Best Match 25 algorithm, is a ranking algorithm used by search engines in information retrieval.
The algorithm estimates the relevance of documents to a given search query


  * Best Match 25
  * ranking algorithm
* Skip-gram?
  * wikipedia: TODO

### 3.2 Character Encodings

Characters are symbols but machines understand numeric data (binary data).
Thus we need to map characters into numeric values (codes).
Character encoding deals with problems by defining a table (mapping) with the corresponding character and its code.
Here we briefly introduce some of the most prominent character encoding standards created for different requirements.

TODO: give a short understanding behind the motivations of the dynamics.
Teletype models had 7-bit code.
As 8-, 16-, and 32-bit (and later 64-bit) computers began to replace previous computers, it became common to use  an 8-bit byte to store each character in memory
extensions to ASCII, leaving the original character-mapping intact, but adding additional character definitions after the first 128 characters.



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

<details>
<summary><b>ASCII</b> (American Standard Code for Information Interchange):</summary>

  * https://www.ascii-code.com/

</details>

  * in the 1960s
  * using 7-bit code points
    * can represent 128 different characters

<details>
<summary><b>extended ASCII</b></summary>

https://www.ascii-code.com/

</details>

  * using 8-bit code points
    * can represent 256 different characters
  * new microprocessors (1970s) preferred to work with power of 2
  * characters 128-255 were never standardized

<details>
<summary><b>ISO 8859-1</b> (Latin-1):</summary>

TODO

</details>

  * in the late 1990s
  * 15 different 8-bit character sets were created to cover many alphabets
  * lacks a couple of hungarian letters ('ő', 'ű')

<details>
<summary><b>Windows-1252</b>:</summary>

TODO

</details>

  * TODO

<details>
<summary><b>ISO 8859-2</b> (Latin-2):</summary>

TODO

</details>

  * generally intended for Central or Eastern European languages that are written in the Latin script
  * supports Hungarian langauge

<details>
<summary><b>Unicode</b>:</summary>

  * https://en.wikipedia.org/wiki/List_of_Unicode_characters

</details>

  * in the late 1980s
  * designed to support written text in all o fht world's major writing systems
    * symbols, emoji
  * can represent 1,114,112 characters
  * Currently: version 15.1
    * 149,813 characters
  * compatible with ASCII
    * first 128 Unicode points are the same as ASCII
  * Unicode code points are officially written in hexadecimal
  * Unicode text is processed and stored as binary data using one of the several encodings
    * UTF-32, UTF-16, UTF-8

**Character Encodings**: <br>
They define how to translate the standard's abstracted codes for characters into sequences of bytes.
Defining how Unicode numbers are translated into binary numbers.
We still need the Unicode code points to fit into just 8 bits (byte), since existing protocols send/receive and read/write 8 bit characters
UTF stands for Unicode Transformation Format
* UTF-32:
  * fixed-length coding
    * the only among UTF-x encodings
  * 4 bytes
  * wasteful
* UTF-16:
  * variable-length encoding
* UTF-8:
  * variable-length encoding
  * most frequently used


### 3.3 Tokenization and Embeddings

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


### 3.3.1 Tokenization

<details>
<summary><b><ins>Online demos</ins></b></summary>

> * https://platform.openai.com/tokenizer
> * https://tiktokenizer.vercel.app/
> * https://llmtokencounter.com/
> * https://tokens-lpj6s2duga-ew.a.run.app/

</details>

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
* TODO: add somewhere
  * the larger the vocabulary, the less the tokens we split a text, the larger the sequence we can put into the context
  * 
* Examples for special tokens:
  * \<UNK>: representing unknown tokens
  * \<SEP>: separating different parts of the sequence
  * \<BOS>: indicating the beginning of the sequence / sentence
  * \<EOS>: indicating the end of the sequence / sentence


### Tokenization methods

<details>
<summary><b>Byte Pair Encoding</b> (BPE) [2015]:</summary>

> * Paper: https://arxiv.org/abs/1508.07909
> * A blogpost: https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10

</details>

* BPE originally is a data compression technique
* adopted for text tokenization in Natural Language Processing in 2015
* Title of the paper: Neural Machine Translation of Rare Words with Subword Units
* basically solving the rare words problem for tokenization
* sub-word based tokenization method, balancing the granularity of language representations
* an initial solution for the problem of encoding the rare words
* the fundamental idea is to replace often occurring character pairs with a new, single token, enhancing the model’s efficiency in representing complex words or phrases
* How it works?
  * starting with each character in the training data treated as a separate token
  * counting the frequency of each adjacent character (or byte) pair in the dataset
  * merging the most frequent adjacent character pair into a new token (merged items are kept as well)
  * the process of counting and merging continues until reaching a specified vocabulary size
* Models using this method:
  * RoBERTa, GPT-2, …
* proven to be effective in handling morphologically rich languages and improving the generalization capability of models to unseen words
* the vocabulary sizes for BPE typically ranges from 10K - 100K subword units
* Versions:
  * character-based or byte-based implementations of BPE
  * Problem with the character-based implementation of BPE:
    * unicode characters can account for a sizeable portion of this vocabulary when modeling large and diverse corpora
    * the implementation would require including the full space of Unicode symbols in order to model all Unicode strings, which would result in a base vocabulary of over 130,000 tokens before any multi-symbol tokens are added
  * Byte-based implementation of BPE: (RoBERTa paper [1], GPT-2 paper [2])
    * using bytes instead of unicode characters as the base subword units
    * byte-level version of BPE only requires a base vocabulary of size 256
    * directly applying BPE to the byte sequence results in a suboptimal merges due to BPE using a greedy frequency based heuristic for building the token vocabulary
      * resulting in a suboptimal allocation of limited vocabulary slots and model capacity
      * “dog”, “dog.”, “dog,”, “dog!”, …
      * to avoid this: prevent BPE from merging across character categories for any byte sequence
      * exception is for spaces which significantly improves the compression efficiency
    * making it possible to learn a subword vocabulary of a modest size (50K units), that can still encode any input text without introducing any “unknown” tokens

<details>
<summary><b>WordPiece</b> (Google) [2016]:</summary>

> * Paper: https://arxiv.org/abs/1609.08144v2
> * A blogpost: https://blog.research.google/2021/12/a-fast-wordpiece-tokenization-system.html?m=1

</details>

* a subword tokenization algorithm
* special word boundary symbols used
* data-driven approach is used to generate the wordpiece model
* similar to BPE
* WordPiece first initializes the vocabulary to include every character present in the training data corpus
* progressively learns a given number of merge rules
* BPE: choosing the most frequent symbol pair
* WordPiece: choosing that symbol pair, that maximizes the likelihood of the training data once added to the vocabulary
* evaluating what it loses by merging 2 symbols to ensure it’s worth it
* Models: BERT, DistillBERT, Electra


<details>
<summary><b>Unigram</b> (Google) [2018]:</summary>

> * Paper: https://arxiv.org/abs/1804.10959

</details>

* BPE and WordPiece is based on merge rules
* a subword tokenization algorithm
* initializing the base vocabulary to a large number of symbols and progressively trims down each symbol to obtain a smaller vocabulary
* the base vocabulary could correspond to all pre-tokenized words and the most common substrings
* at each step: Unigram algorithm defines a loss (often defined as log-likelihood) over the training data given the current vocabulary and a unigram language model
* for each symbol in the vocabulary, the algorithm computes how mch the overall loss would increase if the symbol was to be removed from the vocabulary
* removing p percent of the symbols whose loss increase is the lowers (least affecting the overall loss over the training data)
* the process is repeated until the desired vocabulary size is reached
* since Unigram is not based on merge rules, the algorithm has several ways of tokenizing new text after training
* the algorithm simply picks the most likely tokenization in practice
* probabilities of each possible tokenization can be computed
* Unigram saves the probability of each token in the training corpus 
* Models: not used directly for any of the models, but it’s used in conjunction with SentencePiece
* TODO

<details>
<summary><b>SentencePiece</b> (Google) [2018]:</summary>

> * Paper: https://arxiv.org/abs/1808.06226
> * Github: https://github.com/google/sentencepiece

</details>

* previous tokenization methods have the same problem: it is assumed that the input text uses spaces to separate words
* not all languages use spaces to separate words
* one possible choice: using language specific tokenizer (e.g., XLM model)
* solving this problem more generally: SentencePiece
* treating the input as a raw input stream, thus including the space in the set of characters to use
* then using the BPE or unigram algorithm to construct the appropriate vocabulary
* language independent subword tokenized and detokenizer
* language agnostic approach
* 4 components:
  * Normalizer
  * Trainer
  * Encoder
  * Decoder
* Models: ALBERT, XLNet, T5

> #### Side note:
> * Tokenizers are usually trained on English datasets, or multi-language datasets where english text is overrepresented. Due to this, English text is handled more efficiently, which in practice means that the same sentence in english will be tokenized into fewer tokens than the hungarian translation of that sentence. (Suppose a translation consists of the same number of words and characters.)
> * Since LLMs make prices based on the used tokens, using a LLM (e.g. ChatGPT) is more expensive for the Hungarian language than for English.


### 3.3.2 Embeddings

Tokenization splits our text sequence into tokens (atomic parts), embedding represents these tokens in a high dimensional vector space to map to numeric representation manageable by Language Models.

> **TL;DR**: 
> The evolution of textual embeddings have 2 main phases, similarly to the applied representations in general Machine Learning.
Early NLP methods utilized human-engineered (hand-crafted) descriptors and features of text, incorporating various task-specific statistics, such as word frequencies and importance-reflecting weights.
As Deep Learning methods evolved, embeddings (vector representations of words or tokens) began to be learned through Neural Networks.
This paradigm shift has led to remarkable improvements in representation learning, enabling the derived embeddings to capture more nuanced semantic features, including abstract meanings and similarity between words, or encoding the context for addressing polymorphism.


Required features of embeddings:
* embeddings should encode abstract meaning of words
  * words with similar meaning should be close in the vector space
  * words with different meaning should be distant in the vector space
* semantic relationships should be encoded in the vector space structure
  * “Queen relates to King in the same way as Woman relates to Man”
* distributed representations
* context-dependent representations to handle polymorphism

Demo: https://www.cs.cmu.edu/~dst/WordEmbeddingDemo/

### Embedding methods:

#### <ins>Classical methods</ins>:

TODO: write a description for classical methods.
Describing the trend in ML that first hand-crafted features are used then learning the features with Deep Learning.

<details>
<summary><b>One-hot encoding</b>:</summary>

> * Video: https://www.youtube.com/watch?v=G2iVj7WKDFk

</details>

  * one-hot encoding is a fundamental technique to transform categorical data, such as tokens, into a numerical format that ML models can work with
  * each word / token in the vocabulary is assigned a unique ID, and each one is represented as a sparse vector of zeros except for the index corresponding to the ID, which is marked as 1
    * first word (token): [1, 0, 0, ...]
    * second word (token): [0, 1, 0, ...]
    * last word (token): [0, 0, 0, , ..., 1]
  * in NN models, one-hot encoded vectors are often used as the input to an embedding layer, which then maps these spase vectors into a dense, lower-dimensional space
  * Pros:
    * Simplicity: one-hot encoding is straightforward to understand and implement
  * Cons:
    * Sparsity: one-hot vectors are extremely sparse, leading to inefficient use of memory and computational resources
    * Lack of Semantic Information: one-hot encoding does not capture any semantic relationships between words
    * Scalability issues: the size of the vectors increases with the size of the vocabulary
      * vocabulary with 40,000 words (tokens) --> vectors with dimension of 40,000

<details>
<summary><b>Bag-of-Words</b> (BoW): </summary>

> * Video: https://www.youtube.com/watch?v=IRKDrrzh4dE
> * Video: https://www.youtube.com/watch?v=OGK9SHt8SWg

</details>

  * BoW is a fundamental technique used in classical NLP to convert text data into numerical form
  * unlike one-hot encoding, which represents each word as a unique vector, BoW represents text as a collection of the counts of words that appear in the document
    * disregarding the order in which the words appear
  * a document / text is represented as a bag (or multiset) of its words
    * the words are usually preprocessed
    * the grammar and the order of the words are disregarded by the BoW representation
    * multiplicities of words are kept
    * BoW only concerns whether known words occur in the document or not
    * each unique word in the entire dataset becomes a feature in the model
    * for each document, the values in its feature vector correspond to the frequencies of each word in the document
  * Pros:
    * Simplicity and Efficiency: BoW is straightforward to implement and understand
    * Scalability: efficient implementations exist that can handle large vocabularies and documents
  * Cons:
    * Loss of Context: BoW does not capture the order of words, meaning that can lose the context and meaning of words in sentences
    * Sparsity: the resulting feature vectors are often sparse (many zeros), which can be inefficient for some ML algorithms
    * High Dimensionality: with a large vocabulary, the feature space can become very large, leading to high memory and computational costs

<details>
<summary>Example</summary>

```
Demonstration of the Bag of Words (BoW) method on a toy example.

    We have a dataset (corpus) with 3 documents. Each document comprises of a sentence.
    We want to create the BoW descriptors for these documents to provide them to NNs.

 ■ Sentences / Documents:
    
    The 3 sentences in the 3 documents.
    
  - Sentence 1: "The annual software technology conference showcased the latest innovations in software and hardware."
  - Sentence 2: "Attendees of the technology conference gained insights into new software applications and digital technologies."
  - Sentence 3: "The software designed for pet management helps dog owners schedule vet appointments and track their pet's health."


 ■ Cleaning text: removing Stop Words and applying Stemming / Lemmatization
 
    As a first step we remove non-informative words (Stop Words).
    In the secont step we 'normalize' the words by reducint them to their base forms.

  - Sentence 1: ["annual", "software", "technology", "conference", "showcase", "latest", "innovation", "software", "hardware"]
  - Sentence 2: ["attendee", "technology", "conference", "gain", "insight", "new", "software", "application", "digital", "technology"]
  - Sentence 3: ["software", "design", "pet", "management", "help", "dog", "owner", "schedule", "vet", "appointment", "track", "pet", "health"]

    Now we have a list of unique words for each document.

 ■ Creating the vocabulary:
 
    We need to create the vocabulary. This contains all the root words in all the documents of the corpus.
    We will compare documents based on this vocabulary that servers as a 'basis', or as the features of our descriptor.

  - Vocabulary: ["annual", "application", "appointment", "attendee", "conference", "design", "digital", "dog", "gain", "hardware", "health", "help", "innovation", "insight", "latest", "management", "new", "owner", "pet", "schedule", "showcase", "software", "technology", "track", "vet"]

 ■ Summarizing the frequencies of words:
 
    We count how many times the words appear. Create list with unique words (aka. a set) and save the frequency of the words (how many times they appear in the document).
    We can see that we ordered the words in alhpabetical order: this descriptor removes this information.

  - Frequencies of Sentence 1: ["annual": 1, "conference": 1, "hardware": 1, "innovation": 1, "latest": 1, "showcase": 1, "software": 2, "technology": 1]
  - Frequencies of Sentence 2: ["application": 1, "attendee": 1, "conference": 1, "digital": 1, "gain": 1, "insight": 1, "new": 1, "software": 1, "technology": 2]
  - Frequencies of Sentence 3: ["appointment": 1, "design": 1, "dog": 1, "health": 1, "help": 1, "management": 1, "owner": 1, "pet": 2, "schedule": 1, "software": 1, "track": 1, "vet": 1]

 ■ Constructing BoW representations:
 
    We need not only the appearing words for describing the document, but all the words appearing in the vocabulary.
    Those words don't appear in the document gets the 0 frequancy.
    The order of the words (features) is the same for all the documents.
    The specific order doesn't matter, but it has to match for all the documents. 

  - BoW Representation of Sentence 1: ["annual": 1, "application": 0, "appointment": 0, "attendee": 0, "conference": 1, "design": 0, "digital": 0, "dog": 0, "gain": 0, "hardware": 1, "health": 0, "help": 0, "innovation": 1, "insight": 0, "latest": 1, "management": 0, "new": 0, "owner": 0, "pet": 0, "schedule": 0, "showcase": 1, "software": 2, "technology": 1, "track": 0, "vet": 0]
  - BoW Representation of Sentence 2: ["annual": 0, "application": 1, "appointment": 0, "attendee": 1, "conference": 1, "design": 0, "digital": 1, "dog": 0, "gain": 1, "hardware": 0, "health": 0, "help": 0, "innovation": 1, "insight": 1, "latest": 0, "management": 0, "new": 1, "owner": 0, "pet": 0, "schedule": 0, "showcase": 0, "software": 1, "technology": 2, "track": 0, "vet": 0]
  - BoW Representation of Sentence 3: ["annual": 0, "application": 0, "appointment": 1, "attendee": 0, "conference": 0, "design": 1, "digital": 0, "dog": 1, "gain": 0, "hardware": 0, "health": 1, "help": 1, "innovation": 1, "insight": 0, "latest": 0, "management": 1, "new": 0, "owner": 1, "pet": 2, "schedule": 1, "showcase": 0, "software": 1, "technology": 0, "track": 1, "vet": 1]

 ■ Bow representations vector:

    The BoW descriptor vectors are only the values (words / features are omitted).

  - BoW Representation of Sentence 1: [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0]
  - BoW Representation of Sentence 2: [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0] 
  - BoW Representation of Sentence 3: [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1, 1]
 ```

</details>
  
<details>
<summary><b>TF-IDF: Term Frequency - Inverse Document Frequency</b>:</summary>

> * Video: https://www.youtube.com/watch?v=t2Nq3MFK_pg
> * Demo: https://remykarem.github.io/tfidf-demo/

</details>
 
  * TF-IDF is a fundamental technique used in classical NLP to convert text data into numerical form
  * improves upon Bag of Words by considering not just the frequency of words within a single document, but also how unique these words are across all documents in the corpus
  * TF: Term Frequency:
    * measures how frequently a term occurs in a document
    * the more times a word appears in a document, the higher its term frequency
  * IDF: Inverse Document Frequency:
    * measures how important a term is within the corpus
    * words that appear in many documents have lower IDF score, as they are considered less significant
  * TF-IDF score:
    * the product of TF and IDF
    * provides a weight for each word
    * words with higher TF-iDF scores are considered more important to the document they appear in
  * Pros:
    * Relevance: TF-IDF can highlight words that are more relevant to a document's context, improving the quality of information retrieval and text mining tasks
    * Dimensionality Reduction: by prioritizing important words, TF-IDF can help reduce the dimensionality of the text data, focusing on words that offer more meaning
  * Cons:
    * Complexity: the calculation of IDF value requires knowledge of the entire corpus, making TF-IDF more complex to implement and compute than simpler models like BoW
    * Context Ignorance: although TF-IDF considers the rarity of words across documents, it still ignores the context and syntax within the text, potentially missing nuanced meanings
    * Fixed Vocabulary: similar to BoW, the TF-IDF model has a fixed vocabulary, and it can't handle new words that weren't in the training corpus without retraining

#### <ins>Deep Learning-based methods - Embeddings</ins>:

> TL;DR: TODO
> Outline the dynamics of the methods.
> * Architecture: shallow NN, deep NN (LSTM, Transformer)
> * Capabilities: meaning and similarity in vector space, global context, polysemy, shallow bidirectionality, deep bidrectionality

Distributed representations:
* add a short description which describes the deep learning based learned representations, which are distributed.
* What are the advantages of distributed representations compared to Hand-Crafted representations?

<details>
<summary><b>Word2Vec</b> (Google) [2013]: </summary>

> * Paper: https://arxiv.org/abs/1301.3781
> * Paper: https://arxiv.org/abs/1310.4546
> * Demo: https://remykarem.github.io/word2vec-demo/

</details>

  * Neural Network-based models used to produce learned word embeddings, which are vector representations of words
    * shallow neural network, not Deep Neural Network / Deep Learning
  * the learned embeddings are distributed representations
  * can capture a large number of precise syntactic and semantic word relationships simply by vector arithmetic
    * placing words that have similar meanings close to each other in a multidimensional vector space
    * the first method capturing these algebraic representations:
      * Word2Vec("Queen") = Word2Vec("King") + [Word2Vec("Woman") - Word2Vec("King")]
  * Model architecture:
    * shallow, 2-layer Neural Networks
    * trained to reconstruct linguistic contexts of words
    * 2 main architectures:
      * Continuous Bag of Words (CBoW):
        * Objective: predicting the target word based on context words surrounding it
        * given one or more context words, it predicts the target word in the middle
        * generally faster than Skip-Gram
      * Skip-Gram:
        * Objective: predicting surrounding context words for a given target word
        * given a specific word, it predicts the likelihood of other words appearing in its vicinity within a sentence
        * generally slower than CBoW
  * Pros:
    * Efficiency: quickly processes large vocabularies
    * Semantic Information: captures complex word relationships and similarities
    * Scalability: can handle massive datasets effectively
  * Cons:
    * Context Limitation: struggles with words that have multiple meanings based on context
    * Static Representations: each word has a fixed representation, not considering the word's context in different sentences
    * Data Hungry: requires a large amount of text data to perform well
    * Domain Specificity: pre-trained models may not perform well on text from specialized domains without additional training 


<details>
<summary><b>GloVe</b> (Stanford) [2014]: </summary>

> * Paper: https://aclanthology.org/D14-1162/

</details>

  * GloVe stands for Global Vectors for Word Representations
  * matrix statistics-based models used to produce learned word embeddings, which are vector representations of words
  * unlike Word2Vec, which uses local context information, GloVe constructs an explicit word-context (or word co-occurrence) matrix using statistics across the whole text corpus
  * the model applies matrix factorization techniques to approximate the matrix
    * learning to rich word embeddings that reflect both the semantic and syntactic meaning of words 
    * 2 matrix factorization methods:
      * Global Matrix Factorization:
        * capturing global statistics
        * the entire corpus is used to generate a co-occurrence matrix
          * analyzing how often words co-occur with others across the corpus to capture global statistics
      * Local context window-based method:
        * capturing local context
  * Pros:
    * Rich Word Embeddings: captures complex patterns beyond mere word co-occurrences, including semantic and syntactic nuances
    * Efficient Use of Corpus: by utilizing global co-occurrence statistics, it leverages more information than just local context, leading to more informed embeddings
    * Scalability: can process large corpora effectively
  * Cons:
    * Memory Intensive: building and storing the global co-occurrence matrix can be memory-intensive, especially for large corpora
    * Fixed Context Size: the initial co-occurrence matrix construction doesn't account for varying context sizes or dynamic context windows
    * Preprocessing Overhead: requires the construction of a co-occurrence matrix before training
    * Less Effective for Rare Words: the model's performance can degrade for words that appear infrequently in the corpus, as their co-occurrence statistics are less robust

<details>
<summary><b>CoVe</b> [2018]: </summary>

> * Paper: https://arxiv.org/abs/1708.00107

</details>

  * CoVe stands for Contextualized word Vectors
  * Deep Neural Network-based models used to produce learned word embeddings, which are vector representations of words
  * unlike (traditional) static word embedding models like Word2Vec and GloVe, CoVe provides word representations that are sensitive to the context in which a word appears
    * a step forward in capturing the nuances of languages
      * Polysemy: words with multiple meanings
      * complex syntactic structures
  * Architecture:
    * CoVe model is generated by a model that uses a Machine Translation architecture
      * Sequence-to-Sequence (Seq2Seq) model
      * Encoder-Decoder architecture
      * initially trained on Machine Translation task
      * Deep Learning-based Deep Neural Networks (DNNs) are used
    * Encoder:
      * RNN (LSTM, GRU)
      * learning to represent input sentences in a way that captures both semantic and syntactic information contextually
      * the word vectors generated by this encoder are what referred to as CoVe
  * Pros:
    * Contextual Awareness: CoVe adjusts the representation of each word based on the sentence it appears in
      * the same word can get different representations based on the actual context
      * can handle complex linguistic nuances
      * In contrast: traditional embeddings provide a single static representation for each word regardless of its context
    * Leverages Deep Learning: uses deep learning models trained on large-scale machine translation tasks, benefiting from the nuanced understanding of langauge
    * Complementary to Static Embeddings: CoVe are often used in combination with static embeddings like GloVE, providing both the efficiency of static representations and the contextual awareness of CoVe
    * Rich Representations: captures deeper semantic and syntactic information than static embeddings
  * Cons:
    * Computational Complexity: requires significant computational resources due to the complexity of the underlying models
    * Training Time: training CoVe from scratch can be time-consuming
    * Integration Complexity: integrating CoVe with existing NLP systems can be more complex than static embeddings
      * requiring adjustments to accommodate the dynamic nature of the embeddings

<details>
<summary><b>ELMo</b> [2018]: </summary>

> * Paper: https://arxiv.org/abs/1802.05365

</details>

  * ELMo stands for Embeddings from Language Models
  * Deep Neural Network (bidirectional LSTM) based model used to produce learned word embeddings, which are vector representations of words
  * Architecture:
    * bidirectional LSTM (biLSTM)
    * trained on Language Modeling objective
      * predicting the next word based on the entire context of a word (both past and future)
    * layers are stacked
      * learning layers of representation that correspond to different aspects of language
    * trained on a large text corpus
  * <ins>Pros</ins>:
    * **Contextualized Embeddings**: ELMo generates words embeddings that are context-dependent, providing rich representations that capture a wide array of syntactic and semantic information
    * **Deep Representations**: utilizing Deep Learning (biLSTMs) to model language which allows ELMo to capture complex characteristics of language
    * **Dynamic Embeddings**: ELMo's dynamic embeddings are particularly effective at handling words with multiple meanings (polysemy), providing context-specific representations
  * <ins>Cons</ins>:
    * **Shallow Concatenation of Bidirectional Representations**: the forward and backward LM-s (LSTMs) are concatenated in  a shallow way
      * in contrast: BERT applies a deep bidirectional representations
    * **Computational Requirements:** generating ELMo embeddings is computationally intensive due to the complexity of the underlying biLSTM model
    * **Increased Model Complexity**: integrating ELMo into existing models increases the overall complexity of the model, which may impact training and inference times
    * **Resource Intensity**: ELMo models require significant memory and processing power
    * TODO: why shallow?

<details>
<summary><b>BERT</b> (Google) [2018]: </summary>

> * Paper: https://arxiv.org/abs/1810.04805

</details>

  * BERT stands for Bidirectional Encoder Representations from Transformers
  * Deep Neural Network (Transformer) based model used to produce learned word embeddings, which are vector representations of words
  * represents a landmark innovation in the field of NLP
    * introducing a new paradigm for obtaining rich, contextualized word embeddings
  * deeply bidirectional representations due to the attention mechanism in the Transformer
    * previous models read the text input sequentially (either left-to-right or right-to-left)
    * BERT processes the entire sequence of words at once
      * capturing the context of a word based on all of its surroundings
  * Architecture:
    * encoder-only Transformer
    * a type of attention mechanism (self-attention) that learns contextual relations between words (tokens) in a text
  * Training:
    * pre-trained on a large corpus of text from the internet
    * trained using 2 novel pre-training tasks:
      * Masked Language Modeling (MLM):
        * randomly masking some tokens from tne input
        * predicting masked tokens based on the context
      * Next Sentence Prediction (NSP)
        * predicting whether 2 segments of text occur in sequence of not
        * understanding the relationship between sentences
  * <ins>Pros</ins>:
    * **Deep Contextualized Embeddings**: BERT provides embeddings that are deeply contextual
      * capturing subtle nuances of language
      * due to the bidirectional context, BERT considers the full context of a word by looking at the words that come before and after it
    * **State-Of-The-Art (SOTA) performance**: SOTA results on a lot of tasks
    * **Versatility**: its architecture allows it to be fine-tuned for a broad range of tasks, making it incredible versatile
      * pre-trained BERT models are available, which has been trained on vast text corpora
      * BERT can be fine-tuned with just one additional output layer for a wide range of tasks
  * <ins>Cons</ins>:
    * **Resource Intensity**: training and even fine-tuning BERT can require significant computational resources
    * **Complexity and Inference Time**: the complexity of the model can lead to longer inference times, which might be a bottleneck
    * **Understanding Model Decisions**: the complexity and size of the model can also make it challenging to understand why BERT makes certain decisions or predictions
      * complicating efforts to improve model transparency and explainability


### 3.4 Text Embeddings

> TL;DR: TODO
> What is text embedding about. In which domains are used? (RAG)


Features of Text Embeddings:
* we can turn a text with 'arbitrary' size into a fixed-size embedding vector
* by being able to measure the relatedness of texts we unlock use cases for a couple of applications:
  * e.g., Semantic Search, Anomaly Detection, Retrieval-Augmented Generation (RAG)
* Embedding vector space is structured:
  * similar or related texts are close to each other
  * embeddings are similar to meaning in mathematical techniques
* Frequently used Text-Embedding models:
  * OpenAI:
    * text-embedding-3-small
      * context size: 8191 input tokens
      * embedding dimension: 1536
      * 1$: 62,500 pages (~800 tokens per page)
    * text-embedding-3-large
      * context size: 8191 input tokens
      * embedding dimension: 3072
      * 1$: 9,615 pages (~800 tokens per page)
    * tokenizer: cl100k_base
    * tiktoken tokenizer can split text to tokens to get the number of tokens in Python
* Native support for shortening embeddings:
  * using larger embeddings: generally costs more and consumes more compute, memory and storage
  * the new embedding models were trained with a technique that allows developers to trade-off performance and cost of using embeddings
  * developers can shorten embeddings (i.e., remove some numbers from the end of the sequence) without the embedding losing its concept-representing properties by passing in the dimensions
* Implementations:
  * OpenAI: Embeddings - documentation: https://platform.openai.com/docs/guides/embeddings
  * Google - Vertex AI

### How to create a Text Embedding model:
* Naive method:
  * averaging word embeddings
* Modern method:
  * using a transformer to compute a context-aware representation of each word, then take an average of the context-aware representations

### Similarity measures for Text Embeddings:
* cosine similarity
* euclidean distance
* dot-product: cosine * length of both vectors
  * equivalent to cosine similarity when vectors are normalized to 1
* Which distance function should be used?
  * cosine similarity is recommended
  * OPenAI embeddings are normalized to length 1
  * cosine similarity can be computed slightly faster using just a dot product
  * cosine similarities and Euclidean distance will result in the identical rankings

### Visualization of Text Embeddings:
* similarity in the space
* heatmap visualization
* PCA and t-SNE

### Applications of Text Embeddings:
  * Semantic Search:
    * results are ranked by relevance to a query string
  * Clustering:
    * text strings are grouped by similarity
  * Anomaly / Outlier detection:
    * outliers with little relatedness are identified
  * Diversity measurement:
    * similarity distributions are analyzed
  * Classification:
    * text strings are classified by their most similar label
  * Recommendations:
    * items with related text strings are recommended
  * Retrieval-Augmented Generation (RAG):
    * augmenting the context of LLMs with retrieved information
  * Question Answering:
    * TODO


## 4. Additional Resources
* Videos:
  * [Google: Introduction to Large Language Models (16 min)](https://www.youtube.com/watch?v=zizonToFXDs&pp=ygUVbGFyZ2UgbGFuZ3VhZ2UgbW9kZWxz)
  * [Andrej Karpathy: Intro to Large Language Models (60 min)](https://www.youtube.com/watch?v=zjkBMFhNj_g&t=995s&pp=ygUVbGFyZ2UgbGFuZ3VhZ2UgbW9kZWxz)
  * [Google: Introduction to Generative AI (22 min)](https://www.youtube.com/watch?v=G2fqAlgmoPo)
  * [Harvard University: Large Language Models and The End of Programming (67 min)](https://www.youtube.com/watch?v=JhCl-GeT4jw)
  * [Jeff Bezos: Large Language Models (10 min)](https://www.youtube.com/watch?v=e4FXX4yX0nA)
* Papers:
  * [A survey of Large Language Models (2023)](https://arxiv.org/abs/2303.18223)

# TODO list:
* TODO: add history between demo and why so successful