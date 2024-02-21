# Lecture 1: Introduction and Foundations

## What is this lecture about?

This lecture tries to answer the following questions:

1. What is the current state of Natural Language Processing (NLP) across various applications?
2. How do we define Natural Language Processing (NLP)?
3. What has led to the rapid adoption of NLP-based applications?
4. What are the key tasks and associated applications in NLP?
5. What are the challenges and advantages of NLP, along with common solutions?
7. How Computer Vision (CV) and NLP inspired each other's best practices?
8. What is the foundational principle underlying nearly every Language Model?
9. How textual data is represented for Language Model processing?


## Contents:
* [Motivation](#motivation)
  * [Demo of recent NLP related applications](#demo-of-recent-nlp-related-applications)
  * [Definition of Natural Language Processing (NLP)](#definition-of-natural-langauge-process-nlp)
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

### Demo of recent NLP-related applications

The goal of this section is to provide inspiration and motivation by presenting the latest AI advancements through showcasing a selection of cutting-edge NLP-related applications.
What particularly makes NLP successful is its ability to go beyond mere text processing by utilizing natural language, enabling seamless integration with other modalities (image, sound) as an effective interface.

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
        Modifying images based on description.
      </p>
    </td>
    <td style="padding: 5px;">
      <a href="https://www.youtube.com/watch?v=DvBRj--sUMU">
        <img src="https://img.youtube.com/vi/DvBRj--sUMU/0.jpg" style="width: 100%; height: auto;">
      </a>
      <p style="text-align: center;">
        Adobe FireFly<br>
        Generating images described by text.<br>
        Modifying images based on description.
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


### Definition of Natural Langauge process (NLP)

#### Interdisciplinarity of NLP:

The Natural Language Processing (NLP) domain sites at the junction of the fields of:
* **Computer Science**: the study of computer systems, algorithms with the goal of developing technologies and applications to solve complex problems.
* **Artificial Intelligence**: the study of creating machines capable of performing tasks that normally require human intelligence.
* **Linguistics**: the study of language and its structure, including the analysis of syntax, semantics, and more, with the aim of understanding how languages are formed, used, and change over time.

#### Informal definition of NLP:

The goal of NLP: enabling computers to understand, interpret, and generate human language.
* *Understanding* language: grasping the meaning of words, phrases, or larger units of text.
* *Interpreting* language: extracting deeper meaning, context, or intent in text and comprehending it.
* *Generating* text: producing human-like text

High-quality NLP systems, through the provision of natural language interfaces, enable humans to seamlessly and efficiently leverage computers for a diverse range of useful tasks.

### Reasons behind the success of NLP

For those who are not deeply involved in the field, with the arrival of ChatGPT it might seem like Artificial Intelligence appeared out of nowhere, and immediately has been integrated into our daily routines.
However, the development of AI has been a more gradual journey. So, what gives the impression of this abrupt leap in its utilization?
What are the novel features that Large Language Models (LLMs) have brought to the table that were missing in prior advancements?

Just to highlight one qualitative measure of ChatGPT's success: it reached 100 million users faster than any previous application.
(At the moment, Meta's Threads leads the leaderboard.)

Key factors contributing to the success include:
* Unlike Computer Vision (CV), NLP-based applications offer the most natural and convenient means of interaction through natural language.
* The simplicity of non-technical communication eliminates the barrier of requiring specialized knowledge for usage.
* The capability to perform new tasks, such as creating photo-realistic images on any topic without the need for expert Photoshop skills, fosters a sense of achievement and unleashes creativity.
* Almost any domain can be approached with Natural Language. By enhancing the NLP abilities, we can improve the adaptation of infinite many fields.
* Support for multiple languages for the same tasks enhances accessibility and global reach.
* 
* The performance of LLMs has achieved human lever on many tasks. [TODO]
* No expertise is needed for LLMs only natural prompting to customize the model to our usage. [TODO]
* Lastly, the ability of Large Language Models (LLMs) to tackle surprisingly complex tasks and provide sophisticated solutions not just in predefined domains but on customized tasks with available adaptation methods. [TODO]


## Bird's Eye View

### NLP-specific challenges

Here, we introduce and walk around a couple of challenges specific to NLP, some of them are being present in other domains as well, some are unique to this field.
The goal is to give reason and understanding why different de-facto models and principles evolved.

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

* **General pre-trained model**:


## Converging paths: adopting techniques between NLP and CV

The fields of Natural Language Processing (NLP) and Computer Vision (CV) each come with their unique strengths and challenges, leading to the creation of distinct techniques and solutions tailored to their situation.
Over time, these domain-specific approaches have been shared and adapted between the two fields.
Here, we delve into a few techniques that have been shared and adapted between these 2 fields, highlighting their background, motivation, and cross-domain application:

<ins>CV techniques adopted in NLP</ins>:
* **Two-stage training procedure: Pre-training then Fine-tuning**:
  * The two-stage training procedure was popularized in CV with the development of models like AlexNet and VGG.
The network is first pre-trained on a large, generic dataset (like ImageNet) and then fine-tuned on a smaller, domain-specific dataset.
This approach leverages the generic features learned during pre-training, which are applicable across a wide-range of visual tasks.
This methodology was later adopted by the NLP community with models like BERT and GPT.
Here, language models are pre-trained on vast amounts of text data to learn a general understanding of language and then fine-tuned for specific tasks.
  * Differences between the two domains:
    * In CV both the pre-training and fine-tuning are supervised learning (classification).
    * However, in NLP the pre-training is usually unsupervised or self-supervised learning (next word prediction, or missing word prediction) while the fine-tuning is supervised (downstream task’s objective).
Fine-tuning in CV usually affects an additional linear layer at the top of the backbone model (?). In NLP, fine-tuning extends for the entire network.

<ins>NLP techniques adopted in CV</ins>:
* **Unsupervised / Self-Supervised Learning**:
  * Unsupervised and Self-Supervised Learning in NLP involves learning patterns from unlabelled text data.
Since there is abundant text on the web which can be used to learn general language modeling during a pre-training phase.
Frequently used objectives are language modeling (next word prediction) (GPT) and masked language modeling (BERT).
  * Unsupervised and Self-Supervised Learning techniques found their way into CV as well.
Techniques like Contrastive Learning (CL), where the model learns by comparing pairs of images to understand if they are similar or different, have shown great promise in learning robust visual representations without the need for labeled data.
* **Transformer architecture**:
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


<details>
<summary><b><ins>References</ins></b></summary>

* Hugging Face: https://huggingface.co/tasks
* Papers with Code: https://paperswithcode.com/area/natural-language-processing

</details>


## Details

### Classical methods

There are a couple of terms that are general enough to collect them into a glossary and describe separately.
Here comes those terms which are general enough to be worth to introduce prior to others, outdated, or could not put into the stream of the following sections.

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

<details>
<summary><b><ins>Tokenizer demo</ins></b></summary>

* https://platform.openai.com/tokenizer
* https://llmtokencounter.com/
* https://tokens-lpj6s2duga-ew.a.run.app/

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
* Special tokens:
  * \<UNK>: representing unknown tokens
  * \<SEP>: separating different parts of the sequence
  * \<BOS>: indicating the beginning of the sequence / sentence
  * \<EOS>: indicating the end of the sequence / sentence


### Tokenization methods

<details>
<summary><b>Byte Pair Encoding (BPE) [2015]:</b></summary>

* Paper: https://arxiv.org/abs/1508.07909
* A blogpost: https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10

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

**WordPiece** (Google) [2016]:
* Literature:
  * Paper: https://arxiv.org/abs/1609.08144v2
  * Blogpost: https://blog.research.google/2021/12/a-fast-wordpiece-tokenization-system.html?m=1
  * Demo:
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

**Unigram** (Google) [2018]:
* Literature:
  * Paper: https://arxiv.org/abs/1804.10959
  * Blogpost:
  * Demo:
* BPE and WordPiece is based on merge rules
* a subword tokenization algorithm
* initializing the base vocabulary to a large number of symbols and progressively trims down each symbol to obtain a smaller vocabulary
* the base vocabulary could correspond to all pre-tokenized words and the most common substrings
* at each step: Unigram algorithm defines a loss (often defined as log-likelihood) over the training data given the current vocabulary and a unigram language model
* for each symbol in the vocabulary, the algorithm computes how mch the overall loss would increase if the symbol was to be removed from the vocabulary
* removing p percent of the symbols whose loss increase is the lowers (least affecting the overall loss over the training data)
* the process is repeated until the desired vocabulary size is reached
* since Unigram is not based on merge rules, the algorithm has several ways of tokenizing new text after training
* the algorithm simply picks the most likely tokenization in practive
* probabilities of each possible tokenization can be computed
* Unigram saves the probability of each token in the training corpus 
* Models: not used directly for any of the models, but it’s used in conjunction with SentencePiece
* TODO

**SentencePiece** (Google) [2018]:
* Literature:
  * Paper: https://arxiv.org/abs/1808.06226
  * Github: https://github.com/google/sentencepiece
  * Demo:
* previous tokenizations have the same problem: it is assumed that the input text uses spaces to separate words
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
* One-hot encoding:
  *  
  * does not really make sense, but demonstrates the problem with sparse representations
    * no meaning encoded in any way
    * not scalable: vocabulary with 40,000 words → vectors with size of 40,000 dimensions
    * sparse
  * distributed representations are preferred 

#### Classical methods:
* TF-IDF: Term Frequency - Inverse Document Frequency
  * Resources:
    * Demo: https://remykarem.github.io/tfidf-demo/
  * a numerical statistic
  * reflecting how important a word is to a document in a collection or corpus
  * Term Frequency: the number of times a term occurs in a document
  * Inverse Document Frequency: diminishes the weight of terms that occur very frequently in the document set and increases the weight of terms that occur rarely
  * Application:
  * information retrieval: query to a text
* n-gram:
  * Resources:
    * Paper: TODO
    * Demo: https://www.reuneker.nl/files/ngram/
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
* Bag-of-Words (BoW):
  * TODO
* BM25
  * Resources:
    * Paper: TODO
  * Best Match 25
  * ranking algorithm
* Skip-gram?
  * wikipedia: TODO

#### Deep Learning-based methods:
* Word2Vec (Google - 2013):
  * Resources:
    * Paper: TODO 
    * Demo: https://remykarem.github.io/word2vec-demo/
  * first method captioning algebraic representation
  * using shallot feed-forward networks
  * 2 methods:
    * Continuous Bag of Words (CBOW)
      * predicting a word based on its context
    * Continuous Skip-Gram model
      * predicting the context of a given word
  * NN models to learn representations (mappings)
    * representations are distributed
* GloVe (Stanford - 2014):
  * Resources:
      * Paper: TODO
  * GloVe = Global Vectors for Word Representations
  * 2 methods:
    * Global Matrix Factorization
    * Local context window-based method
* CoVe (2018):
  * Resources:
      * Paper: TODO
  * CoVe = Contextualized word Vectors
  * encoder-decoder architecture
    * Encoder: 2-layer bidirectional LSTM
    * Decoder: 
  * attentional sequence-to-sequence model
  * used with GloVe (?) concatenated
* ELMo (2018):
  * Resources:
      * Paper: TODO
  * Embeddings from Language Models
  * deep contextualized word representation
    * polysemy: same word can get different representations based on its context
    * the representation of a token is the function of the entire input sequence
  * model: bidirectional LM (LSTM)
    * forward LM (LSTM)
    * backward LM (LSTM)
    * shallow concatenation (compared to BERT)
    * deep representations:
      * linear function of all the internal layers of the biLM model
* BERT:
  * Resources:
      * Paper: TODO
  * transformer-based solution
  * deeply bidirectional

### Text embeddings

> TL;DR: TODO
> What is text embedding about. In which domains are used? (RAG)


Implementations:
* OpenAI: Embeddings - documentation: https://platform.openai.com/docs/guides/embeddings


* turning not only tokens but larger segment of text (sentence, text-chunk, document) into an embedding vector
* unlocking use cases like search
* we can measure the relatedness of texts
* assigning an embedded vector with fixed size to an arbitrary text
* similar to meaning in mathematical techniques
* An embedding is a vector (list) of floating point numbers
* the distance between two vectors measure their relatedness
* embedding methods:
  * length:
    * 1536 for text-embedding-3-small
    * 3072 for text-embedding-3-large
    * vertexai.language_models import TextEmbeddingModel
      * vector dimension: 768
  * embedding models:
  * V3 and V2 generations: 8191 max input tokens, tokenizer: cl100k_base, September 2021
  * priced per input token: 1$ (~800 tokens per page):
    * text-embedding-3-large: 62 500 pages
    * text-embedding-3-small: 9 615 pages
    * tiktoken tokenizer can split text to tokens to get the number of tokens in Python


Application:
  * Search: results are ranked by relevance to a query string
  * Clustering: text strings are grouped by similarity
  * Anomaly / Outlier detection: outliers with little relatedness are identified
  * Diversity measurement: similarity distributions are analyzed
  * Classification: text strings are classified by their most similar label
  * Semantic Search:
  * Recommendations: items with related text strings are recommended

Demo for use cases:
 * Question answering using embeddings-based search
 * Text search using embeddings
 * Recommendations using embeddings
 * Classification using the embedding features
 * Clustering
 * RAG
 

Manipulation and usage embeddings, Metrics: measuring similarity
* cosine similarity
* euclidean distance
* dot-product: cosine * length of both vectors
  * equivalent to cosine similarity when vectors are normalized to 1
* Which distance function should be used?
  * cosine similarity is recommended
  * OPenAI embeddings are normalized to length 1
  * cosine similarity can be computed slightly faster using just a dot product
  * cosine similarities and Euclidean distance will result in the identical rankings

Embedding demo:
* similarity in the space
* heatmap visualization
* PCA and t-SNE


How we create text-embedding models? 
* Simple method: averaging word embeddings
  * naive approach
* Modern method: using a transformer to compute a context-aware representation of each word, then take an average of the context-aware representations

Features of text embeddings:
* Native support for shortening embeddings:
* using larger embeddings: generally costs more and consumes more compute, memory and storage
* the new embedding models were trained with a technique that allows developers to trade-off performance and cost of using embeddings
* developers can shorten embeddings (i.e., remove some numbers from the end of the sequence) without the embedding losing its concept-representing properties by passing in the dimensions
