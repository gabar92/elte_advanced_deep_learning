# RAG Retriever

## Description

Implement a Retriever module as part of a RAG (Retrieval-Augmented Generation) framework.  
The retriever should be capable of retrieving relevant documents from a given dataset based on a query.

The topic of Retrieval-Augmented Generation (RAG) was planned for the last lecture, which unfortunately was cancelled.  
However, we have already discussed parts of the RAG pipeline (especially the Retrieval part) in previous lectures.  
Thus, this assignment offers a great opportunity to deepen your understanding of the RAG framework, an important topic with many real-world applications and up-to-date research developments.

Leverage the tools made available by AI to assist you in solving this assignment!

The goal of this task is to gain hands-on experience with the RAG framework by using tools that facilitate learning and problem-solving — but **not** by letting AI solve the problem entirely for you.

## Parts of the Assignment

The task consists of multiple parts:

1. **Prompting an LLM-based application** (e.g., ChatGPT, Grok, etc.) to construct the background knowledge for the task:
   1. A description of the RAG system.
   2. An elaboration on the retriever module.
   3. Identification of trends in the field.
   4. Overview of state-of-the-art solutions.
   5. List of the most popular frameworks for implementation.
   6. Discussion of the challenges in the field.
   7. Typical (hyper)parameters to consider for the retriever module.
   8. Any additional information you consider important for the task/field.
2. **Generating pseudocode** for the retriever module.
3. **Implementing** the retriever module in Python.
4. **Using** the retriever module with the provided sample dataset of tales and **creating a brief report** on the results.
5. **Documenting** the challenges and design decisions encountered during the implementation, and how they were resolved.

### Assessment

This task is worth **15 points**.

### Hints

Although the task may seem complex, it should not be overthought.

- You should not spend more than **20–30 minutes** on steps 1 and 2 (the AI-assisted background research and pseudocode).
- You should not spend more than **5–10 minutes** on the report for steps 4 and 5.
- If you get stuck with the implementation (step 3) and spend more than **1 hour**, please contact me for assistance.

### Dates

- **Release date:** April 28, 2025
- **Due date:** May 16, 2025

---

Here’s the corrected, cleaned-up, and properly formatted version of your second text block, keeping your original ideas and structure intact:

---

## The Process of the Retrieval Stage

1. Load textual documents (6 tales in Hungarian) from the corpus folder.
2. Embed the documents using a Text Embedding model.
3. Get a query from the user and embed it.
4. Calculate the similarity between the query and the documents.
5. Return the most relevant document based on the similarity scores.

> You may deviate from the process described above, but your final implementation must maintain the same overall functionality.

This task can be implemented either by using low-level tools like **NumPy** for similarity search 
(which has the advantage of letting you observe all stages closely) 
or by using a framework (e.g., **LlamaIndex**, **LangChain**, etc.), 
where the task can be completed in just a few lines — and you get familiar with industry-standard tools. 

It is up to you to decide which approach you prefer.

---

## Artifacts of the Assignment

1. A saved or shared version of the AI-assisted background research (e.g., a link or a PDF file).
   - Some applications allow you to share the conversation directly, or you can simply copy and paste it.
2. Python code of the solution.
3. A brief (half-page) report discussing the emerging challenges, design decisions, and inference results.
