[Backend]
# 📌 Project: RAG_APP (Retrieval-Augmented Generation System)

## 📖 Overview

**RAG_APP** is a Retrieval-Augmented Generation (RAG) system that allows
users to upload documents, process them into embeddings, store them in a
vector database, and retrieve relevant information to generate accurate,
context-aware responses using an LLM.

The system follows a standard RAG pipeline:

**Document → Chunking → Embeddings → Vector Store → Retrieval → LLM
Response**

------------------------------------------------------------------------

# 🛠️ Technology Stack

## 🔹 AI Frameworks

-   **LangChain** -- Orchestrates the RAG pipeline
-   **LangGraph** -- Manages LLM interaction workflows

## 🔹 LLM Provider

-   **Groq AI**

## 🔹 Model Used

-   **llama-3.3-70b-versatile**
    -   70B parameters → high-quality responses
    -   Optimized for general tasks such as:
        -   Chat
        -   Reasoning
        -   Coding
        -   RAG applications

## 🔹 Backend API

-   **FastAPI**

## 🔹 Vector Database

-   **ChromaDB**

## 🔹 Other Libraries

-   Pydantic
-   Redis (for caching/session handling)
-   Logging
-   python-dotenv (.env)

## 🔹 Versioning

-  Git & Git hub
------------------------------------------------------------------------

# ⚙️ System Workflow

## 📥 1. Ingestion Pipeline

The ingestion process converts documents into searchable vector
embeddings.

### Steps:

1.  **Document Loading**
    -   The system reads input documents.
2.  **Text Chunking**
    -   Documents are split into smaller chunks using:
        -   `chunk_size = 700`
        -   `chunk_overlap = 150`
3.  **Embedding Generation**
    -   Each chunk is converted into embeddings using:
        -   **Sentence Transformer Model:** `all-MiniLM-L6-v2`
4.  **Vector Storage**
    -   Generated embeddings are stored in **ChromaDB** for efficient
        similarity search.

------------------------------------------------------------------------

## 🔎 2. Retrieval Pipeline

When a user submits a query:

### Steps:

1.  **Query Embedding**
    -   The user query is converted into an embedding.
2.  **MMR Search**
    -   ChromaDB retrieves relevant chunks using:
        -   **MMR (Maximal Marginal Relevance)**

    This ensures results are:
    -   Relevant to the query
    -   Diverse (non-redundant)
3.  **Similarity Matching**
    -   Embedding similarity search identifies the most relevant
        document chunks.

> ⚠️ Note: BM25-based retrieval could further improve keyword-based
> search performance.

------------------------------------------------------------------------

## 💬 3. Response Generation (Chat Pipeline)

1.  Retrieved document chunks are passed as **context** to the LLM.
2.  The **Groq-hosted Llama-3.3-70B-Versatile model** generates a
    structured and context-aware response.

------------------------------------------------------------------------
<img width="1556" height="930" alt="image" src="https://github.com/user-attachments/assets/164ae1c9-42ad-4f3b-ad47-5a481cc37c3f" />
<img width="1851" height="555" alt="image" src="https://github.com/user-attachments/assets/dc3e9ec6-fa26-40c9-b185-686d1ce52935" />
<img width="1806" height="815" alt="image" src="https://github.com/user-attachments/assets/20d74274-2a0d-45fc-abff-edc3e2473c46" />


