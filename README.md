<div align="center">

# 🔎 RAG Pipeline: Chat with Your Documents

**A step-by-step Retrieval-Augmented Generation (RAG) pipeline built with LangChain, Sentence Transformers, ChromaDB, and an LLM.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-orange)
![Embeddings](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-green)
![LLM](https://img.shields.io/badge/LLM-Groq%20%7C%20Claude-purple)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Example](#-example)
- [Notes and Limitations](#-notes-and-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🔍 Overview

Large language models can only answer from what they learned during training. **Retrieval-Augmented Generation (RAG)** fixes this by first *retrieving* relevant passages from your own documents and then handing them to the LLM as context, so answers are grounded in your data.

This project builds a complete RAG system from scratch in a single notebook. It ingests PDF and text files, splits them into chunks, converts them into vector embeddings, stores them in a persistent **ChromaDB** vector store, and answers questions by retrieving the most relevant chunks and passing them to an LLM.

The sample knowledge base includes the research paper *"Attention Is All You Need"* (`research.pdf`) and a short overview of Python (`Python.txt`).

---

## ✨ Features

- 📄 Loads **PDF** files (`PyPDFLoader`) and **text** files (`TextLoader`) into LangChain `Document` objects
- 📂 Batch ingestion: loads every PDF from a folder automatically
- ✂️ Smart chunking with `RecursiveCharacterTextSplitter` (500 characters, 50 overlap)
- 🧠 Local embeddings with Sentence Transformers (`all-MiniLM-L6-v2`, 384 dimensions)
- 💾 Persistent vector storage with **ChromaDB**
- 🎯 Semantic retrieval with top-k search and a similarity score threshold
- 🤖 LLM integration through **Groq** (`qwen/qwen3-32b`), with **Claude** (`langchain-anthropic`) also set up
- 🧩 Clean, class-based design: `EmbeddingManager`, `VectorStoreManager`, `RAGRetriever`

---

## 🏗 Architecture

```
                ┌────────────────────── Ingestion ──────────────────────┐
  PDFs / TXT →  │  Load  →  Chunk  →  Embed  →  Store in ChromaDB        │
                └────────────────────────────────────────────────────────┘

                ┌────────────────────── Retrieval + Generation ──────────┐
  User query →  │  Embed query → Similarity search → Top-k chunks        │
                │        → Build prompt (context + query) → LLM → Answer │
                └────────────────────────────────────────────────────────┘
```

| Stage       | Component                         | Role                                          |
| ----------- | --------------------------------- | --------------------------------------------- |
| Load        | `PyPDFLoader`, `TextLoader`       | Convert files into `Document` objects         |
| Chunk       | `RecursiveCharacterTextSplitter`  | Split pages into overlapping text chunks      |
| Embed       | `EmbeddingManager`                | Turn text into 384-dimensional vectors        |
| Store       | `VectorStoreManager`              | Persist vectors and metadata in ChromaDB      |
| Retrieve    | `RAGRetriever`                    | Find the most similar chunks for a query      |
| Generate    | `generate_output()` + LLM         | Answer using the retrieved context            |

---

## 🛠 Tech Stack

| Category            | Technology                                   |
| ------------------- | -------------------------------------------- |
| Language            | Python                                       |
| Framework           | LangChain (`langchain`, `langchain-core`, `langchain-community`) |
| Document Loading    | PyPDF, PyMuPDF                               |
| Text Splitting      | `langchain-text-splitters`                   |
| Embeddings          | Sentence Transformers (`all-MiniLM-L6-v2`)   |
| Vector Database     | ChromaDB (persistent client)                 |
| Similarity Metrics  | scikit-learn                                 |
| LLM Providers       | Groq (`langchain-groq`), Anthropic Claude (`langchain-anthropic`) |
| Environment         | Jupyter Notebook                             |

---

## 📁 Project Structure

```
RAG_pipeline/
│
├── RAG_pipeline.ipynb     # Complete pipeline: ingestion → retrieval → generation
├── Python.txt             # Sample text document
├── research.pdf           # Sample PDF ("Attention Is All You Need")
├── pdfs/                  # Folder of PDFs to ingest (used by load_all_pdfs)
├── vector_store/          # ChromaDB persistent storage (created automatically)
└── README.md              # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com/) (or an Anthropic API key if you use Claude)

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/anujp9806-wq/RAG_pipeline.git
cd RAG_pipeline
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install langchain langchain-core langchain-community langchain-text-splitters \
            pypdf pymupdf sentence-transformers chromadb scikit-learn \
            langchain-groq langchain-anthropic jupyter
```

---

## 🔐 Configuration

The notebook needs an LLM API key. **Never commit a real key to GitHub.**

Set it as an environment variable:

```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_key_here"

# macOS / Linux
export GROQ_API_KEY="your_key_here"
```

Then read it in the notebook instead of hard-coding it:

```python
import os
API_Key_GROQ = os.getenv("GROQ_API_KEY")
```

To use Claude instead, set `ANTHROPIC_API_KEY` and use `ChatAnthropic` from `langchain_anthropic`.

---

## 💡 Usage

1. Put the PDFs you want to search in the `pdfs/` folder.
2. Launch the notebook:

   ```bash
   jupyter notebook RAG_pipeline.ipynb
   ```

3. Run the cells in order to ingest, embed, and store your documents.
4. Ask a question:

   ```python
   answer = generate_output("What is encoder-decoder?", rag_retriever, llm)
   print(answer)
   ```

---

## 🧭 Pipeline Walkthrough

### 1. Document Loading

Each source becomes a LangChain `Document` with `page_content` and `metadata`. `load_all_pdfs()` scans the `pdfs/` folder and loads every `.pdf`. In the sample run it loaded **2 PDFs (32 pages)**.

### 2. Chunking

```python
chunks = split_docs(all_pdf_documents, chunk_size=500, chunk_overlap=50)
```

The 32 pages were split into **320 chunks**. The overlap keeps sentences that fall on a boundary from losing context.

### 3. Embeddings

`EmbeddingManager` wraps the `all-MiniLM-L6-v2` Sentence Transformer, which maps each chunk to a **384-dimensional** vector.

### 4. Vector Store

`VectorStoreManager` creates a persistent ChromaDB collection (`pdf_documents`) in the `vector_store/` folder. Each chunk is saved with a unique ID, its embedding, its text, and its metadata, so the data survives between sessions.

### 5. Retrieval

`RAGRetriever.retrieve(query, top_k=5, score_threshold=0.0)` embeds the query, searches ChromaDB, converts distance to a similarity score (`1 - distance`), and returns ranked results with metadata.

### 6. Generation

`generate_output()` joins the top-k chunks into a context block, builds a prompt with the question, and sends it to the LLM.

```python
llm = ChatGroq(
    groq_api_key=API_Key_GROQ,
    model="qwen/qwen3-32b",
    temperature=0.1,
    max_tokens=1024
)
```

---

## 🧪 Example

**Query**

```
What is encoder-decoder?
```

**Retrieved context (excerpt from the paper)**

> The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack...

**Result**

The LLM uses these retrieved passages to explain that an encoder-decoder architecture has an encoder that processes the input sequence and a decoder that generates the output, with attention connecting the two.

---

## ⚠️ Notes and Limitations

- **Duplicate entries:** ChromaDB persists between runs, and each run assigns new random IDs. Re-running the ingestion cell adds the same chunks again (the notebook's collection grew from 1,920 to 2,240 documents). Delete the `vector_store/` folder to start clean, or use deterministic IDs.
- **API key:** the notebook uses a placeholder key variable. Load real keys from environment variables and keep them out of version control.
- **Reasoning tags:** Qwen3 models may include `<think>...</think>` reasoning text in the response. Strip it if you only want the final answer.
- **Retrieval only:** the answer is only as good as the retrieved chunks. Small chunk sizes can split ideas across chunks.
- **No source citations yet:** metadata is stored but not shown alongside the generated answer.

---

## 🔮 Future Improvements

- [ ] Add source and page citations to answers
- [ ] Deterministic chunk IDs to prevent duplicates
- [ ] Support more file types (DOCX, CSV, web pages)
- [ ] Add a re-ranking step for better retrieval quality
- [ ] Build a chat interface with Streamlit or Gradio
- [ ] Add conversation memory for follow-up questions
- [ ] Evaluate retrieval quality (hit rate, MRR)

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 🙏 Acknowledgements

- [LangChain](https://python.langchain.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [Groq](https://groq.com/)
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (sample document)

---

## 👤 Author

**Anuj P**
GitHub: [@anujp9806-wq](https://github.com/anujp9806-wq)

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
