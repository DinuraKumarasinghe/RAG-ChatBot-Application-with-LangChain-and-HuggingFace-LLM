# RAG Q&A Chatbot using DeepSeek + LangChain + HuggingFace

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the content of a PDF file. It uses the `DeepSeek-R1` large language model from HuggingFace, vector embeddings from `sentence-transformers/all-mpnet-base-v2`, and is built using the LangChain framework.

## 📄 Features

- RAG pipeline to combine retrieval + generation
- HuggingFace LLM (`DeepSeek-R1`) via router
- Embedding using `sentence-transformers/all-mpnet-base-v2`
- Simple terminal-based interface
- Automatic cleaning of special tokens (e.g., `<think>...</think>`)
- Built with LangChain components

## 🛠️ Requirements

- Python 3.8+
- Hugging Face account and API key
- Dependencies (install via pip):

```bash
pip install langchain langchain-openai langchain-community langchain-chroma \
    sentence-transformers chromadb pypdf transformers
