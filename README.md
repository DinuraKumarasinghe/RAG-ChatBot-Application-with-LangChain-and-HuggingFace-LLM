# RAG Q&A Chatbot using DeepSeek + LangChain + HuggingFace

Welcome to my very first Retrieval-Augmented Generation (RAG) application!  
This chatbot can **read any file** and answer your questions based on its content — just like a smart assistant trained on your own documents.

---

## 🚀 What is this project?

This is a terminal-based AI chatbot that:
- Reads a PDF document (like a resume, report, or article)
- Breaks it into chunks and creates embeddings (using HuggingFace)
- Stores them in a vector database (ChromaDB)
- Uses a powerful LLM (DeepSeek-R1) to answer any question based on the content

All powered by **LangChain**, **HuggingFace**, and **DeepSeek**!

---

## 🧠 Technologies Used

| Tool / Library           | Purpose                                   |
|--------------------------|-------------------------------------------|
| LangChain                | Building the RAG pipeline                 |
| DeepSeek-R1 (HuggingFace)| Answer generation (LLM)                   |
| Sentence Transformers    | Creating embeddings from PDF content      |
| ChromaDB                 | Vector store for efficient search         |
| PyPDFLoader              | Reading PDF files                         |
| Python                   | The main language used                    |

---

## 📁 Project Structure
📁 my-rag-chatbot/
 │
 ├── app.py                      # 🧠 Main script with RAG chain and chatbot loop
 ├── Profile.pdf                 # 📄 The PDF document used for question answering
 ├── requirements.txt            # 📦 Python dependencies
 ├── README.md                   # 📘 Project documentation (this file)



---

##✅ Skills Demonstrated
 - Retrieval-Augmented Generation (RAG)
 - Prompt engineerin
 - LangChain architecture
 - Embedding models & vector databases
 - Python scripting
 - Frontend integration (Streamlit/Gradio)
 - HuggingFace Inference API
