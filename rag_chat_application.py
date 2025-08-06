import os
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import re

# Set Hugging Face API Token
os.environ["HF_TOKEN"] = "Hugging Face API key"  # Replace with your token

# Initialize Hugging Face-compatible LLM
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1",
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
    max_tokens=256,
    streaming=True,
    temperature=0.3
)

# Load PDF document
loader = PyPDFLoader("Profile.pdf")
docs = loader.load()

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# Create vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define prompt template
template = """
You are a friendly assistant helping the user based on the context below.
Answer the user's question in a natural, conversational tone.
Don't include any internal thoughts or explanations — just respond clearly and helpfully.

{question}

Context:
{context}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Terminal interaction
def main():
    print("📘 Ask any question based on the PDF. Type 'exit' to quit.\n")
    while True:
        user_input = input("❓ Your question: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Exiting.")
            break
        try:
            response = chain.invoke(user_input)
            # Remove <think>...</think> block from DeepSeek
            cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            print(f"\n🧠 Answer:\n{cleaned_response}\n{'='*60}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")


if __name__ == "__main__":
    main()
