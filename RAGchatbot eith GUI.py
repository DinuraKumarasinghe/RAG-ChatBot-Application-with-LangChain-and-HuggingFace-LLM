%%writefile rag_chat_app.py
import os
import re
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import threading

class RagChatBot:
    def __init__(self):
        # Set Hugging Face API Token
        os.environ["HF_TOKEN"] = "API"  # Replace with your token
        
        self.initialize_llm()
        self.load_documents()
        self.create_vector_store()
        self.build_chain()
    
    def initialize_llm(self):
        self.llm = ChatOpenAI(
            model="deepseek-ai/DeepSeek-R1",
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"],
            max_tokens=256,
            streaming=True,
            temperature=0.3
        )
    
    def load_documents(self):
        loader = PyPDFLoader("Profile.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        self.splits = text_splitter.split_documents(docs)
    
    def create_vector_store(self):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = Chroma.from_documents(documents=self.splits, embedding=embedding_model)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def build_chain(self):
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
        
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def get_response(self, question):
        try:
            response = self.chain.invoke(question)
            return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        except Exception as e:
            return f"⚠️ Error: {str(e)}"

class ChatApplication:
    def __init__(self, bot):
        self.bot = bot
        self.chat_history = []
        self.setup_ui()
        
    def setup_ui(self):
        # Create output area for chat history
        self.output = widgets.Output(layout={'border': '1px solid gray', 'height': '400px', 'overflow_y': 'scroll'})
        
        # Create input widgets
        self.input_field = widgets.Textarea(
            placeholder='Type your message here...',
            layout={'width': '100%', 'height': '80px'},
            disabled=False
        )
        
        self.send_button = widgets.Button(
            description='Send',
            button_style='success',
            layout={'width': '100px', 'margin': '0 0 0 10px'}
        )
        self.send_button.on_click(self.send_message)
        
        self.clear_button = widgets.Button(
            description='Clear',
            button_style='warning',
            layout={'width': '100px', 'margin': '0 0 0 5px'}
        )
        self.clear_button.on_click(self.clear_chat)
        
        # Create input container
        input_container = widgets.HBox(
            [self.input_field, self.send_button, self.clear_button],
            layout={'align_items': 'flex-end', 'margin': '10px 0 0 0'}
        )
        
        # Create header
        header = widgets.HTML(
            "<h2 style='text-align:center; color:#2c3e50;'>PDF Chat Assistant</h2>"
            "<p style='text-align:center; color:#7f8c8d;'>Ask questions about your PDF document</p>"
            "<hr style='border-top: 1px solid #ecf0f1;'>"
        )
        
        # Assemble UI
        self.ui = widgets.VBox(
            [header, self.output, input_container],
            layout={'width': '95%', 'margin': '20px auto', 'border': '1px solid #bdc3c7', 'padding': '20px'}
        )
        
        # Display initial message
        with self.output:
            print("🤖 Welcome to the PDF Chat Assistant! Ask anything about your document.")
        
        display(self.ui)
    
    def send_message(self, button):
        user_input = self.input_field.value.strip()
        if not user_input:
            return
            
        self.input_field.value = ''
        self.input_field.disabled = True
        self.send_button.disabled = True
        
        # Add user message to chat
        self.chat_history.append(('user', user_input))
        self.update_chat_display()
        
        # Process in background thread
        threading.Thread(target=self.process_bot_response, args=(user_input,)).start()
    
    def process_bot_response(self, user_input):
        try:
            response = self.bot.get_response(user_input)
            self.chat_history.append(('bot', response))
        except Exception as e:
            self.chat_history.append(('bot', f"⚠️ Error: {str(e)}"))
        
        # Update UI from main thread
        self.update_chat_display()
        self.input_field.disabled = False
        self.send_button.disabled = False
    
    def update_chat_display(self):
        self.output.clear_output()
        with self.output:
            for sender, message in self.chat_history:
                if sender == 'user':
                    print(f"👤 You: {message}\n")
                else:
                    print(f"🤖 Assistant: {message}\n")
                    print("─" * 80)
    
    def clear_chat(self, button):
        self.chat_history = []
        self.output.clear_output()
        with self.output:
            print("🤖 Chat history cleared. Ask a new question.")

# Initialize and run the app
if __name__ == "__main__":
    # First run: Install required packages
    print("Installing required packages...")
    !pip install -q langchain-openai langchain langchain-community langchain-chroma huggingface_hub chromadb pypdf sentence-transformers

    # Initialize the bot
    bot = RagChatBot()
    
    # Create and display the chat interface
    app = ChatApplication(bot)
