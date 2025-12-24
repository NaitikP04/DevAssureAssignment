import gradio as gr
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from dotenv import load_dotenv

# Import .env file
load_dotenv()

# Configuration 
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Intiate embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Intiate the language model
llm = ChatOpenAI(temperature=0.5, model="gpt-4o")

# Connect to ChromaDB
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

# Setup Retrieval
try:
    # We need to fetch documents for BM25
    print("Initializing Hybrid Search...")
    
    # Get all documents from Chroma to initialize BM25
    # Note: In a production system with millions of docs, you wouldn't load all into memory like this.
    # You would use a persistent BM25 store or a search engine like Elasticsearch/Opensearch.
    existing_data = vector_store.get()
    
    if existing_data['documents']:
        docs = [
            Document(page_content=txt, metadata=meta) 
            for txt, meta in zip(existing_data['documents'], existing_data['metadatas'])
        ]
        
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 5
        
        chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        print("Hybrid Search (BM25 + Chroma) initialized successfully.")
    else:
        print("No documents found in Chroma. Defaulting to basic vector search.")
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
except Exception as e:
    print(f"Failed to initialize Hybrid Search: {e}")
    print("Falling back to standard Vector Search.")
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def user(user_message, history):
    # Ensure user_message is a string
    if isinstance(user_message, list):
        user_message = user_message[0] if user_message else ""
    return "", history + [{"role": "user", "content": user_message}]

def bot(history):
    user_message = history[-1]["content"]
    
    # Ensure user_message is a string for retrieval
    if isinstance(user_message, dict):
        user_message = user_message.get("text", user_message.get("content", ""))
    elif isinstance(user_message, list):
        user_message = user_message[0] if user_message else ""
    
    # Convert to string if not already
    user_message = str(user_message)
        
    # Retrieve relevant chunks
    docs = retriever.invoke(user_message)
    
    # Format Debug Context and Knowledge
    retrieved_context_str = ""
    knowledge = ""
    
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        # Create a snippet (first 100 chars)
        content_snippet = doc.page_content[:100].replace("\n", " ")
        
        retrieved_context_str += f"[{i+1}] Source: {source}\nSnippet: {content_snippet}...\n\n"
        knowledge += doc.page_content + "\n\n"
    
    # Construct the prompt
    rag_prompt = f"""
    You are an Assistant which answers questions based on knowledge which is provided to you.
    While answering, you don't use your internal knowledge,
    but solely the information in the "The knowledge" section below.
    You don't need to mention anything to the user about the provided knowledge.

    The question is: {user_message}

    Conversation history: {history[:-1]}

    The knowledge: {knowledge}
    """

    # Stream the response - append assistant message
    history.append({"role": "assistant", "content": ""})
    for response in llm.stream(rag_prompt):
        history[-1]["content"] += response.content
        yield history, retrieved_context_str

# UI Setup with gr.Blocks
with gr.Blocks(title="DevAssure Chatbot") as demo:
    gr.Markdown("# DevAssure RAG Chatbot")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=600, label="Chat")
            msg = gr.Textbox(placeholder="Ask a question about your documents...", label="Your Message")
            clear = gr.Button("Clear Chat")
            
        with gr.Column(scale=1):
            debug_box = gr.Textbox(label="Debug Context (Retrieved Chunks)", lines=20, interactive=False)
    
    # Event Handling
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, debug_box]
    )
    
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
