import gradio as gr
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import FlashrankRerank
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
llm = ChatOpenAI(
    temperature=0.1, 
    model="gpt-4o",
    model_kwargs={"response_format": {"type": "json_object"}}
)

# Connect to ChromaDB
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

# Setup Retrieval
try:
    # We need to fetch documents for BM25
    print("Initializing Hybrid Search with Reranking...")
    
    # Get all documents from Chroma to initialize BM25
    existing_data = vector_store.get()
    
    if existing_data['documents']:
        docs = [
            Document(page_content=txt, metadata=meta) 
            for txt, meta in zip(existing_data['documents'], existing_data['metadatas'])
        ]
        
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 10
        
        chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )

        # Setup Reranker 
        # This uses a tiny BERT model to re-score the docs and only keep the truly relevant ones
        compressor = FlashrankRerank(top_n=5) # Only keep top 5 actually relevant ones
        
        # Final Retriever Pipeline
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=ensemble_retriever
        )
        
        print("Reranking initialized.")
    else:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
except Exception as e:
    print(f"Reranking setup failed: {e}. Falling back to basic search.")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})


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
    
    # Construct the prompt for JSON test case generation
    rag_prompt = f"""
You are an expert QA Engineer. Your goal is to generate test cases based strictly on the provided context.
Output format: JSON ONLY. Do not include markdown formatting like ```json ... ```.

Required JSON Structure:
[
  {{
    "use_case_title": "...",
    "preconditions": "...",
    "steps": ["1. ...", "2. ..."],
    "expected_results": "...",
    "negative_test_cases": ["..."],
    "boundary_test_cases": ["..."],
    "source_file_reference": "..."
  }}
]

Rules:
- Include a "test_data" field in the JSON structure ONLY if specific data inputs are needed for the test case.
- Do not hallucinate features not present in the context.
- Base your test cases mainly on the provided knowledge below.
- Each test case must reference the source file it came from.

User Request: {user_message}

Provided Knowledge:
{knowledge}

Generate the test cases in strict JSON format now.
    """

    # Collect the full response (non-streaming for JSON validation)
    response = llm.invoke(rag_prompt)
    response_content = response.content
    
    # Validate and pretty-print JSON
    try:
        parsed_json = json.loads(response_content)
        formatted_json = json.dumps(parsed_json, indent=2)
        final_response = formatted_json
    except json.JSONDecodeError as e:
        final_response = f"Error: Invalid JSON returned by LLM.\n\nRaw Response:\n{response_content}\n\nJSON Error: {str(e)}"
    
    # Append assistant message with formatted response
    history.append({"role": "assistant", "content": final_response})
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
