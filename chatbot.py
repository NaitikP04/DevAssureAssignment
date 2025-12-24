from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr

# Import .env file
from dotenv import load_dotenv
load_dotenv()

# Configuration 
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Intiate embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Intiate the language model
llm = ChatOpenAI(temperature=0.5, model="gpt-5.1")

# Connect to ChromaDB
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

# Set up the vector store retriever
num_results = 5 # Max number of results to retrieve
retriever = vector_store.as_retriever(search_kwargs={"k": num_results})

# Call this function for every message added to the chatbot
def stream_response(message, history):
    #print(f"Input message: {message}". History: {history}\n")

    # Retreive relevant chunks based on the questions asked
    docs = retriever.invoke(message)

    # Add all chunks to knowledge
    knowledge = ""

    for doc in docs:
        knowledge += doc.page_content + "\n\n"
    
    # Make the call to the LLM (including the prompt)
    if message is not None:

        partial_message = ""

        rag_prompt = f"""
        You are an Assistant which answers questions based on knowledge which is provided to you.
        While answering, you don't use your internal knowledge,
        but solely the information in the "The knowledge" section below.
        You don't need to mention anything to the user about the provided knowledge.

        The question is: {message}

        Conversation history: {history}

        The knowledge: {knowledge}

        """

        #print(rag_prompt)

        # Stream the response to the gradio chatbot
        for response in llm.stream(rag_prompt):
            partial_message += response.content
            yield partial_message
    
# Set up the Gradio interface
chatbot = gr.ChatInterface(stream_response, textbox=gr.Textbox(placeholder="Send to LLM...",
    container=False,
    autoscroll=True,
    scale=7),
)

# Launch the Gradio app
chatbot.launch()
