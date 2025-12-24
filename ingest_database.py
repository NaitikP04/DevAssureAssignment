import os
import base64
from uuid import uuid4
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# Import .env file
from dotenv import load_dotenv
load_dotenv()

# Configuration 
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Helper function to process images
def process_image(image_path):
    print(f"Processing image: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        llm = ChatOpenAI(model="gpt-4o", max_tokens=1000)
        prompt = "Describe this UI screenshot or diagram in extreme detail for a QA engineer. Mention every button, field, error message, and layout element visible."
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]
        )
        
        response = llm.invoke([message])
        
        return Document(
            page_content=response.content,
            metadata={"source": image_path, "type": "image"}
        )
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def main():

    if os.path.exists(CHROMA_PATH):
        import shutil
        shutil.rmtree(CHROMA_PATH)  # Delete old DB to prevent duplicates
        print(f"Cleared existing database at {CHROMA_PATH}")

    # Intiate embeddings model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initiate the vector store
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH
    )

    raw_documents = []

    # Main Loading Logic
    if os.path.exists(DATA_PATH):
        for root, dirs, files in os.walk(DATA_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                try:
                    if ext == ".pdf":
                        print(f"Loading PDF: {file_path}")
                        loader = PyPDFLoader(file_path)
                        raw_documents.extend(loader.load())
                    elif ext == ".txt":
                        print(f"Loading Text: {file_path}")
                        loader = TextLoader(file_path)
                        raw_documents.extend(loader.load())
                    elif ext == ".md":
                        print(f"Loading Markdown: {file_path}")
                        loader = UnstructuredMarkdownLoader(file_path)
                        raw_documents.extend(loader.load())
                    elif ext in [".doc", ".docx"]:
                        print(f"Loading Word Document: {file_path}")
                        loader = UnstructuredWordDocumentLoader(file_path)
                        raw_documents.extend(loader.load())
                    elif ext in [".png", ".jpg", ".jpeg"]:
                        doc = process_image(file_path)
                        if doc:
                            raw_documents.append(doc)
                    else:
                        print(f"Skipping unsupported file: {file_path}")
                except Exception as e:
                    print(f"Failed to load {file_path}: {e}")
    else:
        print(f"Data directory not found: {DATA_PATH}")

    # split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )

    # Creating the chunks
    if raw_documents:
        chunks = text_splitter.split_documents(raw_documents)
        print(f"Split {len(raw_documents)} documents into {len(chunks)} chunks.")

        # Creating the unique IDs for each chunk
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        # Adding the chunks to the vector store
        vector_store.add_documents(documents=chunks, ids=uuids)
        print("Documents added to vector store.")
    else:
        print("No documents to add.")

if __name__ == "__main__":
    main()