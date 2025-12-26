import os
import shutil
from uuid import uuid4

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from ingestion.loaders import load_documents
from ingestion.image_processor import process_image
from ingestion.chunker import chunk_documents

# Import .env file
from dotenv import load_dotenv
load_dotenv()

# Configuration 
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "example_collection"

def main():

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Delete old DB to prevent duplicates
        print(f"[Ingest] Cleared existing DB at {CHROMA_PATH}")

    # Intiate embeddings model
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # Initiate the vector store
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings_model,
        persist_directory=CHROMA_PATH
    )

    # Load documents from the data directory
    raw_documents = load_documents(DATA_PATH)

    # Load images separately
    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in {".png", ".jpg", ".jpeg"}:
                image_path = os.path.join(root, file)
                doc = process_image(image_path)
                if doc:
                    raw_documents.append(doc)
    
    if not raw_documents:
        print("[Ingest] No documents found.")
        return
                
    # chunk the documents
    chunks = chunk_documents(raw_documents)
    print(f"[Ingest] Created {len(chunks)} chunks")

    # Deduplicate (simple but effective)
    seen = set()
    unique_chunks = []
    for c in chunks:
        key = c.page_content.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_chunks.append(c)
    
    # Add chunks to the vector store
    ids = [str(uuid4()) for _ in unique_chunks]
    vector_store.add_documents(unique_chunks, ids=ids)

    print(f"[Ingest] Stored {len(unique_chunks)} unique chunks")


if __name__ == "__main__":
    main()