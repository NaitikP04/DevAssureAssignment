from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Point to your existing database
CHROMA_PATH = r"chroma_db"
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

db = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

print(f"Total documents in DB: {db._collection.count()}")

# Peek at the data (fetch first 3 items)
results = db.get(limit=15, include=["metadatas", "documents"])

print("\n--- SAMPLE CHUNKS ---")
for i in range(len(results["ids"])):
    print(f"\nSource: {results['metadatas'][i].get('source')}")
    print(f"Content Preview: {results['documents'][i][:200]}...") # Show first 200 chars
    print("-" * 50)