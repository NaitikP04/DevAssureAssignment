from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

# Import .env file
from dotenv import load_dotenv
load_dotenv()

# Configuration 
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

# Intiate embeddings model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH
)

# Loading the PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)

raw_documents = loader.load()

# split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

# Creating the chunks
chunks = text_splitter.split_documents(raw_documents)

# Creating the unique IDs for each chunk
uuids = [str(uuid4()) for _ in range(len(chunks))]

# Adding the chunks to the vector store
vector_store.add_documents(documents=chunks, ids=uuids)