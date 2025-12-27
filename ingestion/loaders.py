import os
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SUPPORTED_DOC_EXTS = {".pdf", ".txt", ".md", ".doc", ".docx"}


def load_single_file(file_path):
    """
    Load a single document file and return list of Documents.
    
    Args:
        file_path: Path to the file to load
        
    Returns:
        List of Document objects, or None if loading fails
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            _tag_docs(docs, file_path, "pdf")

        elif ext == ".txt":
            loader = TextLoader(file_path)
            docs = loader.load()
            _tag_docs(docs, file_path, "text")

        elif ext == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            _tag_docs(docs, file_path, "prd")

        elif ext in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            _tag_docs(docs, file_path, "doc")

        else:
            return None

        return docs

    except Exception as e:
        print(f"[Loader] Failed to load {file_path}: {e}")
        return None


def load_documents(data_path):
    """
    Load all documents from a directory (recursive).
    
    Args:
        data_path: Path to directory containing documents
        
    Returns:
        List of Document objects
    """
    raw_documents = []

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

            # Skip non-document files (images handled separately)
            if ext not in SUPPORTED_DOC_EXTS:
                continue

            docs = load_single_file(file_path)
            if docs:
                raw_documents.extend(docs)

    return raw_documents


def _tag_docs(docs, source, doc_type):
    for d in docs:
        d.metadata["source"] = source
        d.metadata["doc_type"] = doc_type