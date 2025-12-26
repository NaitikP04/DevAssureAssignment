import os
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader
)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

def load_documents(data_path):
    raw_documents = []

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()

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
                    continue

                raw_documents.extend(docs)

            except Exception as e:
                print(f"[Loader] Failed to load {file_path}: {e}")

    return raw_documents


def _tag_docs(docs, source, doc_type):
    for d in docs:
        d.metadata["source"] = source
        d.metadata["doc_type"] = doc_type