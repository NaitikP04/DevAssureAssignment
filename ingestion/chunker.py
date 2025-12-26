from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    chunks = []

    for doc in documents:
        doc_type = doc.metadata.get("doc_type", "text")

        if doc_type == "prd":
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n## ", "\n# ", "\n"],
                chunk_size=700,
                chunk_overlap=100
            )

        elif doc_type == "pdf":
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100
            )

        elif doc_type == "image":
            chunks.append(doc)
            continue

        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )

        doc_chunks = splitter.split_documents([doc])

        for i, c in enumerate(doc_chunks):
            c.metadata["chunk_index"] = i
            chunks.append(c)

    return chunks
