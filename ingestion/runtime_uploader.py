"""
Runtime File Upload Module

Handles processing of files uploaded during chatbot sessions.
Supports documents (PDF, TXT, MD, DOC/DOCX) and images (PNG, JPG, JPEG).

Key Features:
- File size and count validation
- On-the-fly processing, chunking, and embedding
- Metadata tagging for easy cleanup
"""

import os
import tempfile
import time
from datetime import datetime
from typing import List, Tuple, Optional
from uuid import uuid4

from langchain_core.documents import Document

from ingestion.loaders import load_single_file, SUPPORTED_IMAGE_EXTS
from ingestion.image_processor import process_image
from ingestion.chunker import chunk_documents


# ============ CONFIGURATION ============
MAX_FILES_PER_UPLOAD = 5          # Maximum files in a single upload
MAX_FILE_SIZE_MB = 10             # Maximum size per file in MB
MAX_TOTAL_SIZE_MB = 50            # Maximum total size per session
SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".doc", ".docx",
    ".png", ".jpg", ".jpeg"
}


class UploadResult:
    """Result of a file upload operation."""
    
    def __init__(self):
        self.success: bool = False
        self.processed_files: List[str] = []
        self.failed_files: List[Tuple[str, str]] = []  # (filename, error)
        self.chunks_created: int = 0
        self.processing_time: float = 0.0
        self.warnings: List[str] = []
        self.errors: List[str] = []
    
    def to_status_message(self) -> str:
        """Generate a human-readable status message."""
        if not self.success and not self.processed_files:
            return f"❌ Upload failed: {'; '.join(self.errors)}"
        
        msg_parts = []
        
        if self.processed_files:
            msg_parts.append(f"✅ Processed {len(self.processed_files)} file(s)")
            msg_parts.append(f"📦 Created {self.chunks_created} chunks")
            msg_parts.append(f"⏱️ Time: {self.processing_time:.2f}s")
        
        if self.failed_files:
            failed_names = [f[0] for f in self.failed_files]
            msg_parts.append(f"⚠️ Failed: {', '.join(failed_names)}")
        
        if self.warnings:
            msg_parts.append(f"⚠️ Warnings: {'; '.join(self.warnings)}")
        
        return "\n".join(msg_parts)


class RuntimeUploader:
    """
    Handles runtime file uploads for the RAG chatbot.
    
    Processes uploaded files immediately, chunks them,
    and stores them in the vector database with appropriate metadata.
    """
    
    def __init__(self, vector_store, embeddings_model):
        """
        Initialize the uploader.
        
        Args:
            vector_store: ChromaDB vector store instance
            embeddings_model: OpenAI embeddings model
        """
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model
        self.session_upload_size = 0  # Track total uploaded size in session
    
    def validate_files(self, file_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate uploaded files against size and type constraints.
        
        Returns:
            Tuple of (valid_paths, error_messages)
        """
        valid_paths = []
        errors = []
        
        # Check file count
        if len(file_paths) > MAX_FILES_PER_UPLOAD:
            errors.append(f"Too many files. Maximum {MAX_FILES_PER_UPLOAD} files per upload.")
            return [], errors
        
        total_size = 0
        
        for path in file_paths:
            filename = os.path.basename(path)
            ext = os.path.splitext(filename)[1].lower()
            
            # Check extension
            if ext not in SUPPORTED_EXTENSIONS:
                errors.append(f"'{filename}': Unsupported file type '{ext}'")
                continue
            
            # Check file size
            try:
                size_bytes = os.path.getsize(path)
                size_mb = size_bytes / (1024 * 1024)
                
                if size_mb > MAX_FILE_SIZE_MB:
                    errors.append(f"'{filename}': File too large ({size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit)")
                    continue
                
                total_size += size_mb
                
            except OSError as e:
                errors.append(f"'{filename}': Cannot read file - {e}")
                continue
            
            valid_paths.append(path)
        
        # Check total size
        if total_size + self.session_upload_size > MAX_TOTAL_SIZE_MB:
            remaining = MAX_TOTAL_SIZE_MB - self.session_upload_size
            errors.append(f"Total upload size ({total_size:.1f}MB) exceeds session limit. {remaining:.1f}MB remaining.")
            return [], errors
        
        return valid_paths, errors
    
    def process_and_store(self, file_paths: List[str], session_id: str = None) -> UploadResult:
        """
        Process uploaded files and store them in the vector database.
        
        Args:
            file_paths: List of paths to uploaded files
            session_id: Optional session identifier for tracking
            
        Returns:
            UploadResult with processing details
        """
        result = UploadResult()
        start_time = time.time()
        
        if not file_paths:
            result.errors.append("No files provided")
            return result
        
        # Validate files
        valid_paths, validation_errors = self.validate_files(file_paths)
        result.errors.extend(validation_errors)
        
        if not valid_paths:
            return result
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid4())[:8]
        
        upload_timestamp = datetime.now().isoformat()
        all_documents = []
        
        # Process each file
        for file_path in valid_paths:
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1].lower()
            
            try:
                if ext in SUPPORTED_IMAGE_EXTS:
                    # Process image 
                    doc = process_image(file_path)
                    if doc:
                        # Add runtime upload metadata
                        doc.metadata.update({
                            "runtime_upload": True,
                            "session_id": session_id,
                            "upload_timestamp": upload_timestamp,
                            "original_filename": filename
                        })
                        all_documents.append(doc)
                        result.processed_files.append(filename)
                    else:
                        result.failed_files.append((filename, "Image processing failed"))
                else:
                    # Process document
                    docs = load_single_file(file_path)
                    if docs:
                        for doc in docs:
                            doc.metadata.update({
                                "runtime_upload": True,
                                "session_id": session_id,
                                "upload_timestamp": upload_timestamp,
                                "original_filename": filename
                            })
                        all_documents.extend(docs)
                        result.processed_files.append(filename)
                    else:
                        result.failed_files.append((filename, "Document loading failed"))
                        
            except Exception as e:
                result.failed_files.append((filename, str(e)))
        
        if not all_documents:
            result.errors.append("No documents could be processed")
            return result
        
        # Chunk documents
        try:
            chunks = chunk_documents(all_documents)
            result.chunks_created = len(chunks)
        except Exception as e:
            result.errors.append(f"Chunking failed: {e}")
            return result
        
        # Store in vector database
        try:
            ids = [str(uuid4()) for _ in chunks]
            self.vector_store.add_documents(chunks, ids=ids)
            
            # Update session upload size
            for path in valid_paths:
                self.session_upload_size += os.path.getsize(path) / (1024 * 1024)
            
            result.success = True
            
        except Exception as e:
            result.errors.append(f"Database storage failed: {e}")
            return result
        
        result.processing_time = time.time() - start_time
        return result
    
    def reset_session(self):
        """Reset session upload tracking."""
        self.session_upload_size = 0


def get_upload_limits_info() -> str:
    """Get a formatted string describing upload limits."""
    exts = ", ".join(sorted(SUPPORTED_EXTENSIONS))
    return f"""**Upload Limits:**
- Max files per upload: {MAX_FILES_PER_UPLOAD}
- Max file size: {MAX_FILE_SIZE_MB}MB
- Max total per session: {MAX_TOTAL_SIZE_MB}MB
- Supported types: {exts}"""
