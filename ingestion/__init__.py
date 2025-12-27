"""
Ingestion Module

Handles document loading, processing, chunking, and runtime uploads.
"""

from ingestion.loaders import load_documents, load_single_file, SUPPORTED_IMAGE_EXTS, SUPPORTED_DOC_EXTS
from ingestion.chunker import chunk_documents
from ingestion.image_processor import process_image
from ingestion.runtime_uploader import RuntimeUploader, UploadResult, get_upload_limits_info
from ingestion.database_manager import DatabaseManager, FileInfo

__all__ = [
    # Loaders
    "load_documents",
    "load_single_file",
    "SUPPORTED_IMAGE_EXTS",
    "SUPPORTED_DOC_EXTS",
    # Chunker
    "chunk_documents",
    # Image processor
    "process_image",
    # Runtime uploader
    "RuntimeUploader",
    "UploadResult",
    "get_upload_limits_info",
    # Database manager
    "DatabaseManager",
    "FileInfo",
]
