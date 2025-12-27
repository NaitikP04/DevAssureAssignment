"""
Database Manager Module

Provides CRUD operations for the ChromaDB vector store.
Allows viewing, filtering, and deleting documents from the database.
"""

import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from langchain_chroma import Chroma


@dataclass
class FileInfo:
    """Information about a file stored in the database."""
    source: str
    filename: str
    doc_type: str
    chunk_count: int
    is_runtime_upload: bool
    session_id: Optional[str] = None
    upload_timestamp: Optional[str] = None


class DatabaseManager:
    """
    Manages the ChromaDB vector store.
    
    Provides functionality to:
    - List all stored files and their metadata
    - Delete files and all associated chunks
    - Get database statistics
    - Filter by runtime vs preprocessed files
    """
    
    def __init__(self, vector_store: Chroma):
        """
        Initialize the database manager.
        
        Args:
            vector_store: ChromaDB vector store instance
        """
        self.vector_store = vector_store
    
    def get_all_files(self) -> List[FileInfo]:
        """
        Get information about all files in the database.
        
        Returns:
            List of FileInfo objects, one per unique source file
        """
        try:
            data = self.vector_store.get()
        except Exception as e:
            print(f"[DatabaseManager] Error getting data: {e}")
            return []
        
        if not data or not data.get('metadatas'):
            return []
        
        # Group by source file
        file_chunks: Dict[str, List[Dict]] = defaultdict(list)
        
        for metadata in data['metadatas']:
            source = metadata.get('source', 'Unknown')
            file_chunks[source].append(metadata)
        
        # Build FileInfo list
        files = []
        for source, chunks in file_chunks.items():
            first_chunk = chunks[0]
            
            # Extract filename from path
            filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
            
            file_info = FileInfo(
                source=source,
                filename=filename,
                doc_type=first_chunk.get('doc_type', 'unknown'),
                chunk_count=len(chunks),
                is_runtime_upload=first_chunk.get('runtime_upload', False),
                session_id=first_chunk.get('session_id'),
                upload_timestamp=first_chunk.get('upload_timestamp')
            )
            files.append(file_info)
        
        # Sort: runtime uploads first, then by filename
        files.sort(key=lambda f: (not f.is_runtime_upload, f.filename.lower()))
        
        return files
    
    def get_preprocessed_files(self) -> List[FileInfo]:
        """Get only preprocessed (non-runtime) files."""
        return [f for f in self.get_all_files() if not f.is_runtime_upload]
    
    def get_runtime_files(self) -> List[FileInfo]:
        """Get only runtime-uploaded files."""
        return [f for f in self.get_all_files() if f.is_runtime_upload]
    
    def delete_file(self, source: str) -> bool:
        """
        Delete a file and all its chunks from the database.
        
        Args:
            source: The source path of the file to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Get all document IDs for this source
            data = self.vector_store.get(where={"source": source})
            
            if not data or not data.get('ids'):
                print(f"[DatabaseManager] No documents found for source: {source}")
                return False
            
            ids_to_delete = data['ids']
            
            # Delete from ChromaDB
            self.vector_store.delete(ids=ids_to_delete)
            
            print(f"[DatabaseManager] Deleted {len(ids_to_delete)} chunks for: {source}")
            return True
            
        except Exception as e:
            print(f"[DatabaseManager] Error deleting {source}: {e}")
            return False
    
    def delete_files(self, sources: List[str]) -> Dict[str, bool]:
        """
        Delete multiple files from the database.
        
        Args:
            sources: List of source paths to delete
            
        Returns:
            Dict mapping source to success status
        """
        results = {}
        for source in sources:
            results[source] = self.delete_file(source)
        return results
    
    def delete_all_runtime_uploads(self) -> int:
        """
        Delete all runtime-uploaded documents.
        
        Returns:
            Number of chunks deleted
        """
        try:
            data = self.vector_store.get(where={"runtime_upload": True})
            
            if not data or not data.get('ids'):
                return 0
            
            ids_to_delete = data['ids']
            self.vector_store.delete(ids=ids_to_delete)
            
            print(f"[DatabaseManager] Deleted {len(ids_to_delete)} runtime upload chunks")
            return len(ids_to_delete)
            
        except Exception as e:
            print(f"[DatabaseManager] Error deleting runtime uploads: {e}")
            return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dict with statistics about stored documents
        """
        try:
            data = self.vector_store.get()
        except Exception as e:
            return {"error": str(e)}
        
        if not data or not data.get('metadatas'):
            return {
                "total_chunks": 0,
                "total_files": 0,
                "preprocessed_files": 0,
                "runtime_files": 0,
                "doc_types": {}
            }
        
        metadatas = data['metadatas']
        sources: Set[str] = set()
        runtime_sources: Set[str] = set()
        doc_types: Dict[str, int] = defaultdict(int)
        
        for meta in metadatas:
            source = meta.get('source', 'Unknown')
            sources.add(source)
            
            if meta.get('runtime_upload', False):
                runtime_sources.add(source)
            
            doc_type = meta.get('doc_type', 'unknown')
            doc_types[doc_type] += 1
        
        return {
            "total_chunks": len(metadatas),
            "total_files": len(sources),
            "preprocessed_files": len(sources - runtime_sources),
            "runtime_files": len(runtime_sources),
            "doc_types": dict(doc_types)
        }
    
    def format_files_for_display(self) -> str:
        """
        Format file list for Gradio display.
        
        Returns:
            Markdown-formatted string of all files
        """
        files = self.get_all_files()
        
        if not files:
            return "📭 **No files in database**\n\nUpload files or run `ingest_database.py` to add documents."
        
        stats = self.get_statistics()
        
        lines = [
            f"**Database Contents:** {stats['total_files']} files, {stats['total_chunks']} chunks\n",
            "---"
        ]
        
        # Group by type
        runtime_files = [f for f in files if f.is_runtime_upload]
        preprocessed_files = [f for f in files if not f.is_runtime_upload]
        
        if runtime_files:
            lines.append("\n**🔄 Runtime Uploads:**")
            for f in runtime_files:
                icon = self._get_doc_icon(f.doc_type)
                lines.append(f"- {icon} `{f.filename}` ({f.chunk_count} chunks)")
        
        if preprocessed_files:
            lines.append("\n**📁 Preprocessed Files:**")
            for f in preprocessed_files:
                icon = self._get_doc_icon(f.doc_type)
                lines.append(f"- {icon} `{f.filename}` ({f.chunk_count} chunks)")
        
        return "\n".join(lines)
    
    def get_file_choices(self) -> List[str]:
        """
        Get list of file sources for dropdown selection.
        
        Returns:
            List of source paths
        """
        files = self.get_all_files()
        return [f.source for f in files]
    
    def _get_doc_icon(self, doc_type: str) -> str:
        """Get an icon for a document type."""
        icons = {
            "pdf": "📄",
            "text": "📝",
            "prd": "📋",
            "doc": "📃",
            "image": "🖼️"
        }
        return icons.get(doc_type, "📎")
