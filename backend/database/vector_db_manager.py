"""
Persistent Vector Database Manager for AI Multi-Source Summarizer
Handles FAISS vector database persistence and management
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = logging.getLogger(__name__)

class VectorDBManager:
    """Manages persistent vector database storage and retrieval"""
    
    def __init__(self, db_path: str = "vector_db"):
        """
        Initialize the Vector DB Manager
        
        Args:
            db_path: Path to store vector databases
        """
        self.db_path = os.path.abspath(db_path)
        self.ensure_db_directory()
        
        # Metadata file to track databases
        self.metadata_file = os.path.join(self.db_path, "db_metadata.json")
        self.load_metadata()
    
    def ensure_db_directory(self):
        """Ensure the vector database directory exists"""
        os.makedirs(self.db_path, exist_ok=True)
        
        # Create subdirectories for organization
        subdirs = ["pdfs", "youtube", "manual", "combined"]
        for subdir in subdirs:
            os.makedirs(os.path.join(self.db_path, subdir), exist_ok=True)
    
    def load_metadata(self):
        """Load database metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """Save database metadata"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def generate_db_id(self, content_source: str, content_identifier: str) -> str:
        """
        Generate a unique database ID based on content
        
        Args:
            content_source: Source type (pdf, youtube, manual)
            content_identifier: Unique identifier (filename, video_id, etc.)
        
        Returns:
            Unique database ID
        """
        # Create hash of content identifier for uniqueness
        content_hash = hashlib.md5(content_identifier.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{content_source}_{content_hash}_{timestamp}"
    
    def save_vector_db(self, 
                      vector_store: FAISS, 
                      documents: List[Document],
                      content_info: Dict[str, Any]) -> str:
        """
        Save vector database to disk
        
        Args:
            vector_store: FAISS vector store
            documents: Original documents
            content_info: Metadata about the content
        
        Returns:
            Database ID
        """
        try:
            # Generate database ID
            content_source = content_info.get("source_type", "unknown")
            content_identifier = content_info.get("identifier", str(datetime.now().timestamp()))
            db_id = self.generate_db_id(content_source, content_identifier)
            
            # Create database directory
            db_dir = os.path.join(self.db_path, content_source, db_id)
            os.makedirs(db_dir, exist_ok=True)
            
            # Save FAISS vector store
            vector_store.save_local(db_dir)
            logger.info(f"FAISS vector store saved to {db_dir}")
            
            # Save documents separately
            documents_file = os.path.join(db_dir, "documents.pkl")
            with open(documents_file, 'wb') as f:
                pickle.dump(documents, f)
            logger.info(f"Documents saved to {documents_file}")
            
            # Save content info
            content_file = os.path.join(db_dir, "content_info.json")
            content_info_with_timestamp = {
                **content_info,
                "created_at": datetime.now().isoformat(),
                "db_id": db_id,
                "db_path": db_dir,
                "document_count": len(documents)
            }
            
            with open(content_file, 'w', encoding='utf-8') as f:
                json.dump(content_info_with_timestamp, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            self.metadata[db_id] = content_info_with_timestamp
            self.save_metadata()
            
            logger.info(f"Vector database saved successfully with ID: {db_id}")
            return db_id
            
        except Exception as e:
            logger.error(f"Error saving vector database: {e}")
            raise
    
    def load_vector_db(self, db_id: str, embeddings) -> Optional[Dict[str, Any]]:
        """
        Load vector database from disk
        
        Args:
            db_id: Database ID to load
            embeddings: Embeddings model for loading FAISS
        
        Returns:
            Dictionary with vector_store, documents, and content_info
        """
        try:
            if db_id not in self.metadata:
                logger.error(f"Database ID {db_id} not found in metadata")
                return None
            
            db_info = self.metadata[db_id]
            db_path = db_info["db_path"]
            
            if not os.path.exists(db_path):
                logger.error(f"Database path {db_path} does not exist")
                return None
            
            # Load FAISS vector store
            vector_store = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
            logger.info(f"FAISS vector store loaded from {db_path}")
            
            # Load documents
            documents_file = os.path.join(db_path, "documents.pkl")
            with open(documents_file, 'rb') as f:
                documents = pickle.load(f)
            logger.info(f"Documents loaded from {documents_file}")
            
            # Load content info
            content_file = os.path.join(db_path, "content_info.json")
            with open(content_file, 'r', encoding='utf-8') as f:
                content_info = json.load(f)
            
            return {
                "vector_store": vector_store,
                "documents": documents,
                "content_info": content_info
            }
            
        except Exception as e:
            logger.error(f"Error loading vector database {db_id}: {e}")
            return None
    
    def list_databases(self) -> List[Dict[str, Any]]:
        """
        List all available databases
        
        Returns:
            List of database information
        """
        databases = []
        for db_id, info in self.metadata.items():
            # Check if database still exists
            if os.path.exists(info.get("db_path", "")):
                databases.append({
                    "db_id": db_id,
                    "source_type": info.get("source_type", "unknown"),
                    "name": info.get("name", db_id),
                    "created_at": info.get("created_at", "unknown"),
                    "document_count": info.get("document_count", 0),
                    "content_length": info.get("content_length", 0)
                })
        
        # Sort by creation date (newest first)
        databases.sort(key=lambda x: x["created_at"], reverse=True)
        return databases
    
    def delete_database(self, db_id: str) -> bool:
        """
        Delete a database and its files
        
        Args:
            db_id: Database ID to delete
        
        Returns:
            Success status
        """
        try:
            if db_id not in self.metadata:
                logger.error(f"Database ID {db_id} not found")
                return False
            
            db_info = self.metadata[db_id]
            db_path = db_info["db_path"]
            
            # Remove database files
            if os.path.exists(db_path):
                import shutil
                shutil.rmtree(db_path)
                logger.info(f"Deleted database directory: {db_path}")
            
            # Remove from metadata
            del self.metadata[db_id]
            self.save_metadata()
            
            logger.info(f"Database {db_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting database {db_id}: {e}")
            return False
    
    def get_database_info(self, db_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific database
        
        Args:
            db_id: Database ID
        
        Returns:
            Database information
        """
        return self.metadata.get(db_id)
    
    def cleanup_orphaned_databases(self):
        """Remove metadata entries for databases that no longer exist on disk"""
        orphaned_ids = []
        
        for db_id, info in self.metadata.items():
            db_path = info.get("db_path", "")
            if not os.path.exists(db_path):
                orphaned_ids.append(db_id)
        
        for db_id in orphaned_ids:
            logger.info(f"Removing orphaned database metadata: {db_id}")
            del self.metadata[db_id]
        
        if orphaned_ids:
            self.save_metadata()
            logger.info(f"Cleaned up {len(orphaned_ids)} orphaned database entries")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        total_size = 0
        database_count = len(self.metadata)
        total_documents = 0
        
        # Calculate directory size
        for root, dirs, files in os.walk(self.db_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        
        # Count total documents
        for info in self.metadata.values():
            total_documents += info.get("document_count", 0)
        
        return {
            "database_count": database_count,
            "total_documents": total_documents,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "storage_path": self.db_path
        }
