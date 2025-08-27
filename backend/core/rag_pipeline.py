"""
RAG (Retrieval-Augmented Generation) Pipeline for content summarization and Q&A
Supports both Google Gemini and OpenAI models
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import streamlit as st
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
import tiktoken

# Import vector database manager
from backend.database.vector_db_manager import VectorDBManager

# Import both Gemini and OpenAI integrations
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    try:
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.chat_models import ChatOpenAI
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 4000
    temperature: float = 0.3
    model_name: str = "gemini-1.5-flash"
    embedding_model: str = "models/embedding-001"
    max_chunks_for_summary: int = 10
    max_chunks_for_qa: int = 5
    provider: str = "gemini"  # "gemini" or "openai"

class RAGPipeline:
    """RAG Pipeline for document processing, summarization, and Q&A"""
    
    def __init__(self, config: RAGConfig = None, db_path: str = "vector_db"):
        """Initialize RAG Pipeline with configuration"""
        self.config = config or RAGConfig()
        self.embeddings = None
        self.llm = None
        self.vector_store = None
        self.documents = []
        self.source_metadata = {}
        
        # Initialize persistent vector database manager
        self.vector_db_manager = VectorDBManager(db_path)
        self.current_db_id = None
        
        # Detect provider from model name if not explicitly set
        if self.config.model_name.startswith("gemini"):
            self.config.provider = "gemini"
        elif self.config.model_name.startswith("gpt"):
            self.config.provider = "openai"
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LLM and embedding models based on provider"""
        try:
            if self.config.provider == "gemini":
                self._initialize_gemini()
            elif self.config.provider == "openai":
                self._initialize_openai()
            else:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
            
            logger.info(f"RAG components initialized successfully with {self.config.provider}")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {str(e)}")
            st.error(f"Error initializing AI models: {str(e)}")
            raise
    
    def _initialize_gemini(self):
        """Initialize Google Gemini components"""
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini dependencies not installed. Run: pip install langchain-google-genai google-generativeai")
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key or google_api_key == "your_google_api_key_here":
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file")
        
        # Initialize Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.embedding_model,
            google_api_key=google_api_key
        )
        
        # Initialize Gemini chat model
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            google_api_key=google_api_key
        )
    
    def _initialize_openai(self):
        """Initialize OpenAI components"""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI dependencies not installed. Run: pip install langchain-openai openai")
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key == "your_openai_api_key_here":
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=openai_api_key
        )
        
        # Initialize ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            openai_api_key=openai_api_key
        )
    
    def chunk_text(self, text: str, source_type: str = "document", metadata: dict = None) -> List[Document]:
        """
        Split text into chunks for processing
        
        Args:
            text: Input text to chunk
            source_type: Type of source ('pdf', 'youtube')
            metadata: Additional metadata for the document
            
        Returns:
            List of Document objects with chunks
        """
        try:
            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            
            # Create Document objects with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "source_type": source_type,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    **(metadata or {})
                }
                
                doc = Document(
                    page_content=chunk,
                    metadata=doc_metadata
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} chunks from {source_type} content")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            st.error(f"Error processing text: {str(e)}")
            return []
    
    def create_vector_store(self, documents: List[Document], content_info: Dict = None) -> bool:
        """
        Create or update vector store with documents and save persistently
        
        Args:
            documents: List of Document objects to add
            content_info: Metadata about the content source
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                st.error("No documents to process")
                return False

            # Show progress
            progress_bar = st.progress(0)
            st.info("Creating embeddings...")
            
            # Create vector store
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    documents, 
                    self.embeddings
                )
                progress_bar.progress(0.3)
            else:
                # Add to existing vector store
                self.vector_store.add_documents(documents)
                progress_bar.progress(0.3)
            
            # Store documents for later reference
            self.documents.extend(documents)
            progress_bar.progress(0.6)
            
            # Save to persistent storage
            st.info("Saving to persistent database...")
            if content_info:
                try:
                    self.current_db_id = self.vector_db_manager.save_vector_db(
                        self.vector_store,
                        self.documents,
                        content_info
                    )
                    st.success(f"✅ Vector database saved with ID: {self.current_db_id[:8]}...")
                    logger.info(f"Vector database saved with ID: {self.current_db_id}")
                except Exception as save_error:
                    # Continue without persistence if save fails
                    logger.error(f"Failed to save vector DB persistently: {save_error}")
                    st.warning("⚠️ Database created in memory only (not saved to disk)")
            
            progress_bar.progress(1.0)
            st.success(f"Successfully processed {len(documents)} chunks!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            st.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def load_vector_store(self, db_id: str) -> bool:
        """
        Load an existing vector database
        
        Args:
            db_id: Database ID to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            st.info(f"Loading database: {db_id[:8]}...")
            
            # Load from persistent storage
            db_data = self.vector_db_manager.load_vector_db(db_id, self.embeddings)
            
            if db_data:
                self.vector_store = db_data["vector_store"]
                self.documents = db_data["documents"]
                self.source_metadata = db_data["content_info"]
                self.current_db_id = db_id
                
                st.success(f"✅ Loaded database with {len(self.documents)} documents")
                logger.info(f"Loaded vector database: {db_id}")
                return True
            else:
                st.error("Failed to load database")
                return False
                
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            st.error(f"Error loading database: {str(e)}")
            return False
    
    def list_saved_databases(self) -> List[Dict]:
        """Get list of saved databases"""
        return self.vector_db_manager.list_databases()
    
    def delete_database(self, db_id: str) -> bool:
        """Delete a saved database"""
        return self.vector_db_manager.delete_database(db_id)
    
    def get_database_info(self, db_id: str) -> Optional[Dict]:
        """Get information about a specific database"""
        return self.vector_db_manager.get_database_info(db_id)
    
    def get_storage_stats(self) -> Dict:
        """Get storage statistics"""
        return self.vector_db_manager.get_storage_stats()
    
    def get_relevant_chunks(self, query: str, k: int = 5) -> List[Document]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant Document objects
        """
        try:
            if self.vector_store is None:
                st.error("No content has been processed yet. Please upload a document first.")
                return []
            
            # Perform similarity search
            relevant_docs = self.vector_store.similarity_search(
                query, 
                k=k
            )
            
            logger.info(f"Retrieved {len(relevant_docs)} relevant chunks for query")
            return relevant_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            st.error(f"Error searching content: {str(e)}")
            return []
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text based on provider"""
        if self.config.provider == "gemini":
            # Rough estimation for Gemini (no official tokenizer available)
            # Gemini typically has ~4 characters per token
            return len(text) // 4
        else:
            # Use tiktoken for OpenAI models
            try:
                encoding = tiktoken.encoding_for_model(self.config.model_name)
                return len(encoding.encode(text))
            except:
                # Fallback estimation
                return len(text) // 4
    
    def generate_summary(self, summary_type: str = "short") -> Dict[str, str]:
        """
        Generate summary of all processed content
        
        Args:
            summary_type: Type of summary ('short', 'detailed', 'both')
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            if not self.documents:
                return {"error": "No content to summarize"}

            # Get representative chunks for summary
            all_text = " ".join([doc.page_content for doc in self.documents[:self.config.max_chunks_for_summary]])
            
            # Ensure we don't exceed token limits
            tokens_count = self._count_tokens(all_text)
            max_tokens_for_input = self.config.max_tokens - 1000  # Leave room for prompt and response
            
            if tokens_count > max_tokens_for_input:
                # Truncate text based on token estimation
                if self.config.provider == "gemini":
                    # For Gemini, truncate by character count
                    target_chars = max_tokens_for_input * 4
                    all_text = all_text[:target_chars]
                else:
                    # For OpenAI, use tiktoken
                    encoding = tiktoken.encoding_for_model(self.config.model_name)
                    tokens = encoding.encode(all_text)
                    truncated_tokens = tokens[:max_tokens_for_input]
                    all_text = encoding.decode(truncated_tokens)
            
            summaries = {}
            
            if summary_type in ["short", "both"]:
                short_summary = self._generate_short_summary(all_text)
                summaries["short"] = short_summary
            
            if summary_type in ["detailed", "both"]:
                detailed_summary = self._generate_detailed_summary(all_text)
                summaries["detailed"] = detailed_summary
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {"error": f"Error generating summary: {str(e)}"}
    
    def _generate_short_summary(self, text: str) -> str:
        """Generate short bullet-point summary"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Please provide a concise summary of the following content in exactly 5 bullet points.
            Each bullet point should capture a key insight or main point.
            Make each point clear and actionable.
            
            Content:
            {text}
            
            Summary (5 bullet points):
            """
        )
        
        # Use the modern approach: prompt | llm
        chain = prompt | self.llm
        result = chain.invoke({"text": text})
        return result.content.strip()
    
    def _generate_detailed_summary(self, text: str) -> str:
        """Generate detailed paragraph summary"""
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Please provide a comprehensive summary of the following content in 1-2 well-structured paragraphs.
            Include the main themes, key insights, important details, and conclusions.
            Make it informative and easy to understand.
            
            Content:
            {text}
            
            Detailed Summary:
            """
        )
        
        # Use the modern approach: prompt | llm
        chain = prompt | self.llm
        result = chain.invoke({"text": text})
        return result.content.strip()
    
    def answer_question(self, question: str) -> Dict[str, any]:
        """
        Answer question using RAG approach
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source information
        """
        try:
            if self.vector_store is None:
                return {
                    "answer": "No content has been processed yet. Please upload a document first.",
                    "sources": [],
                    "error": True
                }
            
            # Get relevant chunks
            relevant_docs = self.get_relevant_chunks(question, k=self.config.max_chunks_for_qa)
            
            if not relevant_docs:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "error": True
                }
            
            # Prepare context
            context = "\n\n".join([
                f"Source {i+1}:\n{doc.page_content}" 
                for i, doc in enumerate(relevant_docs)
            ])
            
            # Generate answer
            answer = self._generate_answer(question, context)
            
            # Prepare source information
            sources = []
            for i, doc in enumerate(relevant_docs):
                sources.append({
                    "chunk_id": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                    "source_type": doc.metadata.get("source_type", "unknown")
                })
            
            return {
                "answer": answer,
                "sources": sources,
                "error": False
            }
            
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "answer": f"Error processing your question: {str(e)}",
                "sources": [],
                "error": True
            }
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on context"""
        prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            Based on the following context, please provide a comprehensive and accurate answer to the question.
            If the context doesn't contain enough information to answer the question completely, 
            please say so and provide what information you can find.
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
        )
        
        # Use the modern approach: prompt | llm
        chain = prompt | self.llm
        result = chain.invoke({"question": question, "context": context})
        return result.content.strip()
    
    def clear_content(self):
        """Clear all processed content and reset the pipeline"""
        self.vector_store = None
        self.documents = []
        self.source_metadata = {}
        logger.info("RAG pipeline content cleared")
    
    def get_content_stats(self) -> Dict[str, any]:
        """Get statistics about processed content"""
        if not self.documents:
            return {"total_chunks": 0, "total_content": 0}
        
        total_content_length = sum(len(doc.page_content) for doc in self.documents)
        source_types = {}
        
        for doc in self.documents:
            source_type = doc.metadata.get("source_type", "unknown")
            source_types[source_type] = source_types.get(source_type, 0) + 1
        
        return {
            "total_chunks": len(self.documents),
            "total_content_length": total_content_length,
            "source_types": source_types,
            "avg_chunk_size": total_content_length // len(self.documents) if self.documents else 0
        }
