"""
AI Multi-Source Summarizer - Streamlit Application

A comprehensive tool for summarizing content from PDFs and YouTube videos
using RAG (Retrieval-Augmented Generation) with LangChain and OpenAI.
"""

import os
import logging
from dotenv import load_dotenv
import streamlit as st
from typing import Optional, Dict, Any

# Import local modules
from backend.loaders.pdf_loader import PDFLoader
from backend.loaders.enhanced_youtube_loader import EnhancedYouTubeLoader
from backend.core.rag_pipeline import RAGPipeline, RAGConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(
    page_title="AI Multi-Source Summarizer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .summary-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
        color: #2c3e50;
        font-size: 1rem;
        line-height: 1.6;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #ffc107;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #856404;
    }
    .qa-container {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'content_processed' not in st.session_state:
        st.session_state.content_processed = False
    if 'current_content' not in st.session_state:
        st.session_state.current_content = None
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

def validate_api_key() -> bool:
    """Validate API key based on configured provider"""
    # Determine provider from model name
    model_name = os.getenv("DEFAULT_MODEL", "gemini-1.5-flash")
    
    if model_name.startswith("gemini"):
        # Validate Google API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_google_api_key_here":
            st.error("❌ Google API key not found! Please set your API key in the .env file.")
            st.info("💡 Get your API key at: https://makersuite.google.com/app/apikey")
            st.info("💡 Then set GOOGLE_API_KEY in your .env file")
            return False
        st.success("✅ Google Gemini API key configured")
        return True
    else:
        # Validate OpenAI API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            st.error("❌ OpenAI API key not found! Please set your API key in the .env file.")
            st.info("💡 Copy .env.example to .env and add your OpenAI API key")
            return False
        st.success("✅ OpenAI API key configured")
        return True

def initialize_rag_pipeline() -> Optional[RAGPipeline]:
    """Initialize RAG pipeline with configuration"""
    try:
        if st.session_state.rag_pipeline is None:
            # Create configuration from environment variables
            config = RAGConfig(
                chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
                chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
                model_name=os.getenv("DEFAULT_MODEL", "gemini-1.5-flash"),
                embedding_model=os.getenv("EMBEDDING_MODEL", "models/embedding-001")
            )
            
            st.session_state.rag_pipeline = RAGPipeline(config)
            logger.info("RAG pipeline initialized successfully")
        
        return st.session_state.rag_pipeline
    
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {str(e)}")
        st.error(f"Error initializing AI components: {str(e)}")
        return None

def render_sidebar():
    """Render sidebar with content input options"""
    st.sidebar.markdown('<div class="section-header">📚 Content Input</div>', unsafe_allow_html=True)
    
    # Content source selection
    source_type = st.sidebar.radio(
        "Choose content source:",
        ["📄 Upload PDF", "🎥 YouTube Video"],
        key="source_type"
    )
    
    if source_type == "📄 Upload PDF":
        render_pdf_upload()
    else:
        render_youtube_input()
    
    # Content management
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="section-header">⚙️ Actions</div>', unsafe_allow_html=True)
    
    if st.sidebar.button("🗑️ Clear All Content", type="secondary"):
        clear_all_content()
    
    # Display content stats
    if st.session_state.content_processed and st.session_state.rag_pipeline:
        stats = st.session_state.rag_pipeline.get_content_stats()
        st.sidebar.markdown("---")
        st.sidebar.markdown("📊 **Content Statistics**")
        st.sidebar.write(f"• Chunks: {stats['total_chunks']}")
        st.sidebar.write(f"• Content Length: {stats['total_content_length']:,} chars")
        st.sidebar.write(f"• Avg Chunk Size: {stats['avg_chunk_size']} chars")
    
    # Database management section
    render_database_management()

def render_database_management():
    """Render database management interface"""
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="section-header">💾 Saved Databases</div>', unsafe_allow_html=True)
    
    # Initialize RAG pipeline to access database manager
    rag_pipeline = initialize_rag_pipeline()
    if not rag_pipeline:
        st.sidebar.error("RAG pipeline not initialized")
        return
    
    # Get list of saved databases
    try:
        databases = rag_pipeline.list_saved_databases()
        
        if not databases:
            st.sidebar.info("No saved databases found")
            return
        
        # Database selection
        db_options = []
        for db in databases:
            name = db["name"][:30] + "..." if len(db["name"]) > 30 else db["name"]
            date = db["created_at"][:10] if db["created_at"] != "unknown" else "Unknown"
            db_options.append(f"{name} ({date})")
        
        selected_idx = st.sidebar.selectbox(
            "Select database to load:",
            range(len(db_options)),
            format_func=lambda x: db_options[x],
            key="db_selector"
        )
        
        if selected_idx is not None:
            selected_db = databases[selected_idx]
            
            # Database info
            st.sidebar.markdown("**Database Info:**")
            st.sidebar.write(f"• Type: {selected_db['source_type']}")
            st.sidebar.write(f"• Documents: {selected_db['document_count']}")
            st.sidebar.write(f"• Size: {selected_db['content_length']:,} chars")
            
            col1, col2 = st.sidebar.columns(2)
            
            # Load button
            if col1.button("📁 Load", key="load_db"):
                if rag_pipeline.load_vector_store(selected_db["db_id"]):
                    st.session_state.content_processed = True
                    st.session_state.current_content = {
                        "type": "loaded_db",
                        "name": selected_db["name"],
                        "metadata": selected_db
                    }
                    st.sidebar.success("✅ Database loaded!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to load database")
            
            # Delete button
            if col2.button("🗑️ Delete", key="delete_db"):
                if rag_pipeline.delete_database(selected_db["db_id"]):
                    st.sidebar.success("✅ Database deleted!")
                    st.rerun()
                else:
                    st.sidebar.error("❌ Failed to delete database")
        
        # Storage stats
        stats = rag_pipeline.get_storage_stats()
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Storage Stats:**")
        st.sidebar.write(f"• Total DBs: {stats['database_count']}")
        st.sidebar.write(f"• Total Docs: {stats['total_documents']}")
        st.sidebar.write(f"• Storage: {stats['total_size_mb']} MB")
        
    except Exception as e:
        st.sidebar.error(f"Error accessing databases: {str(e)}")

def render_pdf_upload():
    """Render PDF upload interface"""
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF file to summarize and analyze"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("📝 Process PDF", type="primary"):
            process_pdf_file(uploaded_file)

def render_youtube_input():
    """Render YouTube URL input interface"""
    youtube_url = st.sidebar.text_input(
        "Enter YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Enter a YouTube URL to fetch transcript and analyze"
    )
    
    # Add a text area for manual transcript input as fallback
    st.sidebar.markdown("**Or paste transcript manually:**")
    manual_transcript = st.sidebar.text_area(
        "Manual Transcript",
        placeholder="Paste video transcript here if automatic fetching fails...",
        height=100,
        help="If YouTube transcript fetching fails, you can paste the transcript manually"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if youtube_url and st.button("🎥 Process Video", type="primary"):
            process_youtube_url(youtube_url)
    
    with col2:
        if manual_transcript and st.button("📝 Process Text", type="secondary"):
            process_manual_transcript(manual_transcript, youtube_url or "Manual Input")

def process_pdf_file(pdf_file):
    """Process uploaded PDF file"""
    try:
        with st.spinner("Processing PDF..."):
            # Validate PDF
            if not PDFLoader.validate_pdf_file(pdf_file):
                return
            
            # Extract text
            text_content = PDFLoader.extract_text_from_pdf(pdf_file)
            if not text_content:
                return
            
            # Get metadata
            metadata = PDFLoader.get_pdf_metadata(pdf_file)
            
            # Initialize RAG pipeline
            rag_pipeline = initialize_rag_pipeline()
            if not rag_pipeline:
                return
            
            # Clear previous content
            rag_pipeline.clear_content()
            
            # Process content
            documents = rag_pipeline.chunk_text(
                text_content,
                source_type="pdf",
                metadata={
                    "filename": metadata["file_name"],
                    "num_pages": metadata["num_pages"]
                }
            )
            
            # Prepare content info for persistent storage
            content_info = {
                "source_type": "pdf",
                "identifier": metadata["file_name"],
                "name": metadata["file_name"],
                "content_length": len(text_content),
                "num_pages": metadata["num_pages"],
                "file_size": metadata.get("file_size", 0)
            }
            
            if rag_pipeline.create_vector_store(documents, content_info):
                st.session_state.content_processed = True
                st.session_state.current_content = {
                    "type": "pdf",
                    "name": metadata["file_name"],
                    "metadata": metadata
                }
                st.success(f"✅ PDF processed successfully! ({len(documents)} chunks created)")
                st.rerun()
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        st.error(f"Error processing PDF: {str(e)}")

def process_manual_transcript(transcript_text: str, source_name: str):
    """Process manually entered transcript text"""
    try:
        with st.spinner("Processing manual transcript..."):
            if len(transcript_text.strip()) < 100:
                st.warning("⚠️ Transcript seems too short. Please ensure you have pasted the full transcript.")
                return
            
            # Initialize RAG pipeline
            rag_pipeline = initialize_rag_pipeline()
            if not rag_pipeline:
                return
            
            # Clear previous content
            rag_pipeline.clear_content()
            
            # Process content
            documents = rag_pipeline.chunk_text(
                transcript_text,
                source_type="manual",
                metadata={
                    "source_name": source_name,
                    "input_method": "manual",
                    "content_length": len(transcript_text)
                }
            )
            
            # Prepare content info for persistent storage
            content_info = {
                "source_type": "manual",
                "identifier": f"manual_{source_name}_{len(transcript_text)}",
                "name": f"Manual: {source_name}",
                "content_length": len(transcript_text),
                "source_name": source_name,
                "input_method": "manual"
            }
            
            if rag_pipeline.create_vector_store(documents, content_info):
                st.session_state.content_processed = True
                st.session_state.current_content = {
                    "type": "manual",
                    "name": f"Manual: {source_name}",
                    "metadata": {
                        "source_name": source_name,
                        "content_length": len(transcript_text)
                    }
                }
                st.success(f"✅ Manual transcript processed successfully! ({len(documents)} chunks created)")
                st.rerun()
    
    except Exception as e:
        logger.error(f"Error processing manual transcript: {str(e)}")
        st.error(f"Error processing manual transcript: {str(e)}")

def process_youtube_url(youtube_url):
    """Process YouTube URL"""
    try:
        with st.spinner("Fetching YouTube transcript..."):
            # Validate URL
            if not EnhancedYouTubeLoader.validate_youtube_url(youtube_url):
                # Show helpful suggestions when validation fails
                st.markdown("### 💡 **Alternative Options:**")
                st.markdown("1. **Try these reliable video types:**")
                st.markdown("   - TED Talks: https://www.youtube.com/watch?v=UF8uR6Z6KLc")
                st.markdown("   - Educational content: Khan Academy, Crash Course")
                st.markdown("   - News videos from major channels")
                st.markdown("2. **Use Manual Transcript Input** (see sidebar)")
                st.markdown("3. **Upload a PDF instead** for reliable processing")
                return
            
            # Process URL
            result = EnhancedYouTubeLoader.process_youtube_url(youtube_url)
            if not result:
                st.error("❌ Failed to fetch transcript from YouTube")
                st.markdown("### 🔧 **Troubleshooting:**")
                st.markdown("- Make sure the video is **public** and has **captions**")
                st.markdown("- Try the **Manual Transcript** option in the sidebar")
                st.markdown("- Consider uploading a **PDF document** instead")
                return
            
            # Initialize RAG pipeline
            rag_pipeline = initialize_rag_pipeline()
            if not rag_pipeline:
                return
            
            # Clear previous content
            rag_pipeline.clear_content()
            
            # Process content
            documents = rag_pipeline.chunk_text(
                result["content"],
                source_type="youtube",
                metadata={
                    "video_id": result["video_id"],
                    "video_url": result["video_url"],
                    "content_length": result["content_length"]
                }
            )
            
            # Prepare content info for persistent storage
            content_info = {
                "source_type": "youtube",
                "identifier": result["video_id"],
                "name": f"YouTube Video: {result['video_id']}",
                "content_length": result["content_length"],
                "video_url": result["video_url"],
                "extraction_method": result.get("method", "unknown"),
                "video_id": result["video_id"]
            }
            
            if rag_pipeline.create_vector_store(documents, content_info):
                st.session_state.content_processed = True
                st.session_state.current_content = {
                    "type": "youtube",
                    "name": f"Video: {result['video_id']}",
                    "metadata": result
                }
                st.success(f"✅ YouTube video processed successfully! ({len(documents)} chunks created)")
                st.rerun()
    
    except Exception as e:
        logger.error(f"Error processing YouTube URL: {str(e)}")
        st.error(f"❌ Error processing YouTube URL: {str(e)}")
        st.markdown("### 🔧 **What you can try:**")
        st.markdown("1. **Use the Manual Transcript option** in the sidebar")
        st.markdown("2. **Try a different video** with clear captions")
        st.markdown("3. **Upload a PDF document** instead")
        st.markdown("4. **Check the video is public and not restricted**")

def clear_all_content():
    """Clear all processed content"""
    if st.session_state.rag_pipeline:
        st.session_state.rag_pipeline.clear_content()
    
    st.session_state.content_processed = False
    st.session_state.current_content = None
    st.session_state.summaries = {}
    st.session_state.qa_history = []
    
    st.success("🗑️ All content cleared!")
    st.rerun()

def render_summaries():
    """Render summary generation and display"""
    st.markdown('<div class="section-header">📋 Content Summaries</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📝 Generate Short Summary", type="secondary"):
            generate_summary("short")
    
    with col2:
        if st.button("📄 Generate Detailed Summary", type="secondary"):
            generate_summary("detailed")
    
    # Display summaries
    if "short" in st.session_state.summaries:
        st.markdown("### 🔸 Short Summary (Key Points)")
        st.markdown(f'<div class="summary-box">{st.session_state.summaries["short"]}</div>', 
                   unsafe_allow_html=True)
    
    if "detailed" in st.session_state.summaries:
        st.markdown("### 📄 Detailed Summary")
        st.markdown(f'<div class="summary-box">{st.session_state.summaries["detailed"]}</div>', 
                   unsafe_allow_html=True)

def generate_summary(summary_type: str):
    """Generate summary of specified type"""
    try:
        with st.spinner(f"Generating {summary_type} summary..."):
            rag_pipeline = st.session_state.rag_pipeline
            summaries = rag_pipeline.generate_summary(summary_type)
            
            if "error" in summaries:
                st.error(summaries["error"])
            else:
                st.session_state.summaries.update(summaries)
                st.success(f"✅ {summary_type.title()} summary generated!")
                st.rerun()
    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        st.error(f"Error generating summary: {str(e)}")

def render_qa_interface():
    """Render Q&A interface"""
    st.markdown('<div class="section-header">❓ Interactive Q&A</div>', unsafe_allow_html=True)
    
    # Question input
    question = st.text_input(
        "Ask a question about the content:",
        placeholder="What are the main points discussed?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔍 Ask Question", type="primary"):
            if question.strip():
                answer_question(question)
            else:
                st.warning("Please enter a question!")
    
    with col2:
        if st.button("🗑️ Clear Q&A History", type="secondary"):
            st.session_state.qa_history = []
            st.rerun()
    
    # Display Q&A history
    if st.session_state.qa_history:
        st.markdown("### 💬 Q&A History")
        for i, qa in enumerate(reversed(st.session_state.qa_history)):
            render_qa_item(qa, len(st.session_state.qa_history) - i)

def answer_question(question: str):
    """Answer user question using RAG"""
    try:
        with st.spinner("Searching for answer..."):
            rag_pipeline = st.session_state.rag_pipeline
            result = rag_pipeline.answer_question(question)
            
            st.session_state.qa_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "error": result.get("error", False)
            })
            
            st.rerun()
    
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        st.error(f"Error answering question: {str(e)}")

def render_qa_item(qa: Dict[str, Any], index: int):
    """Render individual Q&A item"""
    st.markdown(f'<div class="qa-container">', unsafe_allow_html=True)
    st.markdown(f"**Q{index}: {qa['question']}**")
    
    if qa.get("error", False):
        st.error(qa["answer"])
    else:
        st.markdown(f"**A:** {qa['answer']}")
        
        # Display sources
        if qa["sources"]:
            st.markdown("**📚 Sources:**")
            for i, source in enumerate(qa["sources"]):
                source_type = source.get("source_type", "unknown")
                source_icon = "📄" if source_type == "pdf" else "🎥"
                st.markdown(
                    f'<div class="source-box">'
                    f'{source_icon} <strong>Source {i+1}:</strong> {source["content"]}'
                    f'</div>', 
                    unsafe_allow_html=True
                )
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")

def render_main_content():
    """Render main content area"""
    # Header
    st.markdown('<h1 class="main-header">🤖 AI Multi-Source Summarizer</h1>', unsafe_allow_html=True)
    
    if not st.session_state.content_processed:
        # Welcome screen
        st.markdown("""
        ### Welcome! 👋
        
        This tool helps you summarize and analyze content from multiple sources:
        
        - **📄 PDF Documents**: Upload and analyze text from PDF files
        - **🎥 YouTube Videos**: Extract and summarize video content with multiple methods
        
        **Features:**
        - 📋 Generate short bullet-point summaries
        - 📄 Create detailed paragraph summaries  
        - ❓ Ask questions about your content with citations
        - 🔍 Powered by Google Gemini AI and vector search
        
        **Enhanced YouTube Processing:**
        - 📝 YouTube Transcript API (fastest)
        - 🔧 yt-dlp subtitle extraction (backup)
        - 📖 Video description analysis (fallback)
        - 🎧 Audio transcription with Whisper (comprehensive)
        - 🎬 Visual content analysis (NEW - for videos without transcript/audio)
        
        **Get Started:**
        1. Make sure you have a Google API key set in the .env file
        2. Choose a content source from the sidebar
        3. Upload a PDF or enter a YouTube URL
        4. Generate summaries and ask questions!
        
        **Switching AI Providers:**
        - Currently using: **Google Gemini** 🤖
        - To use OpenAI instead, change DEFAULT_MODEL in .env to "gpt-3.5-turbo"
        """)
        
        # API key check
        if not validate_api_key():
            st.stop()
    
    else:
        # Content processed - show current content info
        content = st.session_state.current_content
        if content:
            st.info(f"📊 Currently analyzing: **{content['name']}** ({content['type'].upper()})")
        
        # Render main features
        render_summaries()
        st.markdown("---")
        render_qa_interface()

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Render sidebar
        render_sidebar()
        
        # Render main content
        render_main_content()
        
    except Exception as e:
        logger.error(f"Unexpected error in main app: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page and try again.")
        st.exception(e)

if __name__ == "__main__":
    main()
