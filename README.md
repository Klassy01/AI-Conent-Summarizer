# 🤖 AI Multi-Source Summarizer

A comprehensive Python application for summarizing and analyzing content from multiple sources using advanced AI technologies. Built with Streamlit, LangChain, FAISS vector database, and Google Gemini AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.0.350+-green.svg)

## 🌟 Features

### 📄 Multi-Source Content Processing
- **PDF Documents**: Extract and analyze text from PDF files
- **YouTube Videos**: Advanced transcript extraction with multiple fallback methods
- **Manual Input**: Direct text input for custom content

### 🔍 Advanced YouTube Processing (5-Method Fallback)
1. **YouTube Transcript API** - Fast official transcript extraction
2. **yt-dlp Subtitle Extraction** - Backup subtitle extraction
3. **Video Description Analysis** - Content analysis from video metadata
4. **Audio Transcription** - Whisper AI for audio-to-text conversion
5. **Visual Analysis** - Computer vision for video content understanding (NEW!)

### 🧠 AI-Powered Analysis
- **Smart Summarization**: Generate both short bullet-point and detailed summaries
- **Interactive Q&A**: Ask questions with source citations
- **Persistent Storage**: Save and manage multiple document databases
- **Vector Search**: Semantic search using FAISS for accurate retrieval

### 💾 Database Management
- **Persistent Vector Database**: Save processed content for future use
- **Database Manager**: Load, view, and delete saved databases
- **Storage Statistics**: Monitor storage usage and document counts
- **Metadata Tracking**: Track source type, creation date, and content statistics

## 🏗️ Architecture

```
Content Summarizer/
├── app.py                      # Main Streamlit application
├── backend/                    # Core backend modules
│   ├── core/
│   │   └── rag_pipeline.py     # RAG pipeline with LangChain
│   ├── loaders/
│   │   ├── pdf_loader.py       # PDF content extraction
│   │   ├── youtube_loader.py   # Basic YouTube loader
│   │   └── enhanced_youtube_loader.py  # Advanced YouTube processing
│   ├── analyzers/
│   │   └── video_analyzer.py   # Video visual analysis
│   └── database/
│       └── vector_db_manager.py # Vector database management
├── vector_db/                  # Persistent vector storage
├── test_*.py                   # Test scripts
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- Git (for cloning)
- Google API key (for Gemini AI)

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Content-Summarizer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

1. Copy the environment template:
```bash
copy .env.example .env
```

2. Edit `.env` file and add your Google API key:
```env
GOOGLE_API_KEY=your_google_api_key_here
DEFAULT_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/embedding-001
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

3. Get your Google API key at: https://makersuite.google.com/app/apikey

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 Usage Guide

### Processing PDF Documents
1. Select "📄 Upload PDF" in the sidebar
2. Upload your PDF file
3. Click "📝 Process PDF"
4. Generate summaries and ask questions

### Processing YouTube Videos
1. Select "🎥 YouTube Video" in the sidebar
2. Enter a YouTube URL
3. Click "🎥 Process Video"
4. The system will try multiple methods to extract content

### Generating Summaries
- **Short Summary**: Key points in bullet format
- **Detailed Summary**: Comprehensive paragraph summary

### Interactive Q&A
- Ask specific questions about your content
- Get answers with source citations
- View Q&A history

### Database Management
- Save processed content for future use
- Load previously saved databases
- View storage statistics
- Delete unwanted databases

## 🛠️ Technologies Used

### Core Framework
- **Streamlit**: Web application framework
- **LangChain**: LLM integration and RAG pipeline
- **FAISS**: Vector similarity search
- **Google Gemini**: Large language model

### Content Processing
- **PyPDF2**: PDF text extraction
- **youtube-transcript-api**: YouTube transcript fetching
- **yt-dlp**: Video downloading and subtitle extraction
- **OpenAI Whisper**: Audio transcription
- **OpenCV & BLIP**: Computer vision and visual analysis

### Data Management
- **SQLite**: Metadata storage
- **Pickle**: Vector database serialization
- **JSON**: Configuration and metadata

## 🔧 Configuration Options

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `DEFAULT_MODEL` | LLM model to use | `gemini-1.5-flash` |
| `EMBEDDING_MODEL` | Embedding model | `models/embedding-001` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |

### Supported Models
- **Google Gemini**: `gemini-1.5-flash`, `gemini-1.5-pro`
- **OpenAI**: `gpt-3.5-turbo`, `gpt-4` (requires OpenAI API key)

## 🧪 Testing

Run the test scripts to verify functionality:

```bash
# Test basic YouTube loading
python test_youtube.py

# Test enhanced YouTube processing
python test_enhanced_youtube.py

# Test vector database
python test_vector_db.py

# Test video analysis
python test_video_analysis.py
```

## 📁 File Structure

### Core Files
- `app.py`: Main Streamlit application with UI
- `requirements.txt`: Python package dependencies
- `.env`: Environment configuration (create from .env.example)

### Backend Modules
- `backend/core/rag_pipeline.py`: RAG implementation with LangChain
- `backend/loaders/`: Content loading modules
- `backend/analyzers/`: Content analysis modules  
- `backend/database/`: Vector database management

### Storage
- `vector_db/`: Persistent vector databases and metadata
- `venv/`: Python virtual environment (created during setup)

## 🚨 Troubleshooting

### Common Issues

1. **YouTube video not processing**:
   - Ensure video is public and has captions
   - Try the manual transcript option
   - Check video URL format

2. **API key errors**:
   - Verify Google API key in `.env` file
   - Check API key permissions and quotas

3. **Module import errors**:
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

4. **Video analysis not working**:
   - Install torch and torchvision
   - Ensure sufficient system memory
   - Try smaller video files

### Performance Tips

- Use shorter documents for faster processing
- Enable visual analysis only for videos without transcripts
- Regularly clean up old vector databases
- Monitor storage usage in the sidebar

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 2+ cores
- **RAM**: 4GB (8GB recommended for video analysis)
- **Storage**: 2GB free space
- **Python**: 3.8+
- **Internet**: Required for AI API calls

### Recommended Setup
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **GPU**: Optional (for faster video processing)
- **Storage**: 5GB+ for multiple databases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini API for powerful language processing
- LangChain for RAG framework
- Streamlit for the web interface
- OpenAI Whisper for audio transcription
- FAISS for efficient vector search

## 📞 Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the test scripts for examples
3. Create an issue in the repository
4. Check API key configuration and quotas

---

**Built with ❤️ for intelligent content analysis**
