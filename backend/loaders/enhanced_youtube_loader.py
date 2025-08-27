"""
Enhanced YouTube content extraction with multiple fallback methods
"""

import streamlit as st
import os
import tempfile
import subprocess
from typing import Optional, Dict, List
import logging
import json

# Try different extraction methods
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

from .youtube_loader import YouTubeLoader as OriginalYouTubeLoader
from ..analyzers.video_analyzer import VideoAnalyzer, is_video_analysis_available

logger = logging.getLogger(__name__)

class EnhancedYouTubeLoader:
    """Enhanced YouTube content extraction with multiple methods"""
    
    @staticmethod
    def extract_video_id(youtube_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL - reuse original method"""
        return OriginalYouTubeLoader.extract_video_id(youtube_url)
    
    @staticmethod
    def validate_youtube_url(youtube_url: str) -> bool:
        """Validate YouTube URL - reuse original method"""
        return OriginalYouTubeLoader.validate_youtube_url(youtube_url)
    
    @staticmethod
    def get_video_info_with_ytdlp(video_id: str) -> Optional[Dict]:
        """Get video information using yt-dlp"""
        if not YT_DLP_AVAILABLE:
            return None
            
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Check for subtitles/captions
                subtitles = info.get('subtitles', {})
                automatic_captions = info.get('automatic_captions', {})
                
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'description': info.get('description', '')[:500] + '...' if info.get('description') else '',
                    'uploader': info.get('uploader', 'Unknown'),
                    'subtitles': list(subtitles.keys()),
                    'automatic_captions': list(automatic_captions.keys()),
                    'has_subtitles': bool(subtitles or automatic_captions)
                }
        except Exception as e:
            logger.error("Error extracting video info with yt-dlp: %s", str(e))
            return None
    
    @staticmethod
    def extract_subtitles_with_ytdlp(video_id: str) -> Optional[str]:
        """Extract subtitles using yt-dlp"""
        if not YT_DLP_AVAILABLE:
            return None
            
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitlesformat': 'vtt',
                    'subtitleslangs': ['en', 'en-US', 'en-GB'],
                    'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'),
                    'skip_download': True,
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    
                    # Find subtitle files
                    for file in os.listdir(temp_dir):
                        if file.endswith('.vtt'):
                            subtitle_path = os.path.join(temp_dir, file)
                            with open(subtitle_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                # Parse VTT content to extract text
                                text = EnhancedYouTubeLoader._parse_vtt_content(content)
                                if text:
                                    logger.info("Extracted subtitles using yt-dlp for video %s", video_id)
                                    return text
                
        except Exception as e:
            logger.error("Error extracting subtitles with yt-dlp: %s", str(e))
            
        return None
    
    @staticmethod
    def _parse_vtt_content(vtt_content: str) -> str:
        """Parse VTT subtitle content to extract clean text"""
        lines = vtt_content.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip VTT headers, timestamps, and empty lines
            if (line and 
                not line.startswith('WEBVTT') and
                not line.startswith('NOTE') and
                '-->' not in line and
                not line.isdigit()):
                # Remove HTML tags if any
                import re
                clean_line = re.sub(r'<[^>]+>', '', line)
                if clean_line:
                    text_lines.append(clean_line)
        
        return ' '.join(text_lines)
    
    @staticmethod
    def download_and_transcribe_audio(video_id: str) -> Optional[str]:
        """Download audio and transcribe using Whisper (fallback method)"""
        if not YT_DLP_AVAILABLE or not WHISPER_AVAILABLE:
            return None
            
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                audio_path = os.path.join(temp_dir, f"{video_id}.%(ext)s")
                
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'outtmpl': audio_path,
                    'quiet': True,
                    'no_warnings': True,
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                }
                
                # Download audio
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                # Find the downloaded audio file
                audio_file = None
                for file in os.listdir(temp_dir):
                    if file.endswith('.wav') and video_id in file:
                        audio_file = os.path.join(temp_dir, file)
                        break
                
                if audio_file and os.path.exists(audio_file):
                    # Load Whisper model (using base model for speed)
                    model = whisper.load_model("base")
                    
                    # Transcribe audio
                    result = model.transcribe(audio_file)
                    transcript_text = result["text"]
                    
                    logger.info("Successfully transcribed audio for video %s", video_id)
                    return transcript_text.strip()
                
        except Exception as e:
            logger.error("Error downloading and transcribing audio: %s", str(e))
            
        return None
    
    @staticmethod
    def get_video_content(video_id: str) -> Optional[Dict]:
        """Get video content using multiple methods (priority order)"""
        
        st.info("🔄 Trying multiple extraction methods...")
        
        # Method 1: Try original transcript API
        progress_bar = st.progress(0.20)
        st.text("Method 1/5: YouTube Transcript API...")
        
        try:
            transcript = OriginalYouTubeLoader.get_video_transcript(video_id)
            if transcript and len(transcript.strip()) > 100:
                progress_bar.progress(1.0)
                st.success("✅ Success with YouTube Transcript API")
                return {
                    "content": transcript,
                    "method": "transcript_api",
                    "video_id": video_id,
                    "content_length": len(transcript)
                }
        except:
            pass
        
        # Method 2: Try yt-dlp subtitles
        progress_bar.progress(0.40)
        st.text("Method 2/5: yt-dlp subtitle extraction...")
        
        if YT_DLP_AVAILABLE:
            transcript = EnhancedYouTubeLoader.extract_subtitles_with_ytdlp(video_id)
            if transcript and len(transcript.strip()) > 100:
                progress_bar.progress(1.0)
                st.success("✅ Success with yt-dlp subtitles")
                return {
                    "content": transcript,
                    "method": "ytdlp_subtitles",
                    "video_id": video_id,
                    "content_length": len(transcript)
                }
        
        # Method 3: Try video description as fallback
        progress_bar.progress(0.60)
        st.text("Method 3/5: Video description extraction...")
        
        if YT_DLP_AVAILABLE:
            video_info = EnhancedYouTubeLoader.get_video_info_with_ytdlp(video_id)
            if video_info and video_info.get('description'):
                description = video_info['description']
                if len(description.strip()) > 200:  # Reasonable length
                    progress_bar.progress(1.0)
                    st.warning("⚠️ Using video description (no transcript available)")
                    return {
                        "content": f"Video Title: {video_info.get('title', 'Unknown')}\n\nDescription:\n{description}",
                        "method": "description",
                        "video_id": video_id,
                        "content_length": len(description),
                        "title": video_info.get('title'),
                        "uploader": video_info.get('uploader')
                    }
        
        # Method 4: Audio transcription with Whisper
        progress_bar.progress(0.75)
        st.text("Method 4/5: Audio transcription...")
        
        if WHISPER_AVAILABLE:
            with st.spinner("🎧 Downloading and transcribing audio... This may take several minutes."):
                transcript = EnhancedYouTubeLoader.download_and_transcribe_audio(video_id)
                if transcript and len(transcript.strip()) > 100:
                    progress_bar.progress(1.0)
                    st.success("✅ Success with audio transcription")
                    return {
                        "content": transcript,
                        "method": "whisper_transcription",
                        "video_id": video_id,
                        "content_length": len(transcript)
                    }
        
        # Method 5: Visual Analysis (NEW - for videos without any transcript/audio)
        progress_bar.progress(0.90)
        st.text("Method 5/5: Visual content analysis...")
        
        # Check if video analysis is available
        capabilities = is_video_analysis_available()
        if capabilities["opencv"] and capabilities["yt_dlp"]:
            st.info("🎬 No transcript/audio found. Analyzing visual content...")
            
            try:
                video_analyzer = VideoAnalyzer()
                analysis_result = video_analyzer.analyze_video_content(video_id)
                
                if analysis_result and analysis_result.get("content_summary"):
                    progress_bar.progress(1.0)
                    st.success("✅ Success with visual content analysis")
                    return {
                        "content": analysis_result["content_summary"],
                        "method": "visual_analysis",
                        "video_id": video_id,
                        "content_length": len(analysis_result["content_summary"]),
                        "analysis_details": analysis_result,
                        "frame_count": analysis_result.get("key_frames_analyzed", 0)
                    }
            except Exception as e:
                logger.error("Error in visual analysis: %s", str(e))
                st.warning(f"⚠️ Visual analysis failed: {str(e)}")
        else:
            st.warning("⚠️ Visual analysis not available (missing dependencies)")
        
        progress_bar.progress(1.0)
        st.error("❌ All extraction methods failed")
        return None
    
    @staticmethod
    def process_youtube_url(youtube_url: str) -> Optional[Dict]:
        """Process YouTube URL with enhanced methods"""
        try:
            # Extract video ID
            video_id = EnhancedYouTubeLoader.extract_video_id(youtube_url)
            if not video_id:
                st.error("Could not extract video ID from URL.")
                return None
            
            st.info(f"🎥 Processing video: {video_id}")
            
            # Get video content using multiple methods
            result = EnhancedYouTubeLoader.get_video_content(video_id)
            
            if result:
                result.update({
                    "source_type": "youtube",
                    "video_url": f"https://www.youtube.com/watch?v={video_id}"
                })
                return result
            else:
                st.error("Failed to extract any content from the video.")
                return None
                
        except Exception as e:
            logger.error("Error processing YouTube URL: %s", str(e))
            st.error(f"Error processing YouTube URL: {str(e)}")
            return None
