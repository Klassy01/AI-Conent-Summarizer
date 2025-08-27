"""
YouTube transcript loading and processing utilities
"""

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class YouTubeLoader:
    """Handle YouTube URL processing and transcript extraction"""
    
    @staticmethod
    def extract_video_id(youtube_url: str) -> Optional[str]:
        """
        Extract video ID from various YouTube URL formats
        
        Args:
            youtube_url: YouTube URL in various formats
            
        Returns:
            Video ID string or None if not found
        """
        # Common YouTube URL patterns
        patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        # If no pattern matches, check if it's just a video ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', youtube_url.strip()):
            return youtube_url.strip()
            
        return None
    
    @staticmethod
    def get_video_transcript(video_id: str, languages: List[str] = None) -> Optional[str]:
        """
        Fetch transcript for a YouTube video with improved error handling
        
        Args:
            video_id: YouTube video ID
            languages: List of preferred languages for transcript
            
        Returns:
            Transcript text or None if not available
        """
        if languages is None:
            languages = ['en', 'en-US', 'en-GB']
        
        try:
            # First, try to get the transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Strategy 1: Try manual transcripts first
            for language in languages:
                try:
                    transcript = transcript_list.find_manually_created_transcript([language])
                    transcript_data = transcript.fetch()
                    
                    # Simple text extraction without formatter
                    text_parts = []
                    for entry in transcript_data:
                        if 'text' in entry and entry['text'].strip():
                            text_parts.append(entry['text'].strip())
                    
                    if text_parts:
                        text_content = ' '.join(text_parts)
                        logger.info("Found manual transcript in %s for video %s", language, video_id)
                        return text_content
                        
                except Exception:
                    continue
            
            # Strategy 2: Try auto-generated transcripts
            for language in languages:
                try:
                    transcript = transcript_list.find_generated_transcript([language])
                    transcript_data = transcript.fetch()
                    
                    # Simple text extraction
                    text_parts = []
                    for entry in transcript_data:
                        if 'text' in entry and entry['text'].strip():
                            text_parts.append(entry['text'].strip())
                    
                    if text_parts:
                        text_content = ' '.join(text_parts)
                        logger.info("Found auto-generated transcript in %s for video %s", language, video_id)
                        return text_content
                        
                except Exception:
                    continue
            
            # Strategy 3: Get any available transcript
            try:
                # Get all available transcripts
                all_transcripts = list(transcript_list)
                if all_transcripts:
                    # Use the first available transcript
                    transcript = all_transcripts[0]
                    transcript_data = transcript.fetch()
                    
                    text_parts = []
                    for entry in transcript_data:
                        if isinstance(entry, dict) and 'text' in entry and entry['text'].strip():
                            text_parts.append(entry['text'].strip())
                    
                    if text_parts:
                        text_content = ' '.join(text_parts)
                        language_code = getattr(transcript, 'language_code', 'unknown')
                        logger.info("Found transcript in %s for video %s", language_code, video_id)
                        return text_content
                        
            except Exception as e:
                logger.error("Error fetching available transcript: %s", str(e))
            
            logger.warning("No transcript found for video %s", video_id)
            return None
            
        except Exception as e:
            error_message = str(e)
            if "Could not retrieve a transcript" in error_message:
                logger.error("No transcript available for video %s", video_id)
            elif "no element found" in error_message.lower():
                logger.error("XML parsing error for video %s - transcript may be corrupted", video_id)
            else:
                logger.error("Error fetching transcript for video %s: %s", video_id, error_message)
            return None
    
    @staticmethod
    def get_video_info(video_id: str) -> Dict:
        """
        Get basic video information
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with video information
        """
        try:
            # Get available transcripts to check if video exists
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Get available languages
            available_languages = []
            try:
                for transcript in transcript_list._manually_created_transcripts.values():
                    available_languages.append(f"{transcript.language} (manual)")
            except:
                pass
                
            try:
                for transcript in transcript_list._generated_transcripts.values():
                    available_languages.append(f"{transcript.language} (auto)")
            except:
                pass
            
            return {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "available_transcripts": available_languages,
                "transcript_available": len(available_languages) > 0
            }
            
        except Exception as e:
            logger.error(f"Error getting video info for {video_id}: {str(e)}")
            return {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "available_transcripts": [],
                "transcript_available": False,
                "error": str(e)
            }
    
    @staticmethod
    def validate_youtube_url(youtube_url: str) -> bool:
        """
        Validate YouTube URL and check if transcript is available
        
        Args:
            youtube_url: YouTube URL to validate
            
        Returns:
            True if valid and transcript available, False otherwise
        """
        if not youtube_url or not youtube_url.strip():
            st.error("Please enter a YouTube URL.")
            return False
        
        # Extract video ID
        video_id = YouTubeLoader.extract_video_id(youtube_url)
        if not video_id:
            st.error("❌ Invalid YouTube URL format. Please use a standard YouTube URL.")
            st.info("💡 Supported formats: https://youtube.com/watch?v=ID or https://youtu.be/ID")
            return False
        
        # Check if video exists and has transcripts
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available_transcripts = list(transcript_list)
            
            if not available_transcripts:
                st.error("❌ No transcripts available for this video.")
                st.info("💡 Try a video with captions/subtitles enabled, such as:")
                st.info("- Educational content (TED Talks, Khan Academy)")
                st.info("- News videos")
                st.info("- Popular videos with community captions")
                return False
            
            # Show available transcript languages
            languages = []
            for transcript in available_transcripts:
                lang_info = getattr(transcript, 'language_code', 'unknown')
                if hasattr(transcript, 'is_generated') and transcript.is_generated:
                    languages.append(f"{lang_info} (auto)")
                else:
                    languages.append(f"{lang_info} (manual)")
            
            st.success(f"✅ Video found with transcripts: {', '.join(languages[:3])}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "Could not retrieve a transcript" in error_msg:
                st.error("❌ This video doesn't have any transcripts available.")
                st.info("💡 Please try a different video with captions enabled.")
            elif "Private video" in error_msg or "unavailable" in error_msg.lower():
                st.error("❌ This video is private, unavailable, or restricted.")
            elif "no element found" in error_msg.lower():
                st.error("❌ Transcript data is corrupted or unavailable for this video.")
                st.info("💡 Try a different video or check if the video has proper captions.")
            else:
                st.error(f"❌ Error accessing video: {error_msg}")
            return False
    
    @staticmethod
    def process_youtube_url(youtube_url: str) -> Optional[Dict]:
        """
        Process YouTube URL and return transcript with metadata
        
        Args:
            youtube_url: YouTube URL to process
            
        Returns:
            Dictionary with transcript and metadata or None if failed
        """
        try:
            # Extract video ID
            video_id = YouTubeLoader.extract_video_id(youtube_url)
            if not video_id:
                st.error("Could not extract video ID from URL.")
                return None
            
            # Get video info
            video_info = YouTubeLoader.get_video_info(video_id)
            
            # Get transcript
            transcript = YouTubeLoader.get_video_transcript(video_id)
            if not transcript:
                st.error("Could not fetch transcript for this video.")
                return None
            
            return {
                "content": transcript,
                "source_type": "youtube",
                "video_id": video_id,
                "video_url": video_info["video_url"],
                "available_languages": video_info["available_transcripts"],
                "content_length": len(transcript)
            }
            
        except Exception as e:
            logger.error(f"Error processing YouTube URL: {str(e)}")
            st.error(f"Error processing YouTube URL: {str(e)}")
            return None
