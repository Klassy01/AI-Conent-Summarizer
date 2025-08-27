"""
Video Analysis Module for YouTube Videos
Handles video frame extraction, visual analysis, and content understanding
"""

import os
import tempfile
import logging
from typing import Optional, List, Dict, Any
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import io

# Try importing computer vision and AI models
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """Analyzes YouTube videos without transcripts using visual content"""
    
    def __init__(self):
        """Initialize video analyzer with AI models"""
        self.blip_processor = None
        self.blip_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TORCH_AVAILABLE else "cpu"
        
        # Initialize BLIP model for image captioning
        if BLIP_AVAILABLE and TORCH_AVAILABLE:
            try:
                st.info("🤖 Loading visual analysis AI model...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model.to(self.device)
                logger.info("BLIP model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load BLIP model: {e}")
                self.blip_processor = None
                self.blip_model = None
    
    def download_video(self, video_id: str, max_duration: int = 300) -> Optional[str]:
        """
        Download video for analysis
        
        Args:
            video_id: YouTube video ID
            max_duration: Maximum video duration to download (seconds)
            
        Returns:
            Path to downloaded video file
        """
        if not YT_DLP_AVAILABLE:
            return None
            
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            video_path = os.path.join(temp_dir, f"{video_id}.%(ext)s")
            
            ydl_opts = {
                'format': 'best[height<=480][duration<={max_duration}]'.format(max_duration=max_duration),
                'outtmpl': video_path,
                'quiet': True,
                'no_warnings': True,
                'extractaudio': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get video info first
                info = ydl.extract_info(url, download=False)
                duration = info.get('duration', 0)
                
                if duration > max_duration:
                    st.warning(f"⚠️ Video is {duration}s long, analyzing first {max_duration}s only")
                
                # Download video
                ydl.download([url])
                
                # Find the downloaded file
                for file in os.listdir(temp_dir):
                    if video_id in file and not file.endswith('.part'):
                        return os.path.join(temp_dir, file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading video {video_id}: {e}")
            return None
    
    def extract_keyframes(self, video_path: str, max_frames: int = 20) -> List[np.ndarray]:
        """
        Extract key frames from video for analysis
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frame arrays
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame extraction interval
            if total_frames <= max_frames:
                interval = 1
            else:
                interval = total_frames // max_frames
            
            logger.info(f"Extracting frames from video: {total_frames} total, every {interval} frames")
            
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < max_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames for analysis")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            return []
    
    def analyze_frame_with_blip(self, frame: np.ndarray) -> Optional[str]:
        """
        Analyze a single frame using BLIP model
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Caption/description of the frame
        """
        if not self.blip_processor or not self.blip_model:
            return None
            
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(frame)
            
            # Process image
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            logger.error(f"Error analyzing frame with BLIP: {e}")
            return None
    
    def analyze_frame_with_gemini(self, frame: np.ndarray) -> Optional[str]:
        """
        Analyze frame using Google Gemini Vision API
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Description of the frame
        """
        try:
            import google.generativeai as genai
            
            # Check if Gemini API key is available
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key or api_key == "your_google_api_key_here":
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Convert frame to PIL Image
            image = Image.fromarray(frame)
            
            # Prepare prompt for video frame analysis
            prompt = """Analyze this video frame and describe what you see. Focus on:
            - Main subjects or objects
            - Activities or actions taking place
            - Text or graphics visible
            - Setting or environment
            - Key visual elements
            
            Provide a concise but informative description in 1-2 sentences."""
            
            # Generate response
            response = model.generate_content([prompt, image])
            
            if response.text:
                return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error analyzing frame with Gemini: {e}")
            
        return None
    
    def detect_scene_changes(self, frames: List[np.ndarray], threshold: float = 0.3) -> List[int]:
        """
        Detect scene changes between frames
        
        Args:
            frames: List of frames
            threshold: Threshold for scene change detection
            
        Returns:
            List of frame indices where scene changes occur
        """
        try:
            if len(frames) < 2:
                return []
            
            scene_changes = [0]  # Always include first frame
            
            for i in range(1, len(frames)):
                # Convert to grayscale for comparison
                gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                
                # Calculate histogram difference
                hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                
                # Compare histograms
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                if correlation < (1 - threshold):
                    scene_changes.append(i)
            
            return scene_changes
            
        except Exception as e:
            logger.error(f"Error detecting scene changes: {e}")
            return list(range(min(5, len(frames))))  # Fallback to first 5 frames
    
    def analyze_video_content(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Perform complete video analysis
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with analysis results
        """
        try:
            st.info("🎬 Starting video analysis (this may take several minutes)...")
            
            # Download video
            progress_bar = st.progress(0.1)
            st.text("Downloading video...")
            
            video_path = self.download_video(video_id)
            if not video_path:
                st.error("Failed to download video for analysis")
                return None
            
            progress_bar.progress(0.3)
            st.text("Extracting key frames...")
            
            # Extract frames
            frames = self.extract_keyframes(video_path)
            if not frames:
                st.error("Failed to extract frames from video")
                return None
            
            progress_bar.progress(0.5)
            st.text("Detecting scene changes...")
            
            # Detect scene changes
            key_frame_indices = self.detect_scene_changes(frames)
            key_frames = [frames[i] for i in key_frame_indices]
            
            progress_bar.progress(0.7)
            st.text("Analyzing visual content...")
            
            # Analyze key frames
            frame_descriptions = []
            
            for i, frame in enumerate(key_frames):
                st.text(f"Analyzing frame {i+1}/{len(key_frames)}...")
                
                # Try Gemini first (better quality)
                description = self.analyze_frame_with_gemini(frame)
                
                # Fallback to BLIP if Gemini fails
                if not description:
                    description = self.analyze_frame_with_blip(frame)
                
                if description:
                    frame_descriptions.append({
                        "frame_index": key_frame_indices[i],
                        "timestamp": f"{(key_frame_indices[i] * 2):.1f}s",  # Approximate timestamp
                        "description": description
                    })
            
            progress_bar.progress(0.9)
            st.text("Generating summary...")
            
            # Create comprehensive analysis
            analysis_result = {
                "video_id": video_id,
                "total_frames_extracted": len(frames),
                "key_frames_analyzed": len(key_frames),
                "frame_descriptions": frame_descriptions,
                "content_summary": self.generate_video_summary(frame_descriptions),
                "analysis_method": "visual_analysis"
            }
            
            progress_bar.progress(1.0)
            st.success(f"✅ Video analysis complete! Analyzed {len(key_frames)} key scenes")
            
            # Cleanup
            try:
                os.remove(video_path)
                os.rmdir(os.path.dirname(video_path))
            except:
                pass
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing video content: {e}")
            st.error(f"Error analyzing video: {str(e)}")
            return None
    
    def generate_video_summary(self, frame_descriptions: List[Dict]) -> str:
        """
        Generate a summary from frame descriptions
        
        Args:
            frame_descriptions: List of frame analysis results
            
        Returns:
            Comprehensive summary text
        """
        if not frame_descriptions:
            return "No visual content could be analyzed."
        
        summary_parts = []
        summary_parts.append("VIDEO VISUAL ANALYSIS SUMMARY")
        summary_parts.append("=" * 40)
        summary_parts.append("")
        
        summary_parts.append(f"This video contains {len(frame_descriptions)} distinct visual scenes:")
        summary_parts.append("")
        
        for i, desc in enumerate(frame_descriptions, 1):
            timestamp = desc.get("timestamp", "Unknown")
            description = desc.get("description", "No description available")
            summary_parts.append(f"Scene {i} ({timestamp}): {description}")
        
        summary_parts.append("")
        summary_parts.append("OVERALL CONTENT:")
        
        # Extract common themes
        all_descriptions = [desc["description"] for desc in frame_descriptions if desc.get("description")]
        
        if all_descriptions:
            combined_text = " ".join(all_descriptions)
            summary_parts.append(f"The video shows: {combined_text}")
        else:
            summary_parts.append("Visual content analysis was not successful.")
        
        return "\n".join(summary_parts)

def is_video_analysis_available() -> Dict[str, bool]:
    """Check which video analysis capabilities are available"""
    return {
        "opencv": True,  # Should always be available after pip install
        "yt_dlp": YT_DLP_AVAILABLE,
        "blip_model": BLIP_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "gemini_vision": bool(os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_API_KEY") != "your_google_api_key_here")
    }
