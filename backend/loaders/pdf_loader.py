"""
PDF loading and text extraction utilities
"""

import streamlit as st
from PyPDF2 import PdfReader
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class PDFLoader:
    """Handle PDF file loading and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> Optional[str]:
        """
        Extract text content from uploaded PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            reader = PdfReader(pdf_file)
            text_content = ""
            
            # Extract text from all pages
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {str(e)}")
                    continue
            
            if not text_content.strip():
                st.error("No text content found in the PDF file.")
                return None
                
            logger.info(f"Successfully extracted {len(text_content)} characters from PDF")
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            st.error(f"Error processing PDF file: {str(e)}")
            return None
    
    @staticmethod
    def validate_pdf_file(pdf_file) -> bool:
        """
        Validate if uploaded file is a proper PDF
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            True if valid PDF, False otherwise
        """
        if pdf_file is None:
            return False
            
        # Check file extension
        if not pdf_file.name.lower().endswith('.pdf'):
            st.error("Please upload a PDF file.")
            return False
        
        # Check file size (limit to 50MB)
        if pdf_file.size > 50 * 1024 * 1024:
            st.error("PDF file size should be less than 50MB.")
            return False
            
        try:
            # Try to read the PDF to validate format
            reader = PdfReader(pdf_file)
            if len(reader.pages) == 0:
                st.error("PDF file appears to be empty.")
                return False
                
            # Reset file pointer for later use
            pdf_file.seek(0)
            return True
            
        except Exception as e:
            st.error(f"Invalid PDF file: {str(e)}")
            return False

    @staticmethod
    def get_pdf_metadata(pdf_file) -> dict:
        """
        Extract metadata from PDF file
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            reader = PdfReader(pdf_file)
            metadata = {
                "num_pages": len(reader.pages),
                "file_size": pdf_file.size,
                "file_name": pdf_file.name,
            }
            
            # Try to get PDF metadata
            if reader.metadata:
                metadata.update({
                    "title": reader.metadata.get("/Title", "Unknown"),
                    "author": reader.metadata.get("/Author", "Unknown"),
                    "subject": reader.metadata.get("/Subject", "Unknown"),
                })
            
            # Reset file pointer
            pdf_file.seek(0)
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not extract PDF metadata: {str(e)}")
            return {
                "num_pages": "Unknown",
                "file_size": pdf_file.size if pdf_file else 0,
                "file_name": pdf_file.name if pdf_file else "Unknown",
            }
