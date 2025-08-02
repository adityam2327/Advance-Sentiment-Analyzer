import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import time
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import hashlib
from urllib.parse import urlparse, parse_qs
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CSV_ROWS = 100
MIN_TEXT_LENGTH = 5
MAX_TEXT_LENGTH = 10000
API_DELAY = 0.5  # Delay between API calls to avoid rate limiting

class SentimentCategory(Enum):
    VERY_POSITIVE = "Very Positive"
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    VERY_NEGATIVE = "Very Negative"

@dataclass
class SentimentResult:
    text: str
    category: SentimentCategory
    score: float
    analysis: str
    confidence: float = 0.0

class SentimentAnalyzer:
    """Main class for sentiment analysis operations."""
    
    def __init__(self, api_key: str):
        """Initialize the sentiment analyzer with Gemini API."""
        self.api_key = api_key
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Gemini model."""
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze sentiment of given text."""
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return SentimentResult(
                text=text,
                category=SentimentCategory.NEUTRAL,
                score=0.0,
                analysis="Text too short for meaningful analysis",
                confidence=0.0
            )
        
        # Truncate text if too long
        text = text[:MAX_TEXT_LENGTH]
        
        prompt = self._create_sentiment_prompt(text)
        
        try:
            response = self.model.generate_content(prompt)
            return self._parse_sentiment_response(text, response.text)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return SentimentResult(
                text=text,
                category=SentimentCategory.NEUTRAL,
                score=0.0,
                analysis=f"Error analyzing sentiment: {str(e)}",
                confidence=0.0
            )
    
    def _create_sentiment_prompt(self, text: str) -> str:
        """Create a structured prompt for sentiment analysis."""
        return f"""
        Analyze the sentiment of the following text with high precision.

        Provide your response in this exact format:
        CATEGORY: [Very Positive/Positive/Neutral/Negative/Very Negative]
        SCORE: [number between -1.0 and 1.0]
        CONFIDENCE: [number between 0.0 and 1.0]
        ANALYSIS: [2-3 sentence explanation]

        Guidelines for scoring:
        - Very Positive: 0.7 to 1.0 (enthusiastic, joyful, extremely satisfied)
        - Positive: 0.3 to 0.6 (happy, satisfied, optimistic)
        - Neutral: -0.2 to 0.2 (balanced, factual, no clear emotion)
        - Negative: -0.6 to -0.3 (disappointed, unhappy, critical)
        - Very Negative: -1.0 to -0.7 (angry, hateful, extremely dissatisfied)

        Text to analyze:
        \"\"\"{text}\"\"\"
        """
    
    def _parse_sentiment_response(self, original_text: str, response: str) -> SentimentResult:
        """Parse the Gemini response into a structured result."""
        try:
            lines = response.strip().split('\n')
            category_line = next((line for line in lines if line.startswith('CATEGORY:')), '')
            score_line = next((line for line in lines if line.startswith('SCORE:')), '')
            confidence_line = next((line for line in lines if line.startswith('CONFIDENCE:')), '')
            analysis_line = next((line for line in lines if line.startswith('ANALYSIS:')), '')
            
            # Extract category
            category_text = category_line.replace('CATEGORY:', '').strip()
            category = self._map_category(category_text)
            
            # Extract score
            score_match = re.search(r'SCORE:\s*(-?\d+\.?\d*)', response)
            score = float(score_match.group(1)) if score_match else 0.0
            score = max(-1.0, min(1.0, score))  # Clamp to valid range
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*(\d+\.?\d*)', response)
            confidence = float(confidence_match.group(1)) if confidence_match else 0.8
            confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range
            
            # Extract analysis
            analysis = analysis_line.replace('ANALYSIS:', '').strip() if analysis_line else response
            
            return SentimentResult(
                text=original_text,
                category=category,
                score=score,
                analysis=analysis,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error parsing sentiment response: {e}")
            # Fallback parsing
            return self._fallback_parse(original_text, response)
    
    def _fallback_parse(self, original_text: str, response: str) -> SentimentResult:
        """Fallback parsing when structured parsing fails."""
        response_lower = response.lower()
        
        # Determine category and score based on keywords
        if any(word in response_lower for word in ['very positive', 'extremely positive', 'highly positive']):
            category = SentimentCategory.VERY_POSITIVE
            score = 0.85
        elif any(word in response_lower for word in ['positive', 'good', 'great', 'excellent']):
            category = SentimentCategory.POSITIVE
            score = 0.5
        elif any(word in response_lower for word in ['very negative', 'extremely negative', 'highly negative']):
            category = SentimentCategory.VERY_NEGATIVE
            score = -0.85
        elif any(word in response_lower for word in ['negative', 'bad', 'poor', 'terrible']):
            category = SentimentCategory.NEGATIVE
            score = -0.5
        else:
            category = SentimentCategory.NEUTRAL
            score = 0.0
        
        return SentimentResult(
            text=original_text,
            category=category,
            score=score,
            analysis=response[:500],  # Truncate if too long
            confidence=0.6
        )
    
    def _map_category(self, category_text: str) -> SentimentCategory:
        """Map category text to SentimentCategory enum."""
        category_map = {
            'very positive': SentimentCategory.VERY_POSITIVE,
            'positive': SentimentCategory.POSITIVE,
            'neutral': SentimentCategory.NEUTRAL,
            'negative': SentimentCategory.NEGATIVE,
            'very negative': SentimentCategory.VERY_NEGATIVE
        }
        return category_map.get(category_text.lower(), SentimentCategory.NEUTRAL)

class YouTubeAnalyzer:
    """Handle YouTube video analysis with multiple fallback methods."""
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
            r'(?:shorts/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    def get_transcript_via_api(video_id: str) -> Tuple[str, Dict[str, Any]]:
        """Get transcript using direct API approach."""
        try:
            # Try multiple methods to get transcript
            transcript_methods = [
                YouTubeAnalyzer._get_transcript_method1,
                YouTubeAnalyzer._get_transcript_method2,
                YouTubeAnalyzer._get_transcript_method3
            ]
            
            for method in transcript_methods:
                try:
                    text, metadata = method(video_id)
                    if text and len(text.strip()) > MIN_TEXT_LENGTH:
                        return text, metadata
                except Exception as e:
                    logger.warning(f"Transcript method failed: {e}")
                    continue
            
            return "", {"error": "No transcript available"}
            
        except Exception as e:
            logger.error(f"Error getting transcript: {e}")
            return "", {"error": str(e)}
    
    @staticmethod
    def _get_transcript_method1(video_id: str) -> Tuple[str, Dict[str, Any]]:
        """Method 1: Try youtube-transcript-api if available."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text = " ".join([entry['text'] for entry in transcript])
            metadata = {
                'source': 'transcript_api',
                'method': 'youtube-transcript-api',
                'duration': len(transcript)
            }
            return text, metadata
        except ImportError:
            logger.warning("youtube-transcript-api not available")
            raise Exception("youtube-transcript-api not installed")
        except Exception as e:
            logger.warning(f"youtube-transcript-api failed: {e}")
            raise e
    
    @staticmethod
    def _get_transcript_method2(video_id: str) -> Tuple[str, Dict[str, Any]]:
        """Method 2: Try pytubefix if available."""
        try:
            from pytubefix import YouTube
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            
            # Get captions
            captions = yt.captions
            if captions:
                # Try English first, then any available language
                caption = captions.get('en') or captions.get('a.en') or list(captions.values())[0]
                if caption:
                    transcript_text = caption.generate_srt_captions()
                    # Clean SRT format
                    text = YouTubeAnalyzer._clean_srt_text(transcript_text)
                    metadata = {
                        'source': 'captions',
                        'method': 'pytubefix',
                        'title': yt.title,
                        'views': yt.views,
                        'length': yt.length,
                        'author': yt.author
                    }
                    return text, metadata
            
            # Fallback to description
            if yt.description and len(yt.description.strip()) > MIN_TEXT_LENGTH:
                metadata = {
                    'source': 'description',
                    'method': 'pytubefix',
                    'title': yt.title,
                    'views': yt.views,
                    'length': yt.length,
                    'author': yt.author
                }
                return yt.description, metadata
            
            raise Exception("No captions or description available")
            
        except ImportError:
            logger.warning("pytubefix not available")
            raise Exception("pytubefix not installed")
        except Exception as e:
            logger.warning(f"pytubefix failed: {e}")
            raise e
    
    @staticmethod
    def _get_transcript_method3(video_id: str) -> Tuple[str, Dict[str, Any]]:
        """Method 3: Try manual web scraping approach."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Get video page
            url = f"https://www.youtube.com/watch?v={video_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Try to extract basic info and description
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title_tag = soup.find('meta', property='og:title')
            title = title_tag['content'] if title_tag else "Unknown Title"
            
            # Extract description
            desc_tag = soup.find('meta', property='og:description')
            description = desc_tag['content'] if desc_tag else ""
            
            if description and len(description.strip()) > MIN_TEXT_LENGTH:
                metadata = {
                    'source': 'description',
                    'method': 'web_scraping',
                    'title': title
                }
                return description, metadata
            
            raise Exception("No description found via web scraping")
            
        except Exception as e:
            logger.warning(f"Web scraping method failed: {e}")
            raise e
    
    @staticmethod
    def _clean_srt_text(srt_text: str) -> str:
        """Clean SRT subtitle format to plain text."""
        if not srt_text:
            return ""
        
        # Remove SRT timestamps and formatting
        lines = srt_text.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, numbers, and timestamp lines
            if (line and 
                not line.isdigit() and 
                not re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line)):
                text_lines.append(line)
        
        return ' '.join(text_lines)
    
    @staticmethod
    def extract_content(video_url: str) -> Tuple[str, str, Dict[str, Any]]:
        """Extract content from YouTube video with multiple fallback methods."""
        try:
            video_id = YouTubeAnalyzer.extract_video_id(video_url)
            if not video_id:
                return "", "error", {"error": "Invalid YouTube URL"}
            
            # Try to get transcript
            text, metadata = YouTubeAnalyzer.get_transcript_via_api(video_id)
            
            if text and len(text.strip()) > MIN_TEXT_LENGTH:
                return text, metadata.get('source', 'transcript'), metadata
            
            # If no transcript, try alternative method
            return YouTubeAnalyzer._fallback_content_extraction(video_id)
                
        except Exception as e:
            logger.error(f"Error extracting YouTube content: {e}")
            return "", "error", {"error": str(e)}
    
    @staticmethod
    def _fallback_content_extraction(video_id: str) -> Tuple[str, str, Dict[str, Any]]:
        """Fallback content extraction method."""
        try:
            # Simple approach - provide instructions for manual input
            fallback_text = f"""
            Unable to automatically extract content from this YouTube video (ID: {video_id}).
            
            This could be due to:
            - Video has no captions/transcript
            - Video is private or age-restricted
            - API limitations or network issues
            
            You can manually copy the video description or transcript and analyze it using the Text Analysis tab.
            """
            
            metadata = {
                'source': 'fallback',
                'method': 'manual_instruction',
                'video_id': video_id
            }
            
            return fallback_text, "fallback", metadata
            
        except Exception as e:
            return "", "error", {"error": str(e)}

class CSVProcessor:
    """Handle CSV file processing for batch analysis."""
    
    @staticmethod
    def identify_text_columns(df: pd.DataFrame) -> list:
        """Identify potential text columns in the DataFrame."""
        text_keywords = ['text', 'content', 'comment', 'description', 'review', 'feedback', 'message']
        text_columns = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in text_keywords):
                text_columns.append(col)
        
        # If no keyword matches, look for string columns with meaningful content
        if not text_columns:
            for col in df.columns:
                if df[col].dtype == 'object':
                    sample_text = df[col].dropna().astype(str)
                    if len(sample_text) > 0:
                        avg_length = sample_text.str.len().mean()
                        if avg_length > 10:  # Arbitrary threshold for meaningful text
                            text_columns.append(col)
        
        return text_columns
    
    @staticmethod
    def process_dataframe(df: pd.DataFrame, analyzer: SentimentAnalyzer, 
                         progress_callback=None) -> Tuple[pd.DataFrame, Optional[str]]:
        """Process DataFrame for sentiment analysis."""
        try:
            # Identify text columns
            text_columns = CSVProcessor.identify_text_columns(df)
            
            if not text_columns:
                return None, "No suitable text columns found. Please ensure your CSV contains text data."
            
            text_column = text_columns[0]  # Use the first identified text column
            
            # Limit rows to avoid API overuse
            original_row_count = len(df)
            if len(df) > MAX_CSV_ROWS:
                df = df.head(MAX_CSV_ROWS)
            
            # Process each row
            results = []
            total_rows = len(df)
            
            for i, row in enumerate(df[text_column].fillna('').astype(str)):
                if progress_callback:
                    progress_callback(i + 1, total_rows, f"Analyzing row {i+1}/{total_rows}")
                
                if len(row.strip()) < MIN_TEXT_LENGTH:
                    result = SentimentResult(
                        text=row,
                        category=SentimentCategory.NEUTRAL,
                        score=0.0,
                        analysis="Text too short for analysis",
                        confidence=0.0
                    )
                else:
                    result = analyzer.analyze_text(row)
                
                results.append({
                    'sentiment_category': result.category.value,
                    'sentiment_score': result.score,
                    'sentiment_analysis': result.analysis,
                    'confidence': result.confidence
                })
                
                # Rate limiting
                time.sleep(API_DELAY)
            
            # Add results to DataFrame
            results_df = pd.DataFrame(results)
            final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
            
            return final_df, None
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return None, f"Error processing CSV: {str(e)}"

class UIComponents:
    """UI component generators."""
    
    @staticmethod
    def create_sentiment_gauge(score: float, title: str = "Sentiment Score") -> go.Figure:
        """Create a sentiment gauge visualization."""
        # Determine color based on score
        if score <= -0.6:
            color = "#d32f2f"  # Dark red
        elif score <= -0.2:
            color = "#ff5722"  # Orange red
        elif score <= 0.2:
            color = "#ff9800"  # Orange
        elif score <= 0.6:
            color = "#4caf50"  # Green
        else:
            color = "#2e7d32"  # Dark green
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 20, 'color': 'darkblue'}},
            delta={'reference': 0, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {
                    'range': [-1, 1], 
                    'tickwidth': 2, 
                    'tickcolor': "darkblue",
                    'tickmode': 'linear',
                    'tick0': -1,
                    'dtick': 0.2
                },
                'bar': {'color': color, 'thickness': 0.8},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': "gray",
                'steps': [
                    {'range': [-1, -0.6], 'color': 'rgba(211, 47, 47, 0.2)'},    # Very negative
                    {'range': [-0.6, -0.2], 'color': 'rgba(255, 87, 34, 0.2)'},  # Negative
                    {'range': [-0.2, 0.2], 'color': 'rgba(255, 152, 0, 0.2)'},   # Neutral
                    {'range': [0.2, 0.6], 'color': 'rgba(76, 175, 80, 0.2)'},    # Positive
                    {'range': [0.6, 1], 'color': 'rgba(46, 125, 50, 0.2)'}       # Very positive
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': score
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=60, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "darkblue", 'family': "Arial, sans-serif"}
        )
        return fig
    
    @staticmethod
    def format_sentiment_display(result: SentimentResult) -> str:
        """Format sentiment result for display."""
        color_map = {
            SentimentCategory.VERY_POSITIVE: "#2e7d32",
            SentimentCategory.POSITIVE: "#4caf50",
            SentimentCategory.NEUTRAL: "#ff9800",
            SentimentCategory.NEGATIVE: "#ff5722",
            SentimentCategory.VERY_NEGATIVE: "#d32f2f"
        }
        
        color = color_map.get(result.category, "#ff9800")
        
        return f"""
        <div style="padding: 15px; border-left: 5px solid {color}; background-color: rgba(0,0,0,0.05); border-radius: 5px;">
            <h4 style="color: {color}; margin: 0 0 10px 0;">{result.category.value}</h4>
            <p style="margin: 5px 0;"><strong>Score:</strong> {result.score:.2f}</p>
            <p style="margin: 5px 0;"><strong>Confidence:</strong> {result.confidence:.1%}</p>
            <p style="margin: 10px 0 0 0;">{result.analysis}</p>
        </div>
        """

# Streamlit App Configuration
def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Advanced Sentiment Analyzer",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """Load custom CSS for better UI."""
    st.markdown("""
        <style>
            .main-header {
                font-size: 48px; 
                font-weight: bold; 
                margin-bottom: 20px;
                text-align: center;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            .sub-header {
                font-size: 28px; 
                margin-top: 30px; 
                margin-bottom: 15px;
                color: #1f4e79;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 5px;
            }
            .stTabs [data-baseweb="tab-list"] {
                gap: 24px;
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 10px;
            }
            .stTabs [data-baseweb="tab"] {
                height: 60px;
                white-space: pre-wrap;
                background-color: white;
                border-radius: 10px;
                border: 2px solid #e0e0e0;
                color: #1f4e79;
                font-weight: 600;
                padding: 15px 20px;
                transition: all 0.3s ease;
            }
            .stTabs [data-baseweb="tab"]:hover {
                border-color: #667eea;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white !important;
                border-color: #667eea;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            .stButton>button {
                width: 100%;
                border-radius: 10px;
                height: 3.5em;
                font-weight: 600;
                font-size: 16px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                color: white;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
            }
            .metric-card {
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
                margin: 10px 0;
            }
            .info-box {
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #17a2b8;
                margin: 15px 0;
            }
            div.block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .stProgress .st-bo {
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            }
            .stSelectbox > div > div {
                border-radius: 10px;
            }
            .stTextInput > div > div > input {
                border-radius: 10px;
                border: 2px solid #e0e0e0;
                transition: border-color 0.3s ease;
            }
            .stTextInput > div > div > input:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
            }
            .stTextArea > div > div > textarea {
                border-radius: 10px;
                border: 2px solid #e0e0e0;
                transition: border-color 0.3s ease;
            }
            .stTextArea > div > div > textarea:focus {
                border-color: #667eea;
                box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
            }
        </style>
    """, unsafe_allow_html=True)

def create_sidebar():
    """Create and populate the sidebar."""
    with st.sidebar:
        st.markdown("## 🎭 About This App")
        st.markdown("""
        <div class="info-box">
        This advanced sentiment analyzer uses Google's Gemini AI to provide comprehensive sentiment analysis with:
        <ul>
            <li>🎥 YouTube video analysis (multiple methods)</li>
            <li>📝 Direct text analysis</li>
            <li>📊 Batch CSV processing</li>
            <li>📈 Advanced visualizations</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎥 YouTube Analysis")
        st.markdown("""
        <div class="metric-card">
        The app tries multiple methods to extract YouTube content:
        <ol>
            <li><strong>Transcript API:</strong> Gets official captions</li>
            <li><strong>PyTubeFix:</strong> Alternative video extraction</li>
            <li><strong>Web Scraping:</strong> Fallback description extraction</li>
        </ol>
        <p><small>⚠️ Some videos may not have accessible transcripts due to privacy settings or lack of captions.</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Sentiment Scoring")
        st.markdown("""
        <div class="metric-card">
        <strong>Score Range:</strong> -1.0 to +1.0<br>
        <strong>+0.6 to +1.0:</strong> Very Positive<br>
        <strong>+0.2 to +0.6:</strong> Positive<br>
        <strong>-0.2 to +0.2:</strong> Neutral<br>
        <strong>-0.6 to -0.2:</strong> Negative<br>
        <strong>-1.0 to -0.6:</strong> Very Negative
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📦 Required Libraries")
        st.markdown("""
        <div class="info-box">
        For full YouTube functionality, install:<br>
        <code>pip install youtube-transcript-api pytubefix beautifulsoup4</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📁 CSV Format Guidelines")
        st.markdown("""
        Your CSV should contain a column with text data. 
        Supported column names include: 'text', 'content', 'comment', 'description', 'review', 'feedback', 'message'.
        """)
        
        # Sample CSV download
        sample_data = pd.DataFrame({
            'text': [
                'I absolutely love this product! It exceeded my expectations.',
                'The service was terrible and the staff was rude.',
                'The item was okay, nothing special but not bad either.',
                'Amazing quality and fast delivery! Highly recommend.',
                'Worst purchase ever. Complete waste of money.'
            ],
            'category': ['Electronics', 'Service', 'Clothing', 'Electronics', 'Service']
        })
        
        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_data.to_csv(index=False),
            file_name="sample_sentiment_data.csv",
            mime="text/csv",
            help="Download a sample CSV file to see the expected format"
        )

def main():
    """Main application function."""
    configure_page()
    load_custom_css()
    create_sidebar()
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    # Main header
    st.markdown('<h1 class="main-header">🎭 Advanced Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; font-size: 18px; color: #666; margin-bottom: 30px;">
    Powered by Google Gemini AI • Analyze sentiment from multiple sources with confidence scores
    </div>
    """, unsafe_allow_html=True)
    
    # API Key Configuration
    api_key = st.text_input(
        "🔑 Enter your Google Gemini API Key:",
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key to continue.")
        st.info("ℹ️ You can get a free API key from Google AI Studio: https://makersuite.google.com/app/apikey")
        return
    
    # Initialize analyzer
    try:
        if st.session_state.analyzer is None:
            with st.spinner("🔄 Initializing AI model..."):
                st.session_state.analyzer = SentimentAnalyzer(api_key)
            st.success("✅ AI model initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize AI model: {str(e)}")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🎥 YouTube Analysis", 
        "📝 Text Analysis", 
        "📊 Batch Analysis"
    ])
    
    # Tab 1: YouTube Analysis
    with tab1:
        st.markdown('<h2 class="sub-header">🎥 YouTube Video Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            youtube_url = st.text_input(
                "📎 YouTube Video URL:",
                placeholder="https://www.youtube.com/watch?v=...",
                help="Enter a valid YouTube video URL"
            )
        with col2:
            analyze_youtube = st.button("🔍 Analyze Video", key="analyze_youtube")
        
        if youtube_url and analyze_youtube:
            if not any(domain in youtube_url.lower() for domain in ['youtube.com', 'youtu.be']):
                st.error("⚠️ Please enter a valid YouTube URL")
                return
            
            with st.spinner("📥 Extracting video content... (trying multiple methods)"):
                text, source_type, metadata = YouTubeAnalyzer.extract_content(youtube_url)
            
            if source_type == "error":
                st.error(f"❌ Error: {metadata.get('error', 'Unknown error')}")
                st.info("💡 **Tip:** Try copying the video transcript or description manually and use the Text Analysis tab instead.")
                return
            
            if source_type == "fallback":
                st.warning("⚠️ Could not automatically extract video content")
                st.info(text)
                st.markdown("""
                ### 🔧 Manual Alternative
                1. Go to the YouTube video
                2. Look for captions/transcript (CC button)
                3. Copy the transcript or video description
                4. Use the **Text Analysis** tab to analyze it
                """)
                return
            
            # Display video metadata
            if metadata and source_type not in ['error', 'fallback']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'title' in metadata:
                        st.metric("📺 Video Title", metadata['title'][:50] + "..." if len(metadata['title']) > 50 else metadata['title'])
                with col2:
                    if 'views' in metadata:
                        st.metric("👀 Views", f"{metadata['views']:,}")
                with col3:
                    if 'length' in metadata:
                        minutes = metadata['length'] // 60
                        seconds = metadata['length'] % 60
                        st.metric("⏱️ Duration", f"{minutes}:{seconds:02d}")
                    elif 'duration' in metadata:
                        st.metric("📝 Transcript Lines", metadata['duration'])
            
            success_msg = f"✅ Successfully extracted {source_type.title()}"
            if 'method' in metadata:
                success_msg += f" (via {metadata['method']})"
            st.success(success_msg)
            
            # Show extracted content
            with st.expander(f"📄 View Extracted {source_type.title()}", expanded=False):
                display_text = text[:2000] + "\n\n..." if len(text) > 2000 else text
                st.text_area("Content:", display_text, height=200, disabled=True)
                
                if len(text) > 2000:
                    st.info(f"📊 Total content length: {len(text):,} characters (showing first 2,000)")
            
            # Only analyze if we have meaningful content
            if len(text.strip()) < MIN_TEXT_LENGTH:
                st.warning("⚠️ Extracted content is too short for meaningful sentiment analysis")
                return
            
            # Analyze sentiment
            with st.spinner("🧠 Analyzing sentiment..."):
                result = st.session_state.analyzer.analyze_text(text)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🎯 Analysis Results")
                st.markdown(UIComponents.format_sentiment_display(result), unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Sentiment Gauge")
                fig = UIComponents.create_sentiment_gauge(result.score)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Text Analysis
    with tab2:
        st.markdown('<h2 class="sub-header">📝 Direct Text Analysis</h2>', unsafe_allow_html=True)
        
        text_input = st.text_area(
            "✍️ Enter text to analyze:",
            height=200,
            placeholder="Type or paste your text here...",
            help=f"Enter text between {MIN_TEXT_LENGTH} and {MAX_TEXT_LENGTH} characters"
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_text = st.button("🔍 Analyze Text Sentiment", type="primary")
        
        if analyze_text:
            if not text_input or len(text_input.strip()) < MIN_TEXT_LENGTH:
                st.error(f"⚠️ Please enter at least {MIN_TEXT_LENGTH} characters")
                return
            
            # Show text statistics
            word_count = len(text_input.split())
            char_count = len(text_input)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📝 Word Count", word_count)
            with col2:
                st.metric("🔤 Character Count", char_count)
            with col3:
                st.metric("📏 Text Length", "Good" if char_count <= MAX_TEXT_LENGTH else "Too Long")
            
            # Analyze sentiment
            with st.spinner("🧠 Analyzing sentiment..."):
                result = st.session_state.analyzer.analyze_text(text_input)
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("### 🎯 Analysis Results")
                st.markdown(UIComponents.format_sentiment_display(result), unsafe_allow_html=True)
            
            with col2:
                st.markdown("### 📊 Sentiment Gauge")
                fig = UIComponents.create_sentiment_gauge(result.score)
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Batch Analysis
    with tab3:
        st.markdown('<h2 class="sub-header">📊 Batch Sentiment Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        Upload a CSV file to analyze multiple texts at once. The system will automatically detect text columns 
        and process up to {} rows to manage API usage efficiently.
        </div>
        """.format(MAX_CSV_ROWS), unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "📁 Choose CSV file",
            type="csv",
            help="Upload a CSV file with text data for batch analysis"
        )
        
        if uploaded_file is not None:
            try:
                # Load and preview CSV
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                
                # Identify text columns
                text_columns = CSVProcessor.identify_text_columns(df)
                
                if not text_columns:
                    st.error("❌ No suitable text columns found. Please ensure your CSV contains text data.")
                    return
                
                # Column selection
                selected_column = st.selectbox(
                    "📋 Select text column to analyze:",
                    text_columns,
                    help="Choose which column contains the text you want to analyze"
                )
                
                # Preview data
                st.markdown("### 👀 Data Preview")
                preview_df = df[[selected_column]].head(10)
                st.dataframe(preview_df, use_container_width=True)
                
                # Analysis configuration
                col1, col2 = st.columns(2)
                with col1:
                    rows_to_analyze = min(len(df), MAX_CSV_ROWS)
                    st.info(f"📊 Will analyze {rows_to_analyze} rows")
                with col2:
                    estimated_time = rows_to_analyze * (API_DELAY + 2)  # Rough estimation
                    st.info(f"⏱️ Estimated time: {estimated_time//60}m {estimated_time%60}s")
                
                # Process button
                if st.button("🚀 Start Batch Analysis", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def progress_callback(current, total, message):
                        progress = current / total
                        progress_bar.progress(progress)
                        status_text.text(message)
                    
                    # Process the data
                    with st.spinner("🔄 Processing batch analysis..."):
                        results_df, error = CSVProcessor.process_dataframe(
                            df, st.session_state.analyzer, progress_callback
                        )
                    
                    if error:
                        st.error(f"❌ {error}")
                        return
                    
                    if results_df is not None:
                        # Clear progress indicators
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"🎉 Successfully analyzed {len(results_df)} entries!")
                        
                        # Create result tabs
                        result_tab1, result_tab2, result_tab3 = st.tabs([
                            "📋 Detailed Results", 
                            "📈 Visualizations", 
                            "📊 Summary Statistics"
                        ])
                        
                        with result_tab1:
                            st.markdown("### 📋 Analysis Results")
                            
                            # Display results with formatting
                            display_df = results_df.copy()
                            
                            # Format columns for better display
                            if 'sentiment_score' in display_df.columns:
                                display_df['sentiment_score'] = display_df['sentiment_score'].round(3)
                            if 'confidence' in display_df.columns:
                                display_df['confidence'] = (display_df['confidence'] * 100).round(1)
                            
                            st.dataframe(
                                display_df,
                                column_config={
                                    "sentiment_score": st.column_config.NumberColumn(
                                        "Sentiment Score",
                                        format="%.3f",
                                        help="Score from -1.0 (very negative) to 1.0 (very positive)"
                                    ),
                                    "confidence": st.column_config.NumberColumn(
                                        "Confidence %",
                                        format="%.1f%%",
                                        help="AI confidence in the analysis"
                                    ),
                                    "sentiment_category": st.column_config.TextColumn(
                                        "Category",
                                        help="Sentiment classification"
                                    ),
                                    "sentiment_analysis": st.column_config.TextColumn(
                                        "Analysis",
                                        width="large",
                                        help="Detailed sentiment explanation"
                                    )
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Download button
                            csv_data = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_data,
                                file_name=f"sentiment_analysis_results_{int(time.time())}.csv",
                                mime="text/csv",
                                help="Download the complete analysis results"
                            )
                        
                        with result_tab2:
                            st.markdown("### 📈 Data Visualizations")
                            
                            if 'sentiment_category' in results_df.columns and 'sentiment_score' in results_df.columns:
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Sentiment distribution pie chart
                                    sentiment_counts = results_df['sentiment_category'].value_counts()
                                    
                                    fig_pie = px.pie(
                                        values=sentiment_counts.values,
                                        names=sentiment_counts.index,
                                        title="📊 Sentiment Distribution",
                                        color_discrete_map={
                                            'Very Positive': '#2e7d32',
                                            'Positive': '#4caf50',
                                            'Neutral': '#ff9800',
                                            'Negative': '#ff5722',
                                            'Very Negative': '#d32f2f'
                                        }
                                    )
                                    fig_pie.update_traces(
                                        textposition='inside', 
                                        textinfo='percent+label',
                                        textfont_size=12
                                    )
                                    fig_pie.update_layout(height=400)
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                with col2:
                                    # Sentiment score histogram
                                    fig_hist = px.histogram(
                                        results_df,
                                        x='sentiment_score',
                                        nbins=20,
                                        title="📈 Sentiment Score Distribution",
                                        color_discrete_sequence=['#667eea']
                                    )
                                    fig_hist.add_vline(
                                        x=0, 
                                        line_width=3, 
                                        line_dash="dash", 
                                        line_color="red",
                                        annotation_text="Neutral"
                                    )
                                    fig_hist.update_layout(
                                        xaxis_title="Sentiment Score",
                                        yaxis_title="Frequency",
                                        height=400
                                    )
                                    st.plotly_chart(fig_hist, use_container_width=True)
                                
                                # Box plot for sentiment scores by category
                                fig_box = px.box(
                                    results_df,
                                    x='sentiment_category',
                                    y='sentiment_score',
                                    title="📦 Sentiment Score Distribution by Category",
                                    color='sentiment_category',
                                    color_discrete_map={
                                        'Very Positive': '#2e7d32',
                                        'Positive': '#4caf50',
                                        'Neutral': '#ff9800',
                                        'Negative': '#ff5722',
                                        'Very Negative': '#d32f2f'
                                    }
                                )
                                fig_box.update_layout(
                                    xaxis_title="Sentiment Category",
                                    yaxis_title="Sentiment Score",
                                    height=400,
                                    showlegend=False
                                )
                                st.plotly_chart(fig_box, use_container_width=True)
                                
                                # Confidence vs Score scatter plot
                                if 'confidence' in results_df.columns:
                                    fig_scatter = px.scatter(
                                        results_df,
                                        x='sentiment_score',
                                        y='confidence',
                                        color='sentiment_category',
                                        title="🎯 Confidence vs Sentiment Score",
                                        color_discrete_map={
                                            'Very Positive': '#2e7d32',
                                            'Positive': '#4caf50',
                                            'Neutral': '#ff9800',
                                            'Negative': '#ff5722',
                                            'Very Negative': '#d32f2f'
                                        },
                                        hover_data=[selected_column] if selected_column in results_df.columns else None
                                    )
                                    fig_scatter.update_layout(
                                        xaxis_title="Sentiment Score",
                                        yaxis_title="Confidence",
                                        height=400
                                    )
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        with result_tab3:
                            st.markdown("### 📊 Summary Statistics")
                            
                            # Overall statistics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                avg_score = results_df['sentiment_score'].mean()
                                st.metric(
                                    "📊 Average Score", 
                                    f"{avg_score:.3f}",
                                    delta=f"{avg_score:.3f}" if avg_score != 0 else None
                                )
                            
                            with col2:
                                most_common = results_df['sentiment_category'].mode()[0]
                                most_common_count = results_df['sentiment_category'].value_counts().iloc[0]
                                st.metric(
                                    "🏆 Most Common", 
                                    most_common,
                                    delta=f"{most_common_count} entries"
                                )
                            
                            with col3:
                                if 'confidence' in results_df.columns:
                                    avg_confidence = results_df['confidence'].mean()
                                    st.metric(
                                        "🎯 Avg Confidence", 
                                        f"{avg_confidence:.1%}",
                                        delta=f"{avg_confidence:.1%}" if avg_confidence != 0 else None
                                    )
                            
                            with col4:
                                score_std = results_df['sentiment_score'].std()
                                st.metric(
                                    "📏 Score Spread", 
                                    f"{score_std:.3f}",
                                    help="Standard deviation of sentiment scores"
                                )
                            
                            # Detailed statistics table
                            st.markdown("#### 📈 Detailed Statistics")
                            
                            stats_data = []
                            for category in results_df['sentiment_category'].unique():
                                category_data = results_df[results_df['sentiment_category'] == category]
                                stats_data.append({
                                    'Category': category,
                                    'Count': len(category_data),
                                    'Percentage': f"{len(category_data)/len(results_df)*100:.1f}%",
                                    'Avg Score': f"{category_data['sentiment_score'].mean():.3f}",
                                    'Score Range': f"{category_data['sentiment_score'].min():.3f} to {category_data['sentiment_score'].max():.3f}",
                                    'Avg Confidence': f"{category_data['confidence'].mean():.1%}" if 'confidence' in results_df.columns else "N/A"
                                })
                            
                            stats_df = pd.DataFrame(stats_data)
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)
                            
                            # Export statistics
                            stats_csv = stats_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Statistics",
                                data=stats_csv,
                                file_name=f"sentiment_statistics_{int(time.time())}.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"❌ Error processing CSV file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 50px;">
        <p>🎭 <strong>Advanced Sentiment Analyzer</strong> | Powered by Google Gemini AI</p>
        <p>Built with ❤️ using Streamlit • For educational and research purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()