# 🎭 Advanced Sentiment Analyzer

A comprehensive sentiment analysis application powered by Google's Gemini AI that provides multi-source sentiment analysis with advanced visualizations and batch processing capabilities.

## ✨ Features

### 🎥 YouTube Video Analysis
- **Multiple Extraction Methods**: Automatically tries different approaches to extract content
  - Official transcript/captions via YouTube Transcript API
  - Alternative extraction via PyTubeFix
  - Fallback web scraping for video descriptions
- **Video Metadata**: Displays title, views, duration, and author information
- **Smart Content Detection**: Handles various video privacy settings and caption availability

### 📝 Direct Text Analysis
- **Real-time Analysis**: Instant sentiment analysis for any text input
- **Text Statistics**: Word count, character count, and length validation
- **Rich Feedback**: Detailed analysis with confidence scores

### 📊 Batch Processing
- **CSV Upload**: Process multiple texts simultaneously
- **Auto Column Detection**: Automatically identifies text columns in your data
- **Smart Sampling**: Processes up to 100 rows to manage API usage
- **Export Results**: Download complete analysis results

### 📈 Advanced Visualizations
- **Interactive Gauges**: Real-time sentiment score visualization
- **Distribution Charts**: Pie charts and histograms for batch analysis
- **Statistical Plots**: Box plots and scatter plots for deeper insights
- **Summary Statistics**: Comprehensive statistical breakdowns

### 🎯 Sentiment Scoring System
- **Range**: -1.0 (Very Negative) to +1.0 (Very Positive)
- **Categories**:
  - Very Positive: 0.6 to 1.0
  - Positive: 0.2 to 0.6
  - Neutral: -0.2 to 0.2
  - Negative: -0.6 to -0.2
  - Very Negative: -1.0 to -0.6
- **Confidence Scoring**: AI confidence levels for each analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Google Gemini API key (free at [Google AI Studio](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone or download the application:**
```bash
git clone <repository-url>
cd advanced-sentiment-analyzer
```

2. **Install required dependencies:**
```bash
pip install -r requirements.txt
```



### Running the Application

1. **Start the Streamlit app:**
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Enter your Gemini API key** in the provided field

4. **Start analyzing!** Choose from YouTube videos, direct text, or CSV batch processing

## 📋 Requirements

### Core Dependencies
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
google-generativeai>=0.3.0
plotly>=5.15.0
```

### Optional Dependencies (for full YouTube functionality)
```
youtube-transcript-api>=0.6.0
pytubefix>=6.0.0
beautifulsoup4>=4.12.0
requests>=2.31.0
```

## 🎯 Usage Guide

### 1. YouTube Video Analysis

**Supported URLs:**
- `https://www.youtube.com/watch?v=VIDEO_ID`
- `https://youtu.be/VIDEO_ID`
- `https://www.youtube.com/shorts/VIDEO_ID`
- `https://www.youtube.com/embed/VIDEO_ID`

**Process:**
1. Paste a YouTube URL
2. Click "Analyze Video"
3. The app tries multiple methods to extract content
4. View sentiment analysis with confidence scores

### 2. Direct Text Analysis

**Guidelines:**
- Minimum 5 characters required
- Maximum 10,000 characters (automatically truncated)
- Supports any language (results may vary)

**Process:**
1. Enter or paste text in the text area
2. Click "Analyze Text Sentiment"
3. View detailed analysis with gauge visualization

### 3. Batch CSV Analysis

**CSV Format Requirements:**
- Must contain at least one text column
- Supported column names: 'text', 'content', 'comment', 'description', 'review', 'feedback', 'message'
- Maximum 100 rows processed per session

**Process:**
1. Upload your CSV file
2. Select the text column to analyze
3. Click "Start Batch Analysis"
4. View results in three tabs: Detailed Results, Visualizations, Summary Statistics
5. Download complete results and statistics

**Sample CSV Structure:**
```csv
text,category
"I love this product! Amazing quality.",Electronics
"Terrible service, very disappointed.",Service
"It's okay, nothing special.",General
```

## 🔧 Configuration

### API Rate Limiting
- **Delay between calls**: 0.5 seconds (configurable via `API_DELAY`)
- **Batch processing limit**: 100 rows (configurable via `MAX_CSV_ROWS`)
- **Text length limits**: 5-10,000 characters

### Customization Options

**Constants in the code you can modify:**
```python
MAX_CSV_ROWS = 100          # Maximum rows for batch processing
MIN_TEXT_LENGTH = 5         # Minimum text length for analysis
MAX_TEXT_LENGTH = 10000     # Maximum text length (truncated if exceeded)
API_DELAY = 0.5            # Delay between API calls (seconds)
```

## 📊 Output Formats

### Individual Analysis Result
```python
{
    'text': 'Original text',
    'category': 'Positive',
    'score': 0.75,
    'analysis': 'Detailed explanation...',
    'confidence': 0.92
}
```

### Batch Analysis CSV Columns
- Original data columns (preserved)
- `sentiment_category`: Classification (Very Positive, Positive, etc.)
- `sentiment_score`: Numerical score (-1.0 to 1.0)
- `sentiment_analysis`: Detailed AI explanation
- `confidence`: AI confidence level (0.0 to 1.0)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional video platforms support
- More visualization types
- Enhanced text preprocessing
- Multi-language optimization
- Performance improvements

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini AI** for powerful sentiment analysis capabilities
- **Streamlit** for the excellent web framework
- **Plotly** for interactive visualizations
- **YouTube Transcript API** and **PyTubeFix** for video content extraction

## 📧 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section above
2. Review the [Google AI Studio documentation](https://ai.google.dev/)
3. Open an issue in the repository

---

**Built with ❤️ for sentiment analysis enthusiasts and researchers**

*Last updated: August 2025*