import streamlit as st
from dotenv import load_dotenv
import os
import io
import json
import re
import fitz  # PyMuPDF
import logging
import base64
from datetime import datetime
from PIL import Image
from collections import Counter
import google.generativeai as genai
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time # For simulating loading progress

# Set Streamlit page configuration FIRST
st.set_page_config(
    page_title="Enhanced ATS Resume Expert",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure load_dotenv is called after imports
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Gemini API config
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file.")
except Exception as e:
    st.error(f"❌ Google API key not configured properly. Please check your .env file. Error: {e}")
    st.stop()

# Analyzer class
class ATSResumeAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash",
            safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
        )

    def get_gemini_response(self, input_text: str, pdf_content: List[Dict], prompt: str) -> str:
        """
        Generates a response from the Gemini model based on job description, resume text, and a prompt.

        Args:
            input_text (str): The job description text.
            pdf_content (List[Dict]): A list containing a dictionary with 'data' (resume text).
            prompt (str): The specific prompt for the Gemini model.

        Returns:
            str: The generated text response from the Gemini model or an error message.
        """
        try:
            combined_prompt = f"""
            Job Description:
            {input_text}

            Resume Text:
            {pdf_content[0]['data']}

            Prompt:
            {prompt}
            """
            response = self.model.generate_content(combined_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response from AI: {str(e)}"

    def extract_text_from_pdf(self, uploaded_file) -> str:
        """
        Extracts text from an uploaded PDF file using PyMuPDF (fitz).

        Args:
            uploaded_file: The Streamlit UploadedFile object.

        Returns:
            str: The extracted text from the PDF, or an error message if extraction fails.
        """
        try:
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()

            if not pdf_bytes:
                return "Error: Uploaded PDF file is empty."

            # Attempt to open as a PDF document
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception:
                return "Error: Could not open PDF. It might be corrupted or not a valid PDF."

            text = ""
            if len(doc) == 0:
                return "Error: PDF contains no pages."

            if len(doc) > 10:
                st.warning("⚠️ Your resume is quite long. Consider trimming to < 10 pages for better analysis speed and to avoid potential token limits.")

            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return f"Error extracting PDF content: {str(e)}"

    def extract_keywords(self, text: str, min_length: int = 3) -> Dict[str, int]:
        """
        Extracts and counts keywords from a given text, filtering out stopwords.

        Args:
            text (str): The input text to extract keywords from.
            min_length (int): The minimum length for a word to be considered a keyword.

        Returns:
            Dict[str, int]: A dictionary of keywords and their frequencies, sorted by frequency.
        """
        tokens = re.findall(r"\b\w+\b", text.lower())
        stopwords = set("""
            a an the and or to with of for in on at by is this that it as from be are was were
            has have had do does did but if then else not can will would should may might must
            shall could about after all also any before between both each either few how many
            more most much now only other same some such than than through under until up upon
            what where which while who why your you me my i he she him her it its they them their
            we us our ourself ourselves himself herself itself themselves from for by with against about
            between into through during before after above below to from up down in out on off over
            under again further then once here there when where why how all any both each few more
            most other some such no nor not only own same so than too very s t can will just don should
            now d ll m o re ve ain aren couldn didn doesn't hadn hasn haven isn mightn mustn needn
            shan shouldn wasn weren won wouldn
        """.split())
        filtered = [w for w in tokens if w not in stopwords and len(w) >= min_length and w.isalpha()]
        return dict(Counter(filtered).most_common(20)) # Limiting to top 20 for chart clarity

    def semantic_match_score(self, jd_text: str, resume_text: str) -> Tuple[float, Dict]:
        """
        Calculates the semantic similarity score between job description and resume text
        using TF-IDF and cosine similarity.

        Args:
            jd_text (str): The job description text.
            resume_text (str): The resume text.

        Returns:
            Tuple[float, Dict]: A tuple containing the similarity score (0-100) and
                                a dictionary of analysis details (common terms, keywords).
        """
        try:
            # Basic cleaning to remove punctuation and convert to lowercase
            jd_clean = re.sub(r'[^\w\s]', ' ', jd_text.lower())
            resume_clean = re.sub(r'[^\w\s]', ' ', resume_text.lower())

            if not jd_clean or not resume_clean:
                return 0.0, {'error': 'Job description or resume text is empty for semantic analysis.'}

            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform([jd_clean, resume_clean])

            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            features = vectorizer.get_feature_names_out()

            jd_scores = tfidf_matrix[0].toarray()[0]
            resume_scores = tfidf_matrix[1].toarray()[0]

            # Find common terms with non-zero scores in both JD and Resume
            common = [
                {
                    'term': features[i],
                    'jd_score': jd_scores[i],
                    'resume_score': resume_scores[i],
                    'combined_score': jd_scores[i] * resume_scores[i] # A simple way to prioritize terms strong in both
                }
                for i in range(len(features)) if jd_scores[i] > 0 and resume_scores[i] > 0
            ]
            common.sort(key=lambda x: x['combined_score'], reverse=True)

            return score * 100, {
                'similarity_score': round(score * 100, 2),
                'common_terms': common[:10], # Top 10 common terms
                'jd_keywords': self.extract_keywords(jd_text),
                'resume_keywords': self.extract_keywords(resume_text)
            }
        except Exception as e:
            logger.error(f"Semantic analysis error: {e}")
            return 0.0, {'error': str(e)}

    def analyze_skills_gap(self, jd_keywords: Dict[str, int], resume_keywords: Dict[str, int]) -> Dict:
        """
        Analyzes the skills gap between job description keywords and resume keywords.

        Args:
            jd_keywords (Dict[str, int]): Keywords extracted from the job description.
            resume_keywords (Dict[str, int]): Keywords extracted from the resume.

        Returns:
            Dict: A dictionary containing matched, missing, and extra skills,
                  and a skill match percentage.
        """
        jd_skills = set(jd_keywords.keys())
        resume_skills = set(resume_keywords.keys())

        matched = list(jd_skills & resume_skills)
        missing = list(jd_skills - resume_skills)
        extra = list(resume_skills - jd_skills)

        match_percentage = len(matched) / len(jd_skills) * 100 if jd_skills else 0
        return {
            'matched_skills': matched,
            'missing_skills': missing,
            'extra_skills': extra,
            'match_percentage': round(match_percentage, 2)
        }

# Initialize session state for input text (more robust initialization)
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            color: white; /* Ensure default text color is white on dark background */
        }
        .stApp > div:first-child {
            background: transparent;
        }
        /* Removed .content-container styles */
        .metric-card {
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem 0;
            height: 100%; /* Ensure cards have consistent height */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .metric-card h3 {
            margin-bottom: 0.2rem;
        }
        .stButton>button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .error-box {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
            font-weight: bold;
        }
        /* Adjustments for better readability on dark background */
        textarea, input[type="text"] {
            background-color: rgba(255, 255, 255, 0.1); /* Slightly transparent input fields */
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
        }
        textarea:focus, input[type="text"]:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        }
        .stFileUploader label {
            color: white;
        }
        .stFileUploader button {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        .stFileUploader button:hover {
            background-color: rgba(255, 255, 255, 0.2);
        }
        .stFileUploader > div > div > div > div > div:nth-child(2) { /* For file name display */
            color: white;
        }
        .stProgress > div > div {
            background-color: #4ECDC4; /* Adjust progress bar color */
        }
        .stProgress > div > div > div {
            background-color: #FF6B6B; /* Adjust progress bar filler color */
        }
        /* Streamlit info, warning, error boxes default style remains, they already have good contrast */
    </style>
""", unsafe_allow_html=True)

# Function to create skills comparison chart
def create_skills_chart(skills_analysis: Dict) -> go.Figure:
    """
    Create interactive skills comparison chart.
    """
    categories = ['Matched Skills', 'Missing Skills', 'Extra Skills']
    values = [
        len(skills_analysis.get('matched_skills', [])),
        len(skills_analysis.get('missing_skills', [])),
        len(skills_analysis.get('extra_skills', []))
    ]
    colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']

    fig = go.Figure(data=[
        go.Bar(x=categories, y=values, marker_color=colors, text=values, textposition='auto')
    ])

    fig.update_layout(
        title="Skills Analysis Overview",
        xaxis_title="Skill Categories",
        yaxis_title="Number of Skills",
        template="plotly_dark", # Changed to dark theme for consistency
        height=400
    )
    return fig

def create_keyword_frequency_chart(keywords: Dict[str, int], title: str) -> go.Figure:
    """
    Create keyword frequency chart.
    Handles empty keyword dictionaries gracefully.
    """
    if not keywords:
        fig = go.Figure()
        fig.add_annotation(
            text="No keywords found to display.",
            xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=f"Top Keywords - {title}",
            xaxis_title="Frequency",
            yaxis_title="Keywords",
            template="plotly_dark", # Changed to dark theme for consistency
            height=400
        )
        return fig

    # Sort keywords by frequency in descending order and take top N for better visualization
    sorted_keywords = sorted(keywords.items(), key=lambda item: item[1], reverse=True)[:10]
    words = [item[0] for item in sorted_keywords]
    frequencies = [item[1] for item in sorted_keywords]

    fig = go.Figure(data=[
        go.Bar(x=frequencies, y=words, orientation='h', marker_color='#3498DB')
    ])

    fig.update_layout(
        title=f"Top Keywords - {title}",
        xaxis_title="Frequency",
        yaxis_title="Keywords",
        template="plotly_dark", # Changed to dark theme for consistency
        height=400,
        yaxis={'categoryorder':'total ascending'} # Ensures highest frequency is at the top
    )
    return fig

def export_analysis_report(analysis_data: Dict, filename: str = "ats_analysis_report.json") -> str:
    """
    Export comprehensive analysis report as a JSON string.
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'ATS Resume Analysis',
        **analysis_data
    }
    return json.dumps(report, indent=2, ensure_ascii=False)

# Initialize the analyzer
@st.cache_resource
def get_analyzer():
    """Caches the ATSResumeAnalyzer instance to avoid re-initializing the Gemini model."""
    return ATSResumeAnalyzer()

analyzer = get_analyzer()

# Header
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: white; font-size: 3rem; margin-bottom: 0;'>🚀 Enhanced ATS Resume Expert</h1>
        <p style='color: white; font-size: 1.2rem; margin-top: 0;'>AI-Powered Resume Analysis & Optimization</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    analysis_depth = st.selectbox(
        "Analysis Depth",
        ["Standard", "Detailed", "Comprehensive"],
        index=1,
        help="Choose the level of detail for the comprehensive analysis. 'Standard' is quicker, 'Comprehensive' is most detailed."
    )

    min_keyword_length = st.slider(
        "Minimum Keyword Length",
        min_value=2,
        max_value=6,
        value=3,
        help="Sets the minimum character length for a word to be considered a keyword during extraction."
    )

    st.header("📊 Analysis Options")
    show_visualizations = st.checkbox("Show Visualizations", value=True, help="Display interactive charts for semantic and skills analysis.")
    show_detailed_keywords = st.checkbox("Show Detailed Keywords", value=True, help="Display word clouds or frequency lists for extracted keywords.")
    export_results = st.checkbox("Enable Export", value=True, help="Allows downloading of analysis reports.")

# Main content layout
# No more `content-container` divs around these sections
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Job Description")
    input_text = st.text_area(
        "Paste the job description here",
        height=200,
        key="input",
        help="Paste the complete job description for better analysis"
    )

    if input_text:
        jd_word_count = len(input_text.split())
        st.info(f"📊 Word count: {jd_word_count}")
    else:
        st.info("👈 Please paste a job description to begin.")

with col2:
    st.subheader("📄 Resume Upload")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)",
        type=["pdf"],
        help="Upload a PDF version of your resume for analysis"
    )

    if uploaded_file:
        st.markdown('<div class="success-box">✅ Resume uploaded successfully!</div>', unsafe_allow_html=True)

        # Show file details
        file_details = {
            "Filename": uploaded_file.name,
            "File size": f"{uploaded_file.size / 1024:.1f} KB",
            "File type": uploaded_file.type
        }

        for key, value in file_details.items():
            st.write(f"**{key}:** {value}")
    else:
        st.info("👆 Please upload your resume (PDF) here.")

st.markdown("---") # Add a separator instead of the container

# Action buttons
if input_text and uploaded_file:
    # Input validation for job description
    if len(input_text) < 100: # Minimum 100 characters for a meaningful JD
        st.markdown('<div class="error-box">🚨 Job Description is too short. Please provide a more detailed job description (minimum 100 characters) for effective analysis.</div>', unsafe_allow_html=True)
        st.stop() # Stop execution if input is too short

    # Removed the content-container div here as well
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        analyze_btn = st.button("🔍 Comprehensive Analysis", use_container_width=True)
    with col2:
        improve_btn = st.button("💡 Get Improvements", use_container_width=True)
    with col3:
        semantic_btn = st.button("🧠 Semantic Analysis", use_container_width=True)
    with col4:
        skills_btn = st.button("🎯 Skills Gap Analysis", use_container_width=True)

    st.markdown("---") # Add a separator

    # Enhanced prompts
    comprehensive_prompt = f"""
    As an expert ATS consultant and technical recruiter, provide a comprehensive analysis of this resume against the job description.

    Analysis depth: {analysis_depth}

    Please structure your response with:

    1. 📊 EXECUTIVE SUMMARY
       - Overall match percentage (X%)
       - Key strengths (top 3)
       - Critical gaps (top 3)

    2. 🎯 KEYWORD OPTIMIZATION
       - Missing critical keywords
       - Keyword density analysis
       - ATS-friendly suggestions

    3. 📋 SECTION-BY-SECTION REVIEW
       - Professional summary effectiveness
       - Skills section optimization
       - Experience section improvements
       - Education and certifications

    4. 🚀 ACTIONABLE RECOMMENDATIONS
       - High-impact changes (implement first)
       - Medium-impact improvements
       - Long-term career development suggestions

    5. 📈 ATS COMPATIBILITY SCORE
       - Formatting score
       - Content relevance score
       - Overall ATS friendliness

    Be specific, actionable, and focus on measurable improvements.
    """

    improvement_prompt = """
    You are a senior technical recruiter specializing in ATS optimization.

    Analyze the resume and provide:

    1. 🎯 IMMEDIATE ACTIONS (implement today):
       - Critical keyword additions
       - Formatting fixes
       - Content restructuring

    2. 💼 CONTENT ENHANCEMENTS:
       - Achievement quantification opportunities
       - Skill gap fills
       - Industry-specific terminology

    3. 📝 OPTIMIZATION STRATEGIES:
       - ATS parsing improvements
       - Human recruiter appeal
       - Interview conversion tactics

    4. 🔢 PRIORITY MATRIX:
       - High impact, low effort changes
       - High impact, high effort changes
       - Quick wins for immediate improvement

    Start with an overall match percentage and prioritize suggestions by impact.
    """

    # Analysis execution
    if analyze_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("🔎 Extracting text from PDF (1/3)...")
        resume_text = analyzer.extract_text_from_pdf(uploaded_file)
        progress_bar.progress(33)
        time.sleep(0.1) # Small delay for UX

        if "Error" not in resume_text:
            pdf_content = [{"mime_type": "text/plain", "data": resume_text}]

            progress_text.text("🧠 Sending to AI for comprehensive analysis (2/3)...")
            response = analyzer.get_gemini_response(input_text, pdf_content, comprehensive_prompt)
            progress_bar.progress(66)
            time.sleep(0.1)

            if "Error" not in response:
                progress_text.text("📊 Generating report and visualizations (3/3)...")
                # Removed content-container div here
                st.subheader("📋 Comprehensive Analysis Report")
                st.markdown(response)
                st.markdown("---") # Separator

                semantic_score = 0.0 # Initialize
                analysis_data = {}

                # Additional metrics
                if show_visualizations:
                    score, analysis_data = analyzer.semantic_match_score(input_text, resume_text)
                    if 'error' not in analysis_data:
                        semantic_score = score
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f'<div class="metric-card"><h3>{semantic_score:.1f}%</h3><p>Semantic Match</p></div>', unsafe_allow_html=True)
                        with col2:
                            word_count = len(resume_text.split())
                            st.markdown(f'<div class="metric-card"><h3>{word_count}</h3><p>Resume Words</p></div>', unsafe_allow_html=True)
                        with col3:
                            # Count pages more accurately with PyMuPDF `len(doc)`
                            try:
                                uploaded_file.seek(0)
                                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                                page_count = len(doc)
                                doc.close()
                            except Exception:
                                page_count = "N/A" # Fallback if PDF read fails again
                            st.markdown(f'<div class="metric-card"><h3>{page_count}</h3><p>PDF Pages</p></div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Semantic analysis error: {analysis_data['error']}")
                else:
                    # Still run semantic analysis to get basic data for export, even if not visualized
                    score, analysis_data = analyzer.semantic_match_score(input_text, resume_text)
                    if 'error' not in analysis_data:
                        semantic_score = score


                if export_results:
                    report_data = {
                        'analysis_type': 'comprehensive',
                        'semantic_score': semantic_score,
                        'response': response,
                        'resume_length': len(resume_text.split()),
                        'jd_length': len(input_text.split()),
                        'semantic_analysis_details': analysis_data if 'error' not in analysis_data else {'status': 'error', 'message': analysis_data['error']}
                    }

                    report_json = export_analysis_report(report_data)
                    st.download_button(
                        "📥 Download Comprehensive Report (JSON)",
                        data=report_json,
                        file_name=f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                progress_bar.progress(100)
                progress_text.text("✅ Analysis complete!")
            else:
                st.error(response) # Display AI error
                progress_bar.empty()
                progress_text.empty()
        else:
            st.markdown(f'<div class="error-box">{resume_text}</div>', unsafe_allow_html=True)
            progress_bar.empty()
            progress_text.empty()

    elif improve_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("🔎 Extracting text from PDF (1/2)...")
        resume_text = analyzer.extract_text_from_pdf(uploaded_file)
        progress_bar.progress(50)
        time.sleep(0.1)

        if "Error" not in resume_text:
            pdf_content = [{"mime_type": "text/plain", "data": resume_text}]

            progress_text.text("💡 Generating improvement suggestions (2/2)...")
            response = analyzer.get_gemini_response(input_text, pdf_content, improvement_prompt)
            progress_bar.progress(100)
            time.sleep(0.1)

            if "Error" not in response:
                # Removed content-container div here
                st.subheader("🚀 Improvement Recommendations")
                st.markdown(response)

                # Extract percentage if available
                match = re.search(r'(\d{1,3})\s*%', response)
                if match:
                    percentage = int(match.group(1))
                    st.subheader(f"🎯 Current ATS Match Score: {percentage}%")

                    # Enhanced progress bar with color coding
                    if percentage >= 80:
                        color = "#28a745"  # Green
                    elif percentage >= 60:
                        color = "#ffc107"  # Yellow
                    else:
                        color = "#dc3545"  # Red

                    st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                            <div style="background-color: {color}; height: 20px; width: {percentage}%; border-radius: 10px; transition: width 0.5s ease;"></div>
                            <p style="text-align: center; margin-top: 0.5rem; font-weight: bold; color: black;">Match Score: {percentage}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("---") # Separator


                if export_results:
                    st.download_button(
                        "📥 Download Improvement Plan (TXT)",
                        data=response,
                        file_name=f"improvement_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                progress_text.text("✅ Improvements generated!")
            else:
                st.error(response) # Display AI error
                progress_bar.empty()
                progress_text.empty()
        else:
            st.markdown(f'<div class="error-box">{resume_text}</div>', unsafe_allow_html=True)
            progress_bar.empty()
            progress_text.empty()


    elif semantic_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("🔎 Extracting text from PDF (1/2)...")
        resume_text = analyzer.extract_text_from_pdf(uploaded_file)
        progress_bar.progress(50)
        time.sleep(0.1)

        if "Error" not in resume_text:
            progress_text.text("🧠 Calculating semantic similarity (2/2)...")
            score, analysis = analyzer.semantic_match_score(input_text, resume_text)
            progress_bar.progress(100)
            time.sleep(0.1)

            # Removed content-container div here
            st.subheader(f"🤖 Semantic Analysis Results")

            if 'error' not in analysis:
                # Main score display
                st.markdown(f"""
                    <div style="text-align: center; background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0;">
                        <h2 style="margin: 0; font-size: 3rem;">{score:.1f}%</h2>
                        <p style="margin: 0; font-size: 1.2rem;">Semantic Similarity Score</p>
                    </div>
                """, unsafe_allow_html=True)

                if show_visualizations and analysis.get('common_terms'):
                    # Common terms visualization
                    terms_df = pd.DataFrame(analysis['common_terms'])
                    fig = px.bar(
                        terms_df,
                        x='combined_score',
                        y='term',
                        orientation='h',
                        title="Top Matching Terms by Combined Relevance",
                        color='combined_score',
                        color_continuous_scale='Viridis',
                        text_auto=True # Show values on bars
                    )
                    fig.update_layout(height=400, template="plotly_dark", yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                elif show_visualizations:
                    st.info("No common terms found for visualization.")

                if show_detailed_keywords:
                    st.markdown("---")
                    st.subheader("📊 Keyword Frequency Breakdown")
                    # Use tabs for better organization
                    tabs = st.tabs(["Job Description Keywords", "Resume Keywords"])

                    with tabs[0]:
                        if analysis.get('jd_keywords'):
                            fig1 = create_keyword_frequency_chart(analysis['jd_keywords'], "Job Description")
                            st.plotly_chart(fig1, use_container_width=True)
                        else:
                            st.info("No keywords extracted from Job Description.")

                    with tabs[1]:
                        if analysis.get('resume_keywords'):
                            fig2 = create_keyword_frequency_chart(analysis['resume_keywords'], "Resume")
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.info("No keywords extracted from Resume.")

                st.markdown("---") # Separator

                if export_results:
                    semantic_report = export_analysis_report({
                        'analysis_type': 'semantic',
                        'semantic_score': score,
                        'semantic_details': analysis
                    })

                    st.download_button(
                        "📥 Download Semantic Analysis (JSON)",
                        data=semantic_report,
                        file_name=f"semantic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                progress_text.text("✅ Semantic analysis complete!")
            else:
                st.error(f"Error during semantic analysis: {analysis['error']}")
                progress_bar.empty()
                progress_text.empty()
        else:
            st.markdown(f'<div class="error-box">{resume_text}</div>', unsafe_allow_html=True)
            progress_bar.empty()
            progress_text.empty()

    elif skills_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)

        progress_text.text("🔎 Extracting text from PDF (1/2)...")
        resume_text = analyzer.extract_text_from_pdf(uploaded_file)
        progress_bar.progress(50)
        time.sleep(0.1)

        if "Error" not in resume_text:
            progress_text.text("🎯 Performing skills gap analysis (2/2)...")
            jd_keywords = analyzer.extract_keywords(input_text, min_keyword_length)
            resume_keywords = analyzer.extract_keywords(resume_text, min_keyword_length)
            skills_analysis = analyzer.analyze_skills_gap(jd_keywords, resume_keywords)
            progress_bar.progress(100)
            time.sleep(0.1)


            # Removed content-container div here
            st.subheader("🎯 Skills Gap Analysis")

            # Skills overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f'<div class="metric-card"><h3>{len(skills_analysis["matched_skills"])}</h3><p>Matched Skills</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><h3>{len(skills_analysis["missing_skills"])}</h3><p>Missing Skills</p></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><h3>{len(skills_analysis["extra_skills"])}</h3><p>Extra Skills</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><h3>{skills_analysis["match_percentage"]:.1f}%</h3><p>Skills Match</p></div>', unsafe_allow_html=True)

            if show_visualizations:
                # Skills comparison chart
                fig = create_skills_chart(skills_analysis)
                st.plotly_chart(fig, use_container_width=True)

            # Detailed skills breakdown using expanders for cleaner UI
            st.markdown("---")
            st.subheader("Detailed Skills Breakdown")
            with st.expander("✅ Matched Skills"):
                if skills_analysis['matched_skills']:
                    for skill in sorted(skills_analysis['matched_skills']):
                        st.markdown(f"• {skill}")
                else:
                    st.info("No matching skills found.")

            with st.expander("⚠️ Missing Skills"):
                if skills_analysis['missing_skills']:
                    st.warning("These skills are in the Job Description but NOT prominently in your Resume:")
                    for skill in sorted(skills_analysis['missing_skills']):
                        st.markdown(f"• {skill}")
                else:
                    st.info("No critical missing skills identified based on extracted keywords.")

            with st.expander("💡 Additional Skills in Resume"):
                if skills_analysis['extra_skills']:
                    st.info("These skills are in your Resume but NOT prominently in the Job Description:")
                    for skill in sorted(skills_analysis['extra_skills']):
                        st.markdown(f"• {skill}")
                else:
                    st.info("No significant additional skills found in your resume not mentioned in the JD.")

            st.markdown("---") # Separator

            if export_results:
                skills_report = export_analysis_report({
                    'analysis_type': 'skills_gap',
                    'skills_analysis': skills_analysis,
                    'jd_keywords_extracted': jd_keywords, # Include keywords used for analysis
                    'resume_keywords_extracted': resume_keywords
                })

                st.download_button(
                    "📥 Download Skills Analysis (JSON)",
                    data=skills_report,
                    file_name=f"skills_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            progress_text.text("✅ Skills analysis complete!")
        else:
            st.markdown(f'<div class="error-box">{resume_text}</div>', unsafe_allow_html=True)
            progress_bar.empty()
            progress_text.empty()

else:
    # Welcome message (removed content-container div here)
    st.markdown("""
        ### 🎯 Welcome to the Enhanced ATS Resume Expert!

        This advanced tool helps you optimize your resume for Applicant Tracking Systems (ATS) and improve your job application success rate.

        **Features:**
        - 🔍 **Comprehensive Analysis**: Deep dive into resume-job description alignment
        - 💡 **Smart Improvements**: AI-powered suggestions for optimization
        - 🧠 **Semantic Analysis**: Advanced text similarity scoring
        - 🎯 **Skills Gap Analysis**: Identify missing and extra skills
        - 📊 **Visual Analytics**: Interactive charts and metrics
        - 📥 **Export Reports**: Download detailed analysis reports

        **Getting Started:**
        1. Paste your target job description in the left text area.
        2. Upload your resume PDF in the right section.
        3. Choose your preferred analysis type below.
        4. Review the insights and recommendations.

        **Pro Tips:**
        - Use the most recent version of your resume.
        - Include the complete job description for better analysis.
        - Try different analysis types for comprehensive insights.
        - Export reports to track your optimization progress.
    """)
    st.markdown("---") # Separator

# Footer
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: white;'>
        <p>Built with ❤️ using Streamlit and Google Gemini AI</p>
        <p><small>🔒 Your data is processed securely and not stored permanently</small></p>
    </div>
""", unsafe_allow_html=True)