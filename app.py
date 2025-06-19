import streamlit as st
from dotenv import load_dotenv
import os
import io
import json
import re
import fitz  # PyMuPDF
import logging
import base64
from datetime import datetime, timedelta
from PIL import Image
from collections import Counter
import google.generativeai as genai
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from docx import Document
from tenacity import retry, wait_exponential, stop_after_attempt
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
import sqlite3
import hashlib

# Set Streamlit page configuration FIRST
st.set_page_config(
    page_title="Enhanced ATS Resume Expert Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

theme_type = st.context.theme.type  # returns "light" or "dark"
text_color = "#212529" if theme_type == "light" else "white"
text_color = "white" if theme_type == "dark" else "black"

# Ensure load_dotenv is called after imports
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Gemini API configuration
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    if not os.getenv("GOOGLE_API_KEY"):
        # For production, consider a more generic error message
        st.error("❌ Google API key not found in .env file. Please set the GOOGLE_API_KEY environment variable.")
        st.stop() # Stop the app if API key is not configured
except Exception as e:
    st.error(f"❌ Google API key configuration failed. Error: {e}")
    st.stop() # Stop the app on configuration error

# Database setup for historical tracking
def init_db():
    """
    Initializes the SQLite database for storing analysis history and benchmarks.
    Adds 'overall_score' column if it doesn't exist to prevent KeyError.
    """
    conn = sqlite3.connect('ats_analytics.db')
    cursor = conn.cursor()
    
    # Create analyses table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_hash TEXT,
            timestamp TEXT,
            analysis_type TEXT,
            semantic_score REAL,
            skills_match_percentage REAL,
            resume_length INTEGER,
            jd_length INTEGER,
            industry TEXT,
            analysis_data TEXT,
            overall_score REAL,
            role_level TEXT 
        )
    ''')
    
    # Check if 'overall_score' and 'role_level' columns exist and add them if not (for schema evolution)
    try:
        cursor.execute("SELECT overall_score FROM analyses LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE analyses ADD COLUMN overall_score REAL")
        conn.commit()

    try:
        cursor.execute("SELECT role_level FROM analyses LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE analyses ADD COLUMN role_level TEXT")
        conn.commit()

    # Create benchmarks table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            industry TEXT,
            role_level TEXT,
            avg_semantic_score REAL,
            avg_skills_match REAL,
            sample_size INTEGER,
            last_updated TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on app startup
init_db()

# --- PROMPT DEFINITIONS ---
# These prompts instruct the Gemini model on how to analyze the resume and job description.
COMPREHENSIVE_PROMPT = """
Let's think step by step. As an expert ATS consultant and technical recruiter, provide a comprehensive analysis of this resume against the job description.

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

IMPROVEMENT_PROMPT = """
Let's think step by step. You are a senior technical recruiter specializing in ATS optimization.

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

AI_COACHING_PROMPT = """
Based on the provided analysis of a resume against a job description, act as a career coach.
Provide actionable and encouraging advice to help the user improve their resume for job applications.
Consider the semantic match score, skills match percentage, and identified missing/extra skills.
Suggest next steps for career development and job search strategy beyond just the resume.
"""
# --- END PROMPT DEFINITIONS ---

# Sample Data for quick testing
SAMPLE_JD = """
We are seeking a highly motivated and experienced Software Engineer to join our dynamic team. The ideal candidate will have a strong background in Python and Java development, with expertise in building scalable web applications. Experience with cloud platforms (AWS, Azure, GCP), RESTful APIs, and database technologies (SQL, NoSQL) is essential. Strong problem-solving skills, excellent communication, and the ability to work effectively in an agile environment are required. A Bachelor's degree in Computer Science or a related field is preferred, along with 5+ years of industry experience.
"""

SAMPLE_RESUME = """
John Doe
johndoe@email.com | (123) 456-7890 | LinkedIn: linkedin.com/in/johndoe

Summary:
Highly skilled Software Developer with 4 years of experience in designing, developing, and deploying robust web applications. Proficient in Python, JavaScript, and SQL, with a proven track record of delivering high-quality code in fast-paced agile settings. Seeking to leverage strong problem-solving abilities and technical expertise to contribute to innovative projects.

Experience:
Software Engineer | Tech Solutions Inc. | 2020 - Present
- Developed and maintained backend services using Python and Django, improving API response times by 15%.
- Implemented new features for a customer-facing web platform, leading to a 10% increase in user engagement.
- Collaborated with cross-functional teams to define project requirements and deliver solutions on schedule.
- Managed SQL databases for various applications, ensuring data integrity and optimal performance.

Junior Developer | Web Innovations Co. | 2019 - 2020
- Assisted in the development of front-end components using React and HTML/CSS.
- Conducted code reviews and participated in daily stand-ups.

Skills:
Programming Languages: Python, Java, JavaScript, SQL
Cloud Platforms: AWS, Docker
Tools: Git, Jira
Soft Skills: Agile Methodologies, Problem-Solving, Team Collaboration, Communication

Education:
Bachelor of Science in Computer Science | University of ABC | 2019
"""


# Enhanced Analyzer class with advanced analytics capabilities
class AdvancedATSAnalyzer:
    """
    Handles resume and job description processing, AI analysis, and data visualization.
    For production, this class would ideally be split into multiple modules (e.g., data_extractor.py, llm_analyzer.py, visualization.py).
    """
    def __init__(self):
        """Initializes the Gemini model and loads industry benchmarks."""
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash",
            safety_settings={"HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"}
        )
        self._load_industry_benchmarks()

    def _load_industry_benchmarks(self):
        """Loads industry benchmarks from a database or defaults if none are found."""
        # For this demo, we use hardcoded defaults, now with role levels
        self.industry_benchmarks = {
            'Technology': {
                'Entry-Level': {'semantic': 65.0, 'skills': 70.0},
                'Mid-Level': {'semantic': 75.0, 'skills': 80.0},
                'Senior-Level': {'semantic': 85.0, 'skills': 90.0}
            },
            'Healthcare': {
                'Entry-Level': {'semantic': 60.0, 'skills': 65.0},
                'Mid-Level': {'semantic': 70.0, 'skills': 75.0},
                'Senior-Level': {'semantic': 80.0, 'skills': 85.0}
            },
            'Finance': {
                'Entry-Level': {'semantic': 68.0, 'skills': 72.0},
                'Mid-Level': {'semantic': 78.0, 'skills': 82.0},
                'Senior-Level': {'semantic': 88.0, 'skills': 92.0}
            },
            'Marketing': {
                'Entry-Level': {'semantic': 62.0, 'skills': 68.0},
                'Mid-Level': {'semantic': 72.0, 'skills': 75.0},
                'Senior-Level': {'semantic': 82.0, 'skills': 88.0}
            },
            'Sales': {
                'Entry-Level': {'semantic': 58.0, 'skills': 60.0},
                'Mid-Level': {'semantic': 68.0, 'skills': 70.0},
                'Senior-Level': {'semantic': 78.0, 'skills': 80.0}
            },
            'Engineering': {
                'Entry-Level': {'semantic': 70.0, 'skills': 75.0},
                'Mid-Level': {'semantic': 80.0, 'skills': 85.0},
                'Senior-Level': {'semantic': 90.0, 'skills': 95.0}
            },
            'Education': {
                'Entry-Level': {'semantic': 55.0, 'skills': 60.0},
                'Mid-Level': {'semantic': 65.0, 'skills': 68.0},
                'Senior-Level': {'semantic': 75.0, 'skills': 78.0}
            },
            'General': {
                'Entry-Level': {'semantic': 60.0, 'skills': 65.0},
                'Mid-Level': {'semantic': 70.0, 'skills': 72.0},
                'Senior-Level': {'semantic': 80.0, 'skills': 82.0}
            }
        }


    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3), reraise=True)
    def get_gemini_response(self, input_text: str, resume_content_list: List[Dict], prompt: str) -> str:
        """
        Generates a response from the Gemini model based on job description, resume text, and a prompt.
        Includes a retry mechanism for transient API issues.
        """
        try:
            combined_prompt = f"""
            Job Description:
            {input_text}

            Resume Text:
            {resume_content_list[0]['data']}

            Prompt:
            {prompt}
            """
            response = self.model.generate_content(combined_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            st.error(f"Failed to get AI response. Please check your API key and try again. Error: {e}")
            raise # Re-raise to trigger tenacity retry or final error handling

    def extract_text_from_pdf(self, uploaded_file) -> str:
        """Extracts text content from an uploaded PDF file."""
        try:
            uploaded_file.seek(0)
            pdf_bytes = uploaded_file.read()

            if not pdf_bytes:
                return "Error: Uploaded PDF file is empty."

            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception:
                # Catch general PDF open errors (e.g., malformed PDF)
                return "Error: Could not open PDF. It might be corrupted or not a valid PDF. Please try another file."

            text = ""
            if len(doc) == 0:
                return "Error: PDF contains no pages."

            if len(doc) > 15: # Increased warning limit
                st.warning("⚠️ Your resume is quite long (>15 pages). This might impact analysis speed and token limits.")

            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n{page.get_text()}"
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            # More specific error message for extraction failure
            return f"Error extracting PDF content: {str(e)}. Please ensure the PDF is text-selectable."

    def extract_text_from_docx(self, uploaded_file) -> str:
        """Extracts text content from an uploaded DOCX file."""
        try:
            uploaded_file.seek(0)
            doc = Document(io.BytesIO(uploaded_file.read()))
            text = []
            for para in doc.paragraphs:
                text.append(para.text)
            return "\n".join(text).strip()
        except Exception as e:
            logger.error(f"DOCX extraction error: {e}")
            return f"Error extracting DOCX content: {str(e)}. Please ensure the DOCX file is not corrupted."

    def extract_keywords(self, text: str, min_length: int = 3) -> Dict[str, int]:
        """
        Extracts and counts keywords from a given text, filtering out common stopwords.
        Returns a dictionary of top 20 keywords and their frequencies.
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
        return dict(Counter(filtered).most_common(20))

    def semantic_match_score(self, jd_text: str, resume_text: str) -> Tuple[float, Dict]:
        """
        Calculates the semantic similarity score between job description and resume text
        using Google's embeddings and cosine similarity.
        """
        try:
            if not jd_text or not resume_text:
                return 0.0, {'error': 'Job description or resume text is empty for semantic analysis.'}

            # Generate embeddings for JD and Resume
            # Note: The embedding model has token limits. For very long texts, consider chunking.
            jd_embedding_response = genai.embed_content(model="models/embedding-001", content=jd_text)
            resume_embedding_response = genai.embed_content(model="models/embedding-001", content=resume_text)

            jd_embedding = np.array(jd_embedding_response['embedding'])
            resume_embedding = np.array(resume_embedding_response['embedding'])

            # Reshape for cosine_similarity (expects 2D arrays)
            score = cosine_similarity(jd_embedding.reshape(1, -1), resume_embedding.reshape(1, -1))[0][0]

            # For common terms and keywords, still rely on extract_keywords as embeddings don't directly give terms
            jd_keywords = self.extract_keywords(jd_text)
            resume_keywords = self.extract_keywords(resume_text)

            # To get 'common_terms' we can find intersection of keywords or use other methods if needed
            # For simplicity, we'll reuse the keyword extraction for common terms visualization
            common_terms = []
            for jd_kw, jd_freq in jd_keywords.items():
                if jd_kw in resume_keywords:
                    common_terms.append({
                        'term': jd_kw,
                        'jd_score': jd_freq, # Using frequency as a mock 'score'
                        'resume_score': resume_keywords[jd_kw],
                        'combined_score': jd_freq * resume_keywords[jd_kw] # Mock combined score
                    })
            common_terms.sort(key=lambda x: x['combined_score'], reverse=True)


            return score * 100, {
                'similarity_score': round(score * 100, 2),
                'common_terms': common_terms[:10], # Limit to top 10 for display
                'jd_keywords': jd_keywords,
                'resume_keywords': resume_keywords
            }
        except Exception as e:
            logger.error(f"Semantic analysis error (embeddings): {e}")
            return 0.0, {'error': str(e)}

    def analyze_skills_gap(self, jd_keywords: Dict[str, int], resume_keywords: Dict[str, int]) -> Dict:
        """
        Analyzes the skills gap between job description keywords and resume keywords,
        categorizing them into matched, missing, and extra skills.
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

    def generate_word_cloud(self, text: str, title: str = "") -> go.Figure:
        """Generates a word cloud visualization from the given text using Plotly."""
        try:
            if not text or len(text.strip()) < 10:
                fig = go.Figure()
                fig.add_annotation(text="Insufficient text for word cloud", 
                                 xref="paper", yref="paper", showarrow=False,
                                 font=dict(color="gray", size=14))
                fig.update_layout(title=f"Word Cloud - {title}", template="plotly_dark", height=400)
                return fig
                
            # Create word cloud using matplotlib backend, then convert to PIL Image
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='black', # Use black for dark theme compatibility
                                colormap='viridis', # Good contrast
                                max_words=100,
                                min_font_size=10).generate(text)
            
            # Convert PIL Image to a base64 encoded PNG for Plotly
            img_byte_arr = io.BytesIO()
            Image.fromarray(wordcloud.to_array()).save(img_byte_arr, format='PNG')
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Create plotly figure
            fig = go.Figure()

            fig.add_layout_image(
                dict(
                    source=f"data:image/png;base64,{img_base64}",
                    x=0,
                    y=1,
                    sizex=1,
                    sizey=1,
                    xref="paper",
                    yref="paper",
                    sizing="stretch",
                    layer="below"
                )
            )

            fig.update_layout(
                title=f"Word Cloud - {title}",
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0, 1]),
                template="plotly_dark",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0)
            )

            return fig
        except Exception as e:
            logger.error(f"Word cloud generation error: {e}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error generating word cloud: {str(e)}", 
                             xref="paper", yref="paper", showarrow=False,
                             font=dict(color="red", size=14))
            fig.update_layout(title=f"Word Cloud - {title}", template="plotly_dark", height=400)
            return fig

    def create_skills_radar_chart(self, skills_data: Dict, selected_industry: str, selected_role_level: str) -> go.Figure:
        """
        Creates a radar chart comparing user's skills against industry benchmarks.
        Scores are dynamically generated or mocked based on real data where feasible.
        """
        categories = ['Technical Skills', 'Soft Skills', 'Domain Knowledge', 
                     'Certifications', 'Experience Alignment', 'Education Match']
        
        # Retrieve specific benchmarks for the selected industry and role level
        # Default to 'General' and 'Mid-Level' if specific combo not found
        industry_level_benchmark = self.industry_benchmarks.get(selected_industry, {}).get(selected_role_level, self.industry_benchmarks['General']['Mid-Level'])
        
        # Calculate user scores more dynamically based on skills_data and keyword presence
        # These are still "simulated" scores as detailed NLP for each category is not in scope for this function,
        # but they aim to be more reflective of the input than static mocks.
        
        total_jd_skills = len(skills_data.get('jd_keywords', {}))
        matched_skills_count = len(skills_data.get('matched_skills', []))
        
        # Base score on overall skills match percentage
        base_skill_score = skills_data.get('match_percentage', 0)

        user_scores = [
            min(100, base_skill_score * 0.9 + len([s for s in skills_data.get('matched_skills', []) if any(tech in s.lower() for tech in ['python', 'java', 'sql', 'cloud', 'aws', 'data', 'ml', 'ai', 'devops', 'api'])]) * 3), # Technical
            min(100, base_skill_score * 0.8 + len([s for s in skills_data.get('matched_skills', []) if any(soft in s.lower() for soft in ['leadership', 'communication', 'teamwork', 'problem-solving', 'management', 'collaboration'])]) * 5), # Soft
            min(100, base_skill_score * 0.7 + (matched_skills_count / total_jd_skills * 100 if total_jd_skills else 0) * 0.5), # Domain knowledge (proxy for keyword density)
            min(100, base_skill_score * 0.6 + len([s for s in skills_data.get('matched_skills', []) if any(cert in s.lower() for cert in ['certified', 'certification', 'license'])]) * 10), # Certifications
            min(100, base_skill_score), # Experience Alignment (using overall match as proxy)
            min(100, 75 + (base_skill_score * 0.1)) # Education Match (slight boost based on overall match, mostly static mock)
        ]
        
        # Benchmark scores will be more specific now, derived from the selected industry/role benchmarks
        # For radar chart, we need a benchmark for each category. We will proportionally scale the overall
        # semantic/skills benchmark from the industry_level_benchmark to mock individual category benchmarks.
        
        # Using the 'semantic' benchmark as a general reference for all categories for simplicity
        benchmark_category_base = industry_level_benchmark.get('semantic', 70) 
        
        # Distribute the benchmark across categories with slight variations; could be refined with more specific benchmarks
        benchmark_scores = [
            min(100, benchmark_category_base * 1.05), # Technical (slightly higher expected)
            min(100, benchmark_category_base * 0.95), # Soft (slightly lower expected)
            min(100, benchmark_category_base * 1.0),  # Domain Knowledge
            min(100, benchmark_category_base * 0.8),  # Certifications (lower default expectation as not all roles require many)
            min(100, benchmark_category_base * 1.0),  # Experience Alignment
            min(100, benchmark_category_base * 0.9)   # Education Match
        ]

        # Ensure scores are within [0, 100]
        user_scores = [max(0, min(100, s)) for s in user_scores]
        benchmark_scores = [max(0, min(100, s)) for s in benchmark_scores]

        # Extend lists to close the radar chart loop
        categories.append(categories[0])
        user_scores.append(user_scores[0])
        benchmark_scores.append(benchmark_scores[0])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=user_scores,
            theta=categories,
            fill='toself',
            name='Your Resume',
            line_color='rgb(255, 102, 102)', # Reddish
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=benchmark_scores,
            theta=categories,
            fill='toself',
            name='Industry Benchmark',
            line_color='rgb(102, 204, 255)', # Bluish
            opacity=0.5
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100]
                )),
            showlegend=True,
            title="Skills Radar Chart: Your Profile vs. Industry Benchmarks",
            template="plotly_dark", # Consistent dark theme
            height=500
        )
        
        return fig

    def create_timeline_visualization(self, historical_data: List[Dict]) -> go.Figure:
        """
        Creates a timeline visualization of semantic score and skills match percentage over time.
        """
        if not historical_data:
            fig = go.Figure()
            fig.add_annotation(text="No historical data available for timeline", 
                             xref="paper", yref="paper", showarrow=False,
                             font=dict(color="gray", size=14))
            fig.update_layout(title="Resume Optimization Progress Over Time", template="plotly_dark", height=400)
            return fig
            
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp') # Ensure chronological order
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['semantic_score'], 
                      mode='lines+markers', name='Semantic Score', line=dict(color='#88CCEE')), # Light Blue
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['skills_match_percentage'], 
                      mode='lines+markers', name='Skills Match %', line=dict(color='#CC6677')), # Light Red
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Time of Analysis")
        fig.update_yaxes(title_text="Semantic Score (%)", secondary_y=False, range=[0, 100])
        fig.update_yaxes(title_text="Skills Match (%)", secondary_y=True, range=[0, 100])
        
        fig.update_layout(
            title="Resume Optimization Progress Over Time",
            template="plotly_dark", # Consistent dark theme
            hovermode="x unified", # Tooltip shows all traces for a given x
            height=400
        )
        
        return fig

    def create_section_heatmap(self, sections_analysis: Dict) -> go.Figure:
        """
        Creates a heatmap for section-wise optimization opportunities.
        (Using mock data as real-time section-level analysis from LLM requires more complex parsing).
        """
        sections = ['Professional Summary', 'Work Experience', 'Skills', 
                   'Education', 'Certifications', 'Projects']
        metrics = ['Keyword Density', 'ATS Compatibility', 'Relevance Score', 'Completeness']
        
        # Mock data for demonstration - in real implementation, this would be extracted from LLM analysis
        # or a dedicated resume parser.
        # Values are percentages from 0-100, higher is better.
        z_data = np.random.randint(40, 100, size=(len(sections), len(metrics)))
        # Introduce some variation to simulate real data, e.g., lower scores for 'completeness' if it's a short resume
        if not sections_analysis: # If no real data provided, use randomized mock
            for i in range(len(sections)):
                for j in range(len(metrics)):
                    if metrics[j] == 'Completeness' and i % 2 == 0: # Simulate some sections being less complete
                        z_data[i][j] = np.random.randint(20, 60)
                    elif metrics[j] == 'Relevance Score' and i % 3 == 0: # Simulate some sections being less relevant
                        z_data[i][j] = np.random.randint(30, 70)
        # If `sections_analysis` had real data, `z_data` would be populated from it.
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=metrics,
            y=sections,
            colorscale='Viridis', # Good for continuous data, green for high, purple for low
            text=z_data,
            texttemplate="%{text}%",
            textfont={"size": 12, "color": text_color},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Section-wise Optimization Heatmap (Illustrative)",
            template="plotly_dark", # Consistent dark theme
            height=400,
            xaxis_nticks=len(metrics),
            yaxis_nticks=len(sections),
            margin=dict(l=100, r=20, t=50, b=20) # Adjust margins for labels
        )
        
        return fig

    def save_analysis_history(self, user_id: str, analysis_data: Dict):
        """Saves a user's analysis record to the SQLite database."""
        conn = sqlite3.connect('ats_analytics.db')
        cursor = conn.cursor()
        
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO analyses 
            (user_hash, timestamp, analysis_type, semantic_score, skills_match_percentage, 
             resume_length, jd_length, industry, analysis_data, overall_score, role_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_hash,
            datetime.now().isoformat(),
            analysis_data.get('analysis_type', 'comprehensive'),
            analysis_data.get('semantic_score', 0),
            analysis_data.get('skills_match_percentage', 0),
            analysis_data.get('resume_length', 0),
            analysis_data.get('jd_length', 0),
            analysis_data.get('industry', 'General'),
            json.dumps(analysis_data), # Store full analysis data as JSON string
            analysis_data.get('overall_score', 0),
            analysis_data.get('role_level', 'Mid-Level') # Save role_level
        ))
        
        conn.commit()
        conn.close()

    def get_analysis_history(self, user_id: str) -> List[Dict]:
        """Retrieves a user's analysis history from the SQLite database."""
        conn = sqlite3.connect('ats_analytics.db')
        cursor = conn.cursor()
        
        user_hash = hashlib.md5(user_id.encode()).hexdigest()
        cursor.execute('''
            SELECT timestamp, analysis_type, semantic_score, skills_match_percentage, 
                   resume_length, jd_length, industry, overall_score, role_level, analysis_data 
            FROM analyses 
            WHERE user_hash = ? 
            ORDER BY timestamp DESC 
            LIMIT 10
        ''', (user_hash,))
        
        results = cursor.fetchall()
        conn.close()
        
        # Map results to a list of dictionaries for easier handling
        columns = ['timestamp', 'analysis_type', 'semantic_score', 'skills_match_percentage', 
                   'resume_length', 'jd_length', 'industry', 'overall_score', 'role_level', 'analysis_data_json']
        
        history_list = []
        for row in results:
            item = dict(zip(columns, row))
            # Parse the stored JSON string back into a dictionary
            try:
                item['analysis_data'] = json.loads(item.pop('analysis_data_json'))
            except (json.JSONDecodeError, TypeError):
                item['analysis_data'] = {} # Handle potential decoding errors
            history_list.append(item)
        return history_list

    def generate_pdf_report(self, analysis_data: Dict, filename: str = "ats_analysis_report.pdf") -> bytes:
        """Generates a comprehensive PDF report from the analysis data."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Custom style for title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1, # Center align
            textColor=colors.darkblue
        )
        story.append(Paragraph("ATS Resume Analysis Report", title_style))
        story.append(Spacer(1, 0.2 * inch)) # Adjust space

        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['h2']))
        summary_text = f"""
        <b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        <b>Overall ATS Compatibility:</b> {analysis_data.get('overall_score', 0):.1f}%<br/>
        <b>Semantic Match Score:</b> {analysis_data.get('semantic_score', 0):.1f}%<br/>
        <b>Skills Match Percentage:</b> {analysis_data.get('skills_match_percentage', 0):.1f}%
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Key Findings
        story.append(Paragraph("Key Findings", styles['h2']))
        findings = [
            f"Resume contains <b>{analysis_data.get('resume_length', 0)} words</b>",
            f"Job description contains <b>{analysis_data.get('jd_length', 0)} words</b>",
            f"Matched <b>{len(analysis_data.get('matched_skills', []))}</b> key skills",
            f"Missing <b>{len(analysis_data.get('missing_skills', []))}</b> important skills from JD"
        ]
        for finding in findings:
            story.append(Paragraph(f"• {finding}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))
        
        # Recommendations
        story.append(Paragraph("Top Recommendations", styles['h2']))
        # Placeholder recommendations based on common ATS advice; could be from AI_COACHING_PROMPT
        recommendations = [
            "Ensure critical keywords from the job description are integrated naturally throughout your resume, especially in the summary and experience sections.",
            "Quantify your achievements with specific numbers and metrics (e.g., 'Increased sales by 20%', 'Managed projects worth $500K').",
            "Use strong action verbs at the beginning of your bullet points to describe responsibilities and achievements.",
            "Review formatting for consistency (fonts, headings, bullet points) to ensure optimal ATS parsing.",
            "Tailor your professional summary to directly address the key requirements and company values mentioned in the job description."
        ]
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        # Add AI detailed response if available
        if 'ai_response' in analysis_data.get('analysis_data', {}):
            story.append(Paragraph("AI Detailed Analysis and Coaching", styles['h2']))
            # Replace markdown with basic HTML for ReportLab compatibility
            ai_content = analysis_data['analysis_data']['ai_response']
            # Simple markdown to HTML conversion for ReportLab Paragraph
            ai_content = ai_content.replace('**', '<b>').replace('##', '<b>').replace('* ', '• ').replace('\n', '<br/>')
            story.append(Paragraph(ai_content, styles['Normal']))
            story.append(Spacer(1, 0.2 * inch))

        # Build PDF
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error building PDF: {e}")
            raise # Re-raise for Streamlit to catch

# Initialize session states
# `user_id` is generated once per session to track history
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Initialize other session states
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "extracted_resume_text" not in st.session_state:
    st.session_state.extracted_resume_text = ""
if "last_analysis_score" not in st.session_state:
    st.session_state.last_analysis_score = None
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = [] # This will be populated from DB
# Initialize export_results and show_detailed_keywords in session_state
if "export_results" not in st.session_state:
    st.session_state.export_results = True # Default value
if "show_detailed_keywords" not in st.session_state: # Initialize show_detailed_keywords
    st.session_state.show_detailed_keywords = True # Default value

# New session state for onboarding
if "onboarding_complete" not in st.session_state:
    st.session_state.onboarding_complete = False

# Cache the analyzer instance
@st.cache_resource
def get_analyzer():
    """Caches the AdvancedATSAnalyzer instance to avoid re-initializing the Gemini model and benchmarks."""
    return AdvancedATSAnalyzer()

analyzer = get_analyzer()

# Load initial analysis history (or refresh on rerun)
st.session_state.analysis_history = analyzer.get_analysis_history(st.session_state.user_id)

# --- Custom CSS for enhanced UI ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            background-attachment: fixed;
            font-family: 'Inter', sans-serif;
            color: text_color;; /* Default text color */
        }
        
        .stApp > header {
            background: rgba(0,0,0,0); /* Transparent header */
        }

        .glass-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        .metric-card-advanced {
            background: linear-gradient(45deg, rgba(255, 107, 107, 0.8), rgba(78, 205, 196, 0.8));
            backdrop-filter: blur(10px);
            color: text_color;;
            width: 180px;
            height: 130px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 0.4rem;
            transition: transform 0.3s ease;
        }

        .metric-card-advanced .animated-counter {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            line-height: 1;
        }

        .metric-card-advanced p {
            font-size: 0.9rem;
            font-weight: 500;
            margin: 0;
        }
            
        .metric-icon {
        font-size: 1.8rem;
        line-height: 1;
        }

        
        .metric-card-advanced:hover {
            transform: translateY(-5px);
        }
        
        .dashboard-header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: text_color;;
            font-weight: 600;
            border: none;
            border-radius: 25px;
            padding: 0.7rem 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box_shadow: 0 6px 20px rgba(0,0,0,0.3);
        }
        
        .animated-counter {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        /* Adjust Streamlit specific elements for dark background */
        textarea, input[type="text"] {
            background-color: rgba(255, 255, 255, 0.1);
            color: text_color;;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 5px;
        }
        textarea:focus, input[type="text"]:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
        }
        .stFileUploader label, .stSelectbox label, .stCheckbox label, .stSlider label {
            color: text_color;;
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
            font-weight: bold;
            color: text_color;; /* Ensure tab titles are visible */
        }
        .stSuccess, .stWarning, .stInfo, .stError {
            background-color: rgba(255, 255, 255, 0.1);
            color: text_color;;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        .stSuccess { border-left: 5px solid #28a745; }
        .stWarning { border-left: 5px solid #ffc107; }
        .stInfo { border-left: 5px solid #17a2b8; }
        .stError { border-left: 5px solid #dc3545; }
        /* Dataframe styling for dark theme */
        .stDataFrame {
            color: text_color;;
        }
        .stDataFrame thead th {
            color: text_color;;
            background-color: rgba(255, 255, 255, 0.1);
        }
        .stDataFrame tbody tr {
            background-color: rgba(255, 255, 255, 0.05);
        }
        .stDataFrame tbody tr:nth-child(odd) {
            background-color: rgba(255, 255, 255, 0.02);
        }
        .stDataFrame tbody td {
            color: text_color;;
        }
        
        /* General styling for header text (defaults to dark theme) */
        .dashboard-header h1 {
            color: text_color;;
        }
        .dashboard-header p {
            color: rgba(255,255,255,0.9);
        }

        /* Fix header visibility in light theme */
        html[theme="light"] .dashboard-header h1 {
            color: #212529 !important;
        }
        html[theme="light"] .dashboard-header p {
            color: #495057 !important;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 2rem 0;
            color: text_color;; /* Default color for dark theme */
        }

        html[theme="light"] .footer {
            color: #343a40; /* Color for light theme */
        }

    </style>
""", unsafe_allow_html=True)


# --- Dashboard Header ---
st.markdown(f"""
    <div class="dashboard-header">
        <h1 style='font-size: 3.5rem; margin-bottom: 0; font-weight: 700; color: {text_color};'>
            🚀 ATS Resume Expert Pro
        </h1>
        <p style='font-size: 1.3rem; margin-top: 0.5rem; color: {text_color};'>
            Advanced AI-Powered Resume Analytics & Optimization Platform
        </p>
    </div>
""", unsafe_allow_html=True)

# --- Real-time Dashboard Metrics ---
# Only display metrics if an analysis has been run at least once
if st.session_state.last_analysis_score is not None:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card-advanced">
            <div class="animated-counter">{st.session_state.last_analysis_score:.0f}%</div>
            <p title="The Overall Score is an average of your Semantic Match Score and Skills Match Percentage, indicating overall ATS compatibility.">Overall Score</p>
        </div>
        """, unsafe_allow_html=True)

    
    with col2:
        analyses_count = len(st.session_state.analysis_history)
        st.markdown(f"""
            <div class="metric-card-advanced">
                <div class="animated-counter">{analyses_count}</div>
                <p style="margin: 0; font-size: 0.9rem;">Analyses Run</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        improvement = 0.0
        if analyses_count >= 2:
            # History is ordered DESC, so history[0] is current, history[1] is previous
            prev_analysis = st.session_state.analysis_history[1] 
            prev_overall_score = prev_analysis.get('overall_score', 0)
            improvement = st.session_state.last_analysis_score - prev_overall_score
            
        st.markdown(f"""
            <div class="metric-card-advanced">
                <div class="animated-counter">{"+" if improvement >= 0 else ""}{improvement:.1f}%</div>
                <p style="margin: 0; font-size: 0.9rem;">Improvement</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        benchmark_status_text = "N/A"
        if st.session_state.last_analysis_score is not None:
            if st.session_state.last_analysis_score > 75:
                benchmark_status_text = "High Performance"
            elif st.session_state.last_analysis_score > 60:
                benchmark_status_text = "Medium Performance"
            else:
                benchmark_status_text = "Low Performance"

        st.markdown(f"""
            <div class="metric-card-advanced">
                <div style="font-size: 1.5rem;">🎯</div>
                <p style="margin: 0; font-size: 0.9rem;">{benchmark_status_text}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
            <div class="metric-card-advanced">
                <div style="font-size: 1.5rem;">📊</div>
                <p style="margin: 0; font-size: 0.9rem;">Live Analytics</p>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("Run an analysis to see your real-time metrics here!")

# Add the Resume Score Explanation section here
with st.expander("💡 Understanding Your Resume Score"):
    st.markdown("""
        The **Overall Score** represents your resume's comprehensive compatibility with the job description.
        It is an average of two key metrics:
        
        * **Semantic Match Score:** This measures how semantically similar your resume content is to the job description using advanced AI embeddings. It goes beyond simple keyword matching to understand the underlying meaning and context of the words used. A higher score indicates a stronger conceptual alignment.
        * **Skills Match Percentage:** This quantifies the percentage of key skills identified in the job description that are also present in your resume. It's a direct measure of how well your skill set aligns with the role's requirements.
        
        Both scores are crucial for ATS (Applicant Tracking System) compatibility and for a human recruiter's initial review.
        Aim for a higher score by tailoring your resume to the specific job description.
    """)

# --- Sidebar for Advanced Controls ---
with st.sidebar:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🔧 Advanced Configuration")
    
    # Analysis Settings
    st.subheader("Analysis Settings")
    analysis_depth = st.selectbox(
        "Analysis Depth",
        ["Standard", "Detailed", "Comprehensive", "Executive"],
        index=2, # Default to Comprehensive
        help="Executive level provides C-suite focused insights",
        key="selectbox_analysis_depth_sidebar"
    )
    
    industry_sector = st.selectbox(
        "Industry Sector",
        ["Technology", "Healthcare", "Finance", "Marketing", "Sales", "Engineering", "Education", "General"],
        index=0, # Default to Technology
        help="Select your target industry for benchmark comparisons",
        key="selectbox_industry_sector_sidebar"
    )

    role_level = st.selectbox( # New Role Level selection
        "Role Level",
        ["Entry-Level", "Mid-Level", "Senior-Level"],
        index=1, # Default to Mid-Level
        help="Select the target role level for benchmark comparisons.",
        key="selectbox_role_level_sidebar"
    )
    
    # Visualization Controls
    st.subheader("📊 Visualization Controls")
    show_word_clouds = st.checkbox("Show Word Clouds", value=True, key="checkbox_wordclouds_sidebar")
    show_radar_charts = st.checkbox("Show Skills Radar", value=True, key="checkbox_radarcharts_sidebar")
    show_heatmaps = st.checkbox("Show Section Heatmaps", value=True, key="checkbox_heatmaps_sidebar")
    show_timeline = st.checkbox("Show Progress Timeline", value=True, key="checkbox_timeline_sidebar")
    
    # Advanced Features
    st.subheader("🚀 Advanced Features")
    enable_benchmarks = st.checkbox("Industry Benchmarks", value=True, key="checkbox_benchmarks_sidebar")
    enable_peer_comparison = st.checkbox("Peer Comparison (Mock)", value=False, help="Compare with anonymous peer data", key="checkbox_peer_compare_sidebar")
    enable_ai_coaching = st.checkbox("AI Career Coaching", value=True, key="checkbox_ai_coaching_sidebar")

    # Coaching & Feedback
    st.subheader("💡 Coaching & Feedback")
    enable_feedback_loop = st.checkbox("Anonymous Feedback Loop", value=False, key="enable_feedback_loop_sidebar", help="Help us improve by sharing anonymous usage data.")
    
    # Storing export_results and show_detailed_keywords in session state
    st.session_state.export_results = st.checkbox("Enable Export", value=True, key="checkbox_export_results_sidebar", help="Allows downloading of analysis reports.")
    st.session_state.show_detailed_keywords = st.checkbox("Show Detailed Keywords", value=True, key="checkbox_detailed_keywords_sidebar", help="Display word clouds or frequency lists for extracted keywords.")

    # Placeholder for theme toggle, acknowledging the suggestion
    st.markdown("---")
    st.markdown("Theme:")
    st.write("Current: Dark. For full theme control, use Streamlit's settings (top-right menu).")


    st.markdown("</div>", unsafe_allow_html=True) # Close glass-card


# --- Main Content Area - Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Analysis Dashboard", "📈 Historical Analytics & Reports", "⚙️ Resume Editor"])

with tab1:
    # Onboarding / Welcome message for first time users
    if not st.session_state.onboarding_complete:
        st.info("""
            **Welcome to ATS Resume Expert Pro!**
            To get started, please follow these steps:
            1.  **Upload Your Resume** (PDF or DOCX) on the left.
            2.  **Paste a Job Description** into the text area on the right.
            3.  Click any of the **'Run Analysis' buttons** below to get instant feedback!
            You can also load sample data to quickly explore the app.
        """)
        st.session_state.onboarding_complete = True # Mark onboarding as complete for this session

    # First glass-card for inputs
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("Upload Resume & Enter Job Description")
    st.write("To get started, please upload your resume and paste the job description below.")
    
    col_upload, col_jd = st.columns(2)
    
    with col_upload:
        uploaded_file = st.file_uploader("Upload Your Resume (PDF, DOCX)", type=["pdf", "docx"], key="resume_uploader_main")
        
        # Load Sample Resume button
        if st.button("Load Sample Resume", key="load_sample_resume_btn"):
            st.session_state.extracted_resume_text = SAMPLE_RESUME
            st.success("Sample resume loaded!")
            st.rerun() # Rerun to update the text area immediately


        if uploaded_file:
            file_type = uploaded_file.type
            with st.spinner("Extracting text from resume..."):
                if file_type == "application/pdf":
                    extracted_text = analyzer.extract_text_from_pdf(uploaded_file)
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    extracted_text = analyzer.extract_text_from_docx(uploaded_file)
                else:
                    extracted_text = "Error: Unsupported file type."
                    st.error("Unsupported file type. Please upload a PDF or DOCX file.")

            if "Error" in extracted_text:
                st.error(extracted_text)
                st.session_state.extracted_resume_text = "" # Clear on error
            elif not extracted_text.strip():
                 st.warning("Could not extract any text from the uploaded resume. Please try another file.")
            else:
                st.session_state.extracted_resume_text = extracted_text
                st.success("Resume text extracted successfully!")
                st.info(f"Extracted {len(st.session_state.extracted_resume_text.split())} words from resume.")
                
        else:
            if not st.session_state.extracted_resume_text: # Only show this if no resume is loaded yet (neither uploaded nor sample)
                st.info("No resume uploaded yet. Please upload a PDF or DOCX file to proceed.")
                
            st.session_state.extracted_resume_text = st.session_state.extracted_resume_text # Maintain state if sample loaded


        if st.session_state.extracted_resume_text:
            st.subheader("Extracted Resume Content (Read-only)")
            st.text_area(
                "Extracted Text:",
                value=st.session_state.extracted_resume_text,
                height=200,
                key="extracted_resume_display_tab1",
                disabled=True,
                help="This is the text extracted from your uploaded resume. Edit it in the 'Resume Editor' tab."
            )

    with col_jd:
        st.session_state.input_text = st.text_area(
            "Paste Job Description Here",
            st.session_state.input_text,
            height=350,
            help="Copy and paste the full job description into this box.",
            key="jd_input_text_area_tab1"
        )
        # Load Sample JD button
        if st.button("Load Sample Job Description", key="load_sample_jd_btn"):
            st.session_state.input_text = SAMPLE_JD
            st.success("Sample Job Description loaded!")
            st.rerun() # Rerun to update the text area immediately

        st.info(f"Job Description has {len(st.session_state.input_text.split())} words.")
    
    st.markdown("</div>", unsafe_allow_html=True) # Close first glass-card in tab1
    
    # Second glass-card for analysis controls and results
    st.markdown('<div class="glass-card">', unsafe_allow_html=True) 
    st.subheader("Run Analysis")

    # Check if analysis inputs are available
    analysis_disabled = not (st.session_state.extracted_resume_text and st.session_state.input_text)
    if analysis_disabled:
        st.warning("Please provide both Job Description and upload a Resume to enable analysis.")
    elif len(st.session_state.input_text) < 50:
        st.warning("Job Description is very short. For best results, use a JD with at least 50 words.")
    elif len(st.session_state.extracted_resume_text) < 50:
        st.warning("Extracted Resume text is very short. Please ensure your resume has substantial content.")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        analyze_btn = st.button("🔍 Comprehensive Analysis", disabled=analysis_disabled, key="analyze_btn_tab1")
    with col_btn2:
        improve_btn = st.button("💡 Get Improvements", disabled=analysis_disabled, key="improve_btn_tab1")
    with col_btn3:
        semantic_btn = st.button("🧠 Semantic Analysis", disabled=analysis_disabled, key="semantic_btn_tab1")
    with col_btn4:
        skills_btn = st.button("🎯 Skills Gap Analysis", disabled=analysis_disabled, key="skills_btn_tab1")

    st.markdown("---") 

    # --- Analysis Execution Logic ---
    current_analysis_data = {} # Initialize for potential saving to history

    if analyze_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("🧠 Sending to AI for comprehensive analysis...")
        progress_bar.progress(30)
        try:
            # Step 1: Get AI Comprehensive Response
            ai_comprehensive_response = analyzer.get_gemini_response(
                st.session_state.input_text,
                [{'data': st.session_state.extracted_resume_text}],
                COMPREHENSIVE_PROMPT
            )
            st.session_state.gemini_response = ai_comprehensive_response
            progress_bar.progress(60)

            # Step 2: Perform Semantic and Skills Analysis for structured data
            semantic_score, semantic_details = analyzer.semantic_match_score(
                st.session_state.input_text, st.session_state.extracted_resume_text
            )
            jd_keywords = semantic_details.get('jd_keywords', {})
            resume_keywords = semantic_details.get('resume_keywords', {})
            skills_analysis = analyzer.analyze_skills_gap(jd_keywords, resume_keywords)
            
            # Step 3: Calculate Overall Score
            overall_score = (semantic_score + skills_analysis['match_percentage']) / 2
            st.session_state.last_analysis_score = overall_score # Update dashboard metric
            
            # Store full analysis data for history
            current_analysis_data = {
                'analysis_type': analysis_depth, # From sidebar
                'semantic_score': semantic_score,
                'skills_match_percentage': skills_analysis['match_percentage'],
                'resume_length': len(st.session_state.extracted_resume_text.split()),
                'jd_length': len(st.session_state.input_text.split()),
                'industry': industry_sector, # From sidebar
                'role_level': role_level, # From sidebar
                'matched_skills': skills_analysis['matched_skills'],
                'missing_skills': skills_analysis['missing_skills'],
                'extra_skills': skills_analysis['extra_skills'],
                'common_terms': semantic_details.get('common_terms', []),
                'jd_keywords': jd_keywords,
                'resume_keywords': resume_keywords,
                'overall_score': overall_score,
                'ai_response': ai_comprehensive_response # Store the AI text response
            }
            analyzer.save_analysis_history(st.session_state.user_id, current_analysis_data)
            st.session_state.analysis_history = analyzer.get_analysis_history(st.session_state.user_id) # Refresh history

            progress_bar.progress(100)
            st.success("Comprehensive analysis complete!")
            st.balloons()

            # Display Comprehensive Analysis Report
            st.subheader("📋 Comprehensive Analysis Report")
            st.markdown(st.session_state.gemini_response)

            st.markdown("---") # Separator
            st.subheader("📊 Visualizations for Comprehensive Analysis")

            # Display Word Clouds
            if show_word_clouds:
                st.plotly_chart(analyzer.generate_word_cloud(st.session_state.input_text, "Job Description Keywords"), use_container_width=True)
                st.plotly_chart(analyzer.generate_word_cloud(st.session_state.extracted_resume_text, "Your Resume Keywords"), use_container_width=True)
            
            # Display Skills Radar Chart
            if show_radar_charts and enable_benchmarks:
                # Pass selected industry and role level to the radar chart function
                st.plotly_chart(analyzer.create_skills_radar_chart(skills_analysis, industry_sector, role_level), use_container_width=True)
            elif show_radar_charts and not enable_benchmarks:
                st.info("Enable 'Industry Benchmarks' in the sidebar to see the Skills Radar Chart.")
            
            # Display Section Heatmap
            if show_heatmaps and analysis_depth in ["Detailed", "Comprehensive", "Executive"]:
                st.plotly_chart(analyzer.create_section_heatmap({}), use_container_width=True) # Mock data for now
                st.info("Section heatmap is illustrative. Real data requires deeper resume parsing and LLM structuring.")

            if enable_ai_coaching:
                st.subheader("💡 AI Career Coach Insights")
                with st.spinner("Generating personalized coaching advice..."):
                    ai_coaching_response = analyzer.get_gemini_response(
                        st.session_state.input_text,
                        [{'data': st.session_state.extracted_resume_text}],
                        AI_COACHING_PROMPT
                    )
                    st.markdown(ai_coaching_response)

            if enable_peer_comparison:
                st.subheader("👥 Peer Comparison (Mock Feature)")
                st.info("This feature would compare your resume anonymously to similar profiles in your chosen industry. (Under development)")

        except Exception as e:
            st.error(f"Error during comprehensive analysis: {e}. Please check your inputs and API key.")
        finally:
            progress_bar.empty()
            progress_text.empty()

    elif improve_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("💡 Generating improvement suggestions...")
        progress_bar.progress(50)
        try:
            ai_improvement_response = analyzer.get_gemini_response(
                st.session_state.input_text,
                [{'data': st.session_state.extracted_resume_text}],
                IMPROVEMENT_PROMPT
            )
            st.session_state.gemini_improvement_suggestions = ai_improvement_response

            # Attempt to extract score from AI response for dashboard metric
            match = re.search(r'Overall Match Score.*?(\d+\.?\d*)%', ai_improvement_response)
            if match:
                st.session_state.last_analysis_score = float(match.group(1))
            else:
                # If score not found, use a fallback or previous score
                st.session_state.last_analysis_score = st.session_state.get('last_analysis_score', 0)

            # Save basic record to history for tracking
            current_analysis_data = {
                'analysis_type': 'Improvements',
                'semantic_score': st.session_state.get('semantic_score', 0), # Use last semantic if available
                'skills_match_percentage': st.session_state.get('skills_gap_data', {}).get('match_percentage', 0), # Use last skills match if available
                'resume_length': len(st.session_state.extracted_resume_text.split()),
                'jd_length': len(st.session_state.input_text.split()),
                'industry': industry_sector,
                'role_level': role_level, # From sidebar
                'overall_score': st.session_state.last_analysis_score, # Use extracted/fallback score
                'ai_response': ai_improvement_response # Store the AI text response
            }
            analyzer.save_analysis_history(st.session_state.user_id, current_analysis_data)
            st.session_state.analysis_history = analyzer.get_analysis_history(st.session_state.user_id) # Refresh history

            progress_bar.progress(100)
            st.success("Improvement suggestions generated!")

            st.subheader("🚀 Resume Improvement Suggestions")
            st.markdown(st.session_state.gemini_improvement_suggestions)
            
            if enable_ai_coaching:
                st.subheader("💡 AI Career Coach Insights (for Improvements)")
                with st.spinner("Generating additional coaching advice..."):
                    ai_coaching_response = analyzer.get_gemini_response(
                        st.session_state.input_text,
                        [{'data': st.session_state.extracted_resume_text}],
                        AI_COACHING_PROMPT # Use general coaching prompt
                    )
                    st.markdown(ai_coaching_response)

            if st.session_state.export_results: # Use session state variable
                st.download_button(
                    "📥 Download Improvement Plan (TXT)",
                    data=ai_improvement_response,
                    file_name=f"improvement_suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_improvement_txt"
                )
        except Exception as e:
            st.error(f"Error generating improvement suggestions: {e}. Please check your inputs and API key.")
        finally:
            progress_bar.empty()
            progress_text.empty()

    elif semantic_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("🧠 Calculating semantic similarity...")
        progress_bar.progress(50)
        try:
            semantic_score, semantic_details = analyzer.semantic_match_score(
                st.session_state.input_text, st.session_state.extracted_resume_text
            )
            st.session_state.semantic_score = semantic_score
            st.session_state.semantic_details = semantic_details
            
            # For history, we need skills_match_percentage too
            jd_keywords_for_skills = semantic_details.get('jd_keywords', {})
            resume_keywords_for_skills = semantic_details.get('resume_keywords', {})
            skills_analysis_for_history = analyzer.analyze_skills_gap(jd_keywords_for_skills, resume_keywords_for_skills)
            
            overall_score_for_history = (semantic_score + skills_analysis_for_history['match_percentage']) / 2
            st.session_state.last_analysis_score = overall_score_for_history # Update dashboard metric

            # Save to history
            current_analysis_data = {
                'analysis_type': 'Semantic',
                'semantic_score': semantic_score,
                'skills_match_percentage': skills_analysis_for_history['match_percentage'],
                'resume_length': len(st.session_state.extracted_resume_text.split()),
                'jd_length': len(st.session_state.input_text.split()),
                'industry': industry_sector,
                'role_level': role_level, # From sidebar
                'overall_score': overall_score_for_history,
                'semantic_details': semantic_details # Store specific details
            }
            analyzer.save_analysis_history(st.session_state.user_id, current_analysis_data)
            st.session_state.analysis_history = analyzer.get_analysis_history(st.session_state.user_id) # Refresh history

            progress_bar.progress(100)
            st.success("Semantic analysis complete!")

            st.subheader("🤖 Semantic Match Score")
            score_color = "green" if semantic_score >= 70 else "orange" if semantic_score >= 50 else "red"
            st.markdown(f"**Your resume has a <span style='color:{score_color}; font-size:2em;'>{semantic_score:.1f}%</span> semantic match with the Job Description.**", unsafe_allow_html=True)
            
            if show_word_clouds:
                st.markdown("---")
                st.subheader("Keyword Word Clouds")
                st.plotly_chart(analyzer.generate_word_cloud(st.session_state.input_text, "Job Description Keywords"), use_container_width=True)
                st.plotly_chart(analyzer.generate_word_cloud(st.session_state.extracted_resume_text, "Your Resume Keywords"), use_container_width=True)

            if st.session_state.show_detailed_keywords and semantic_details.get('common_terms'): # Use session state variable
                st.markdown("---")
                st.subheader("Top Common Terms & Their Relevance")
                df_common = pd.DataFrame(semantic_details['common_terms'])
                st.dataframe(df_common[['term', 'jd_score', 'resume_score', 'combined_score']].head(10).style.format({'jd_score': '{:.2f}', 'resume_score': '{:.2f}', 'combined_score': '{:.2f}'}), use_container_width=True)
                
            if st.session_state.export_results: # Use session state variable
                semantic_report = {
                    'analysis_type': 'semantic',
                    'semantic_score': semantic_score,
                    'semantic_details': semantic_details,
                    'jd_length': len(st.session_state.input_text.split()),
                    'resume_length': len(st.session_state.extracted_resume_text.split()),
                }
                st.download_button(
                    "📥 Download Semantic Analysis (JSON)",
                    data=json.dumps(semantic_report, indent=2),
                    file_name=f"semantic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_semantic_json"
                )
        except Exception as e:
            st.error(f"Error during semantic analysis: {e}. Please check your inputs.")
        finally:
            progress_bar.empty()
            progress_text.empty()

    elif skills_btn:
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        progress_text.text("🎯 Performing skills gap analysis...")
        progress_bar.progress(50)
        try:
            jd_keywords = analyzer.extract_keywords(st.session_state.input_text)
            resume_keywords = analyzer.extract_keywords(st.session_state.extracted_resume_text)
            skills_analysis = analyzer.analyze_skills_gap(jd_keywords, resume_keywords)
            st.session_state.skills_gap_data = skills_analysis

            # For history, we need semantic_score too
            semantic_score_for_history, _ = analyzer.semantic_match_score(st.session_state.input_text, st.session_state.extracted_resume_text)
            overall_score_for_history = (semantic_score_for_history + skills_analysis['match_percentage']) / 2
            st.session_state.last_analysis_score = overall_score_for_history # Update dashboard metric

            # Save to history
            current_analysis_data = {
                'analysis_type': 'Skills Gap',
                'semantic_score': semantic_score_for_history,
                'skills_match_percentage': skills_analysis['match_percentage'],
                'resume_length': len(st.session_state.extracted_resume_text.split()),
                'jd_length': len(st.session_state.input_text.split()),
                'industry': industry_sector,
                'role_level': role_level, # From sidebar
                'overall_score': overall_score_for_history,
                'skills_analysis': skills_analysis # Store specific details
            }
            analyzer.save_analysis_history(st.session_state.user_id, current_analysis_data)
            st.session_state.analysis_history = analyzer.get_analysis_history(st.session_state.user_id) # Refresh history

            progress_bar.progress(100)
            st.success("Skills gap analysis complete!")

            st.subheader("🎯 Skills Gap Analysis Results")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f'<div class="metric-card-advanced"><h3>{len(skills_analysis["matched_skills"])}</h3><p>Matched Skills</p></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card-advanced"><h3>{len(skills_analysis["missing_skills"])}</h3><p>Missing Skills</p></div>', unsafe_allow_html=True)
            with col3:
                # Corrected f-string syntax here
                st.markdown(f'<div class="metric-card-advanced"><h3>{len(skills_analysis["extra_skills"])}</h3><p>Extra Skills</p></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card-advanced"><h3>{skills_analysis["match_percentage"]:.1f}%</h3><p>Skills Match</p></div>', unsafe_allow_html=True)

            if show_radar_charts and enable_benchmarks:
                st.markdown("---")
                st.subheader("Skills Radar Chart")
                current_benchmarks = analyzer.industry_benchmarks.get(industry_sector, {}).get(role_level, analyzer.industry_benchmarks['General']['Mid-Level'])
                st.plotly_chart(analyzer.create_skills_radar_chart(skills_analysis, industry_sector, role_level), use_container_width=True)
            elif show_radar_charts and not enable_benchmarks:
                st.info("Enable 'Industry Benchmarks' in the sidebar to see the Skills Radar Chart.")

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
                    st.success("No critical missing skills identified based on extracted keywords.")

            with st.expander("💡 Additional Skills in Resume"):
                if skills_analysis['extra_skills']:
                    st.info("These skills are in your Resume but NOT prominently in the Job Description:")
                    for skill in sorted(skills_analysis['extra_skills']):
                        st.markdown(f"• {skill}")
                else:
                    st.info("No significant additional skills found in your resume not mentioned in the JD.")

            if st.session_state.export_results: # Use session state variable
                skills_report_data = {
                    'analysis_type': 'skills_gap',
                    'skills_analysis': skills_analysis,
                    'jd_keywords_extracted': jd_keywords,
                    'resume_keywords_extracted': resume_keywords
                }
                st.download_button(
                    "📥 Download Skills Analysis (JSON)",
                    data=json.dumps(skills_report_data, indent=2),
                    file_name=f"skills_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_skills_json"
                )
        except Exception as e:
            st.error(f"Error during skills gap analysis: {e}. Please check your inputs.")
        finally:
            progress_bar.empty()
            progress_text.empty()
            
    # Conditional display for initial state of the analysis results area
    if st.session_state.last_analysis_score is None and not (analyze_btn or improve_btn or semantic_btn or skills_btn):
        st.info("Upload a resume and paste a job description above, then click an analysis button to see detailed results here!")

    st.markdown("</div>", unsafe_allow_html=True) # Close glass-card for analysis results


with tab2: # Historical Analytics & Reports Tab
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("📈 Historical Analytics & Reports")
    st.write("Review your past resume analysis performances and generate reports.")
    
    st.session_state.analysis_history = analyzer.get_analysis_history(st.session_state.user_id) # Refresh history
    
    if st.session_state.analysis_history:
        st.subheader("Your Analysis History")
        df_history = pd.DataFrame(st.session_state.analysis_history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(df_history[['timestamp', 'analysis_type', 'semantic_score', 'skills_match_percentage', 'overall_score', 'industry', 'role_level']].head(10), use_container_width=True)
        
        if show_timeline:
            st.subheader("Optimization Progress Over Time")
            st.plotly_chart(analyzer.create_timeline_visualization(st.session_state.analysis_history), use_container_width=True)
            
        st.subheader("Generate Comprehensive PDF Report")
        if st.button("📥 Download Latest Analysis Report (PDF)", key="download_pdf_button"):
            if st.session_state.analysis_history:
                latest_analysis = st.session_state.analysis_history[0] # Get the most recent analysis
                try:
                    pdf_report_bytes = analyzer.generate_pdf_report(latest_analysis)
                    st.download_button(
                        label="Click to Download PDF",
                        data=pdf_report_bytes,
                        file_name="ATS_Analysis_Report.pdf",
                        mime="application/pdf",
                        key="pdf_download_final"
                    )
                    st.success("PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Failed to generate PDF report: {e}. Ensure all data is valid.")
            else:
                st.warning("No analysis data available to generate a report.")
    else:
        st.info("No historical analysis data found. Run an analysis in the 'Analysis Dashboard' tab to see your history here.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3: # Resume Editor Tab
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("⚙️ Resume Editor")
    st.write("Edit your extracted resume content directly here for quick iterations.")
    
    if st.session_state.extracted_resume_text:
        edited_resume_content = st.text_area(
            "Edit Your Resume Content",
            st.session_state.extracted_resume_text,
            height=600,
            help="Make changes to your resume text here. These changes will be used for the next analysis.",
            key="resume_editor_text_area"
        )
        
        if st.button("Update Resume Content", key="update_resume_content_button"):
            st.session_state.extracted_resume_text = edited_resume_content
            st.success("Resume content updated! Navigate back to 'Analysis Dashboard' tab to re-run analysis with the new content.")
            st.session_state.last_analysis_score = None # Reset score when text changes significantly
            st.session_state.analysis_history = [] # Clear history as content changed drastically
            st.rerun() # Rerun to reflect the text change and clear old analysis outputs
        else:
            st.info("Make changes above to update your resume content.")
        
        if st.button("Re-Analyze Edited Resume (Go to Analysis Tab)", key="re_analyze_edited_resume_button"):
            st.info("Please navigate to the 'Analysis Dashboard' tab and click 'Analyze Resume' to re-run with edited content.")

    else:
        st.warning("Please upload a resume in the 'Analysis Dashboard' tab to enable editing.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit and Google Gemini AI</p>
        <p><small>🔒 Your data is processed securely and not stored permanently (unless explicitly saved for history).</small></p>
    </div>
""", unsafe_allow_html=True)
