# 🧠 Enhanced ATS Resume Expert Pro – Gemini AI + Streamlit + NLP

An **AI-powered resume analyzer** built with Google Gemini Pro, NLP, and interactive Streamlit visualizations. Designed to help job seekers optimize their resumes for ATS (Applicant Tracking Systems) and stand out to recruiters. 🌟

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-blueviolet?style=for-the-badge&logo=google)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🔗 Live Demo

Try it out now 👉 [ats-resume-expert.streamlit.app](https://ats-resume-expert-vaz74q2toftmzv2fykb7wh.streamlit.app/)

No install needed. Upload your resume and JD to get real-time AI insights.

---

## 🎯 What’s This?

A smart resume optimization platform that:
- 📄 Analyzes your resume against a job description
- 🧠 Uses Google Gemini Pro for deep LLM insights
- 🔍 Detects **semantic match**, **keyword overlap**, and **skills gap**
- 🎯 Offers AI-driven career coaching & improvement tips
- 📊 Visualizes metrics with **Plotly**, **WordCloud**, and **Radar Charts**
- 🚀 Tracks progress and exports full PDF reports

---

## 🚀 Features

- 🤖 LLM-powered insights via Gemini (comprehensive, coaching, improvement modes)
- 📄 PDF + DOCX resume parsing with full-text editor tab
- 🌐 Semantic similarity scoring using TF-IDF + cosine similarity
- 📊 Interactive visuals: Word clouds, radar charts, heatmaps, timelines
- 📅 SQLite database for historical analysis tracking
- 🔄 Export analysis in TXT, JSON, and downloadable PDF report

---

## 🏐 Architecture

```
Resume PDF / DOCX + Job Description
👇
Text Extraction (PyMuPDF / python-docx)
👇
Gemini Pro Analysis + NLP Matching
👇
Semantic + Skills Gap Evaluation
👇
Visualizations + Career Suggestions
👇
Reports + Historical Tracking (SQLite)
```

---

## 🛠️ Setup

### 1. Clone this repo
```bash
git clone https://github.com/SachinLoddiyaKarthik/ats-resume-expert
cd ats-resume-expert
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your `.env` file
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Launch the app
```bash
streamlit run ats_resume_expert.py
```

---

## 📸 Screenshots

> Modern dashboard with semantic match scoring, keyword analysis, AI suggestions, and resume editor.

---

## 🛠️ Tech Stack

| Tech                 | Purpose                          |
|----------------------|----------------------------------|
| Streamlit            | Web interface                    |
| Google Gemini Pro    | AI prompt-based resume analysis |
| PyMuPDF / python-docx| Resume text extraction           |
| scikit-learn         | TF-IDF + cosine similarity       |
| Plotly + WordCloud   | Interactive visualizations       |
| pandas               | Data wrangling                   |
| reportlab            | PDF export generation            |
| SQLite3              | Historical analytics             |

---

## 📈 Sample Output

```
😂 Overall Score: 84.2%
🌟 Matched Skills: data engineering, python, spark
🚨 Missing: airflow, pipeline orchestration
🚀 Suggestions:
 - Add quantifiable project outcomes
 - Mention CI/CD or orchestration tools
 - Align resume header formatting for ATS
```

---

## 💡 Use Cases

- Tailor your resume for every job
- Optimize for ATS and recruiter readability
- Track resume performance improvements over time
- Export and share insights with career coaches

---

## 👨‍💼 Author

**Sachin Loddiya Karthik**  
[GitHub](https://github.com/SachinLoddiyaKarthik) • [LinkedIn](https://www.linkedin.com/in/sachin-lk/)

---

## ⭐ Like It?

Star the repo ⭐ and share with job seekers who want to stand out and get past the bots! 

**#AI #Gemini #ATS #ResumeOptimizer #Streamlit**
