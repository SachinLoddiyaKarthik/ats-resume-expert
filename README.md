# 🧠 Enhanced ATS Resume Expert – Gemini AI + Streamlit + NLP

An **AI-powered resume analyzer** that blends Google Gemini, NLP, and beautiful Streamlit charts — designed to help you **beat the bots** (ATS) and land your dream job faster! 🎯

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-blueviolet?style=for-the-badge&logo=google)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🔗 Live Demo

Try it out now 👉 [ats-resume-expert.streamlit.app](https://ats-resume-expert-vaz74q2toftmzv2fykb7wh.streamlit.app/)

No install needed. Just upload your resume and JD to see the magic.

---

## 🎯 What’s This?

A smart resume matching tool that:
- 📄 Analyzes your resume vs. a job description
- 🧠 Uses Google Gemini for deep insight
- 🔍 Detects **semantic similarity** and **keyword overlap**
- 🎯 Highlights **skills gaps** and improvement tips
- 📊 Visualizes results with beautiful charts
- 🚀 Exports full reports in JSON or TXT

---

## 🚀 Features

- 🤖 Gemini AI-driven content scoring and suggestions
- 📎 PDF Resume parsing with page-wise context
- 📈 Cosine similarity + TF-IDF for deep text comparison
- 📊 Interactive charts (Plotly) for keyword & skills analysis
- 💾 One-click export of reports and insights
- 🧪 Optimized for ATS compatibility and human recruiters

---

## 🏗️ Architecture

```

Resume PDF + Job Description
↓
Text Parsing
↓
+---------------------+
|  Gemini AI Analysis |
+---------------------+
↓
Semantic Matching + Skills Gap
↓
Visualization + Download

````

---

## 🛠 Setup

### 1. Clone this repo

```bash
git clone https://github.com/SachinLoddiyaKarthik/ats-resume-expert
cd ats-resume-expert
````

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

## 📸 Screenshot

> A sleek Streamlit dashboard where you paste a job description, upload your resume, and get an AI-driven report with charts, scores, and suggestions.
> Because your resume deserves more than just a keyword scan. 💼✨

---

## 🧰 Tech Stack

| Tech                 | Role                         |
| -------------------- | ---------------------------- |
| Streamlit            | Web App Interface            |
| Google Generative AI | Resume Analysis (LLM)        |
| PyMuPDF              | Resume Text Extraction       |
| scikit-learn         | TF-IDF + Cosine Similarity   |
| Plotly               | Charts + Data Visualizations |
| python-dotenv        | API Key Security             |
| pandas               | Data Handling & Export       |

---

## ✨ Sample Output

```
✅ Match Score: 78.5%
🎯 Matched Keywords: data engineer, azure, etl
⚠️ Missing Skills: airflow, sql optimization
🚀 Recommendations:
 - Add quantifiable achievements
 - Mention relevant cloud tools
 - Improve formatting for ATS
```

---

## 💡 Use Cases

* Apply to roles with **optimized resumes**
* Know **what keywords to add/remove**
* Prioritize **high-impact improvements**
* Export reports for job coaching or feedback

---

## 👨‍💻 Author

**Sachin Loddiya Karthik**
🔗 [GitHub](https://github.com/SachinLoddiyaKarthik) • [LinkedIn](https://www.linkedin.com/in/sachin-lk/)

---

## ⭐ Like It?

Give the repo a ⭐ if you found it helpful.
Let’s help more people build **resumes that pass the bots**!
