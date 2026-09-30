# 🎓 AI Question Paper Analyzer

An AI-powered question paper analysis application that combines **Gemini LLM**, **traditional machine learning**, and **statistical analysis** to analyze previous-year question papers (PYQs).

The application accepts PDF, image, or text question papers, extracts individual questions, classifies them using Gemini, groups similar questions using machine learning, performs frequency and Pareto analysis, and generates study recommendations.

---

## 🚀 Features

### 📄 Question Paper Extraction
Supports:

- PDF files
- Images (`.png`, `.jpg`, `.jpeg`)
- Text files (`.txt`)

The application extracts text using:

- **PDFPlumber** for PDFs
- **EasyOCR** for images
- Python file handling for text files

### 🤖 Gemini LLM Analysis

Gemini analyzes each extracted question and identifies:

- Subject
- Unit
- Topic
- Difficulty
- Question type
- AI confidence score

The system uses the supplied syllabus as the classification space rather than relying only on exact keyword matches.

### 🧠 Machine Learning Analysis

The project uses traditional machine-learning techniques:

- **TF-IDF Vectorization** — converts questions into numerical text features
- **K-Means Clustering** — groups similar questions
- **Cosine Similarity** — identifies similarity between questions

### 📊 Statistical Analysis

The application performs:

- Topic frequency analysis
- Subject distribution analysis
- Difficulty distribution
- Question-type distribution
- **Pareto 80/20 analysis**

### 📈 Visualization

Two Matplotlib charts are generated:

1. Topic frequency distribution
2. Pareto analysis with cumulative percentage

### 💡 AI Study Recommendations

Gemini generates study recommendations based on:

- Frequently occurring topics
- Pareto-priority topics
- Difficulty distribution
- Question-type distribution

### 🌐 Gradio Interface

The application provides a browser-based Gradio interface where users can:

1. Upload a question paper
2. Select the file type
3. Run the analysis
4. View the generated report
5. View charts
6. Inspect question-level AI and ML results

---

## 🏗️ System Architecture

```text
                QUESTION PAPER
                       │
                       ▼
             ┌───────────────────┐
             │  File Processing  │
             │ PDF / Image / TXT │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Question          │
             │ Extraction        │
             │ Regex             │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │   Gemini LLM      │
             │                   │
             │ Subject           │
             │ Unit              │
             │ Topic             │
             │ Difficulty        │
             │ Question Type     │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Machine Learning  │
             │                   │
             │ TF-IDF            │
             │ K-Means           │
             │ Cosine Similarity │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ Statistical       │
             │ Analysis          │
             │                   │
             │ Frequency         │
             │ Pareto 80/20      │
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │ AI Recommendations│
             └─────────┬─────────┘
                       │
                       ▼
             ┌───────────────────┐
             │   Gradio UI       │
             │ Report + Charts   │
             └───────────────────┘
```

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Gemini API | LLM-based question classification and recommendations |
| `google-genai` | Gemini Python SDK |
| Scikit-learn | Machine learning |
| TF-IDF | Text feature extraction |
| K-Means | Question clustering |
| Cosine Similarity | Question similarity |
| Pandas | Data analysis |
| Matplotlib | Data visualization |
| Gradio | Web interface |
| PDFPlumber | PDF text extraction |
| EasyOCR | Image text extraction |
| Regex | Question extraction |
| python-dotenv | Local environment variable management |

---

## 📁 Project Structure

```text
AI-Powered-PYQ-Analyzer/
│
├── app.py
├── AI_Question_Paper_Analyzer.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

### Main files

**`app.py`**

The main application containing the complete processing pipeline and Gradio interface.

**`AI_Question_Paper_Analyzer.ipynb`**

Jupyter Notebook version of the application.

**`requirements.txt`**

Contains the Python dependencies required to run the project.

**`.gitignore`**

Prevents sensitive files such as `.env` and generated files from being committed.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the repository URL with your actual GitHub repository.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Gemini API Key

The application requires a Gemini API key.

### Local setup

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

The application reads the key using:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

### ⚠️ Security

**Never commit your `.env` file or expose your Gemini API key in your source code.**

The `.gitignore` file already includes:

```text
.env
```

---

## ▶️ Running Locally

Run:

```bash
python app.py
```

The Gradio application will start locally.

Open the URL displayed in the terminal.

---

## 📓 Running the Notebook

Open:

```text
AI_Question_Paper_Analyzer.ipynb
```

using:

- Jupyter Notebook
- JupyterLab
- VS Code
- Google Colab

Install the dependencies first:

```bash
pip install -r requirements.txt
```

Then run the notebook cells.

---

## ☁️ Deployment on Render

This project can be deployed as a **Render Web Service**.

### Render configuration

**Runtime:**

```text
Python 3
```

**Branch:**

```text
main
```

**Root Directory:**

```text
Leave blank
```

**Build Command:**

```bash
pip install -r requirements.txt
```

**Start Command:**

```bash
python app.py
```

### Environment Variable

In Render, add:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Do not upload your `.env` file.

The application uses Render's `PORT` environment variable and binds the Gradio server to:

```text
0.0.0.0
```

so it can be accessed through the deployed web service.

---

## 🔬 Analysis Pipeline

### Step 1 — File Processing

The uploaded question paper is converted into text.

### Step 2 — Question Extraction

Regular expressions identify numbered questions and question sentences.

### Step 3 — LLM Classification

Gemini semantically analyzes each question.

Example:

```text
Question:
Explain the insertion operation in an AVL tree.

Gemini:
Subject: Data Structures
Unit: Trees
Topic: AVL Tree
Difficulty: Medium
Question Type: Conceptual
Confidence: 0.94
```

### Step 4 — TF-IDF

Questions are transformed into numerical vectors based on their textual features.

### Step 5 — K-Means

Similar questions are grouped into clusters.

### Step 6 — Cosine Similarity

The system calculates similarity between questions to identify related questions.

### Step 7 — Frequency Analysis

Topics are counted to determine how frequently they appear.

### Step 8 — Pareto Analysis

Topics are sorted by frequency and cumulative percentage is calculated.

Topics contributing to the first approximately 80% of question volume are identified as priority topics.

### Step 9 — AI Recommendations

Gemini analyzes the resulting statistics and produces study recommendations.

---

## 📊 Example Output

The application produces:

### Analysis Report

```text
AI QUESTION PAPER ANALYSIS REPORT

SUMMARY
-------
Total Questions       : 30
Unique Topics         : 14
Priority Topics       : 6
Priority Coverage     : 83.3%
Average AI Confidence : 91.4%
ML Clusters           : 5
```

### Question-Level Analysis

| Question | Subject | Unit | Topic | Difficulty | Type | AI Confidence | ML Cluster |
|---|---|---|---|---|---|---:|---:|
| Q1 | Data Structures | Trees | AVL Tree | Medium | Conceptual | 94% | 2 |
| Q2 | Data Structures | Sorting | Merge Sort | Medium | Algorithm | 96% | 1 |

---

## 🧪 Machine Learning Component

The project uses **unsupervised machine learning** because there is no manually labelled training dataset.

### TF-IDF

TF-IDF represents each question numerically based on word importance.

### K-Means

K-Means groups questions with similar textual characteristics.

### Cosine Similarity

Cosine similarity measures the similarity between question vectors.

This allows the system to discover groups of related questions without requiring a pre-trained classification model.

---

## 🤖 Role of the LLM vs ML

The project deliberately uses both approaches for different purposes.

### Gemini LLM

Used for semantic interpretation:

```text
Question
   ↓
Gemini
   ↓
Subject / Unit / Topic / Difficulty / Type
```

### Traditional ML

Used for numerical text analysis:

```text
Questions
   ↓
TF-IDF
   ↓
Numerical vectors
   ↓
K-Means
   ↓
Clusters
```

This combination allows the application to perform both **semantic classification** and **data-driven similarity analysis**.

---

## 🔐 Privacy & API Usage

Question papers are sent to the Gemini API for LLM analysis.

Therefore, users should avoid uploading documents containing sensitive or confidential information unless they are comfortable sending that content to the configured Gemini service.

The application itself does not require a database.

---

## ⚠️ Limitations

- Gemini classification depends on the quality of the supplied syllabus.
- OCR accuracy depends on image quality.
- Regex-based extraction may not correctly identify unusually formatted questions.
- K-Means clustering is unsupervised and clusters should be interpreted as groups of textual similarity rather than guaranteed academic topics.
- AI confidence is a model-generated confidence value and should not be treated as a calibrated probability.
- Gemini API usage may incur costs depending on the API account and model usage.
- The application currently analyzes uploaded papers individually rather than maintaining a persistent historical database.

---

## 🔮 Future Improvements

Potential future improvements include:

- Multi-paper batch analysis
- Historical question frequency across multiple years
- Embedding-based semantic clustering
- Automatic syllabus extraction
- Better OCR preprocessing
- Persistent database storage
- Interactive topic dashboards
- Export to Excel/PDF
- Question similarity search
- Automated revision-plan generation
- Trend analysis across examination years
- Authentication and user accounts

---

## 👨‍💻 Project Description

**AI Question Paper Analyzer** is an AI/ML-based academic analytics application designed to help students analyze previous-year question papers.

It combines **Gemini LLM semantic analysis**, **traditional machine learning**, and **statistical Pareto analysis** to identify frequently occurring topics, group similar questions, analyze difficulty and question types, and generate data-driven study recommendations.

---

## 📜 License

Add your preferred license here, such as MIT, if you intend to distribute the project as open source.
