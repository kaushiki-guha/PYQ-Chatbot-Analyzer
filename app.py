import os
import re
import json
import sys
import asyncio
from collections import Counter

import gradio as gr
import pdfplumber
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()

os.environ["OMP_NUM_THREADS"] = "1"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    gemini_client = None


try:
    if sys.platform == "win32":
        if not isinstance(
            asyncio.get_event_loop_policy(),
            asyncio.WindowsSelectorEventLoopPolicy
        ):
            asyncio.set_event_loop_policy(
                asyncio.WindowsSelectorEventLoopPolicy()
            )
except Exception:
    pass


SUBJECT_MAP = {
    "Probability and Statistics": {
        "units": {
            "Probability & Random Variables": [
                "probability",
                "random variable",
                "bayes",
                "discrete distribution",
                "normal distribution",
                "binomial distribution",
                "poisson distribution",
                "expectation",
                "variance"
            ],
            "Joint Probability & Distributions": [
                "joint probability",
                "marginal probability",
                "conditional probability",
                "covariance",
                "correlation",
                "joint distribution"
            ],
            "Descriptive Statistics": [
                "regression",
                "histogram",
                "box plot",
                "mean median mode",
                "skewness",
                "kurtosis",
                "dispersion",
                "standard deviation"
            ],
            "Inferential Statistics": [
                "hypothesis testing",
                "confidence interval",
                "central limit theorem",
                "sampling",
                "estimation",
                "p-value",
                "test statistic"
            ]
        }
    },
    "Discrete Structures": {
        "units": {
            "Logic": [
                "proposition",
                "quantifier",
                "rule of inference",
                "mathematical induction",
                "predicate",
                "logical equivalence"
            ],
            "Set, Relation & Function": [
                "set",
                "relation",
                "function",
                "hasse diagram",
                "lattice",
                "equivalence relation",
                "partial order"
            ],
            "Recurrence Relations": [
                "recurrence relation",
                "generating function",
                "recurrence"
            ],
            "Groups and Rings": [
                "group",
                "ring",
                "field",
                "subgroup",
                "lagrange",
                "homomorphism"
            ],
            "Graph Theory (DS)": [
                "graph theory",
                "tree",
                "mst",
                "minimum spanning tree",
                "eulerian",
                "hamiltonian",
                "graph coloring"
            ]
        }
    },
    "Data Structures": {
        "units": {
            "Linear Data Structures": [
                "stack",
                "queue",
                "linked list",
                "doubly linked list",
                "circular linked list"
            ],
            "Trees (DS)": [
                "binary tree",
                "binary search tree",
                "avl tree",
                "height balanced",
                "heap",
                "tree traversal"
            ],
            "Graphs (DS)": [
                "graph",
                "bfs",
                "dfs",
                "topological sort",
                "shortest path"
            ],
            "Sorting and Searching": [
                "sorting",
                "searching",
                "hash table",
                "quick sort",
                "merge sort",
                "bubble sort",
                "selection sort",
                "binary search"
            ]
        }
    },
    "Digital Systems Design": {
        "units": {
            "Basic VLSI System Design": [
                "vlsi",
                "verilog hdl",
                "design flow",
                "moore law",
                "hardware description language"
            ],
            "Binary Codes & Boolean Algebra": [
                "boolean algebra",
                "k-map",
                "binary code",
                "sop",
                "pos",
                "logic gates"
            ],
            "Combinational Circuits": [
                "combinational circuit",
                "adder",
                "multiplexer",
                "decoder",
                "encoder",
                "comparator"
            ],
            "Sequential Circuits": [
                "sequential circuit",
                "flip-flop",
                "shift register",
                "counter design",
                "register",
                "jk flip flop",
                "d flip flop"
            ],
            "Advanced Concepts (DSD)": [
                "cmos",
                "cmos design",
                "gate design",
                "cmos level"
            ]
        }
    },
    "Automata Theory": {
        "units": {
            "Finite Automata": [
                "finite automata",
                "dfa",
                "nfa",
                "myhill nerode",
                "finite state machine"
            ],
            "Regular Expression (RE)": [
                "regular expression",
                "pumping lemma",
                "moore and mealy",
                "kleene",
                "regular language"
            ],
            "CFG and CFL": [
                "context free grammar",
                "cfg",
                "normal form",
                "ambiguity",
                "context free language",
                "chomsky normal form"
            ],
            "Push Down Automata (PDA)": [
                "push down automata",
                "pda",
                "conversion of cfg",
                "pushdown automaton"
            ],
            "Turing Machines & Undecidability": [
                "turing machine",
                "tm",
                "halting problem",
                "undecidable",
                "recursive",
                "recursively enumerable"
            ]
        }
    },
    "Operating Systems": {
        "units": {
            "Introduction (OS)": [
                "operating systems",
                "os operations",
                "virtualization",
                "os services",
                "system calls"
            ],
            "Process Management": [
                "scheduling",
                "critical section",
                "deadlock",
                "process synchronization",
                "semaphores",
                "process",
                "thread"
            ],
            "Memory Management": [
                "memory management",
                "paging",
                "segmentation",
                "virtual memory",
                "thrashing",
                "page replacement"
            ],
            "File System": [
                "file system",
                "file implementation",
                "protection domains",
                "disk scheduling",
                "file allocation"
            ]
        }
    }
}


reader = None


def get_ocr_reader():
    global reader
    if reader is None:
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
        except Exception:
            reader = "error"
    return reader


def get_file_path(file):
    if file is None:
        return None
    if isinstance(file, str):
        return file
    if hasattr(file, "name"):
        return file.name
    return str(file)


def extract_text_from_file(file, file_type):
    try:
        file_path = get_file_path(file)
        if not file_path:
            return "Error: No file selected."

        if file_type == "PDF":
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text

        if file_type == "Image":
            ocr_reader = get_ocr_reader()
            if ocr_reader == "error":
                return "Error: EasyOCR is not available. Install it using: pip install easyocr"
            Image.open(file_path).convert("L")
            result = ocr_reader.readtext(file_path, detail=0)
            return " ".join(result)

        if file_type == "Text":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        return "Error: Invalid file type."
    except Exception as e:
        return f"Error extracting text: {str(e)}"


def extract_questions(text):
    questions = []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    numbered_pattern = re.compile(
        r"^\s*(?:Q\.?\s*\d+|Question\s*\d+|\d+[\.:\)] )\s*(.+?)\s*$",
        re.IGNORECASE,
    )

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = numbered_pattern.match(line)
        if match:
            question = match.group(1).strip()
            if len(question) >= 10:
                questions.append(question)

    question_marks = re.findall(r"([^\n]{15,}\?)", text)
    for question in question_marks:
        question = question.strip()
        if len(question) >= 10:
            questions.append(question)

    cleaned = []
    seen = set()
    for question in questions:
        question = re.sub(r"\s+", " ", question).strip()
        key = question.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(question)
    return cleaned


def build_syllabus_text():
    output = []
    for subject, data in SUBJECT_MAP.items():
        output.append(f"SUBJECT: {subject}")
        for unit, keywords in data["units"].items():
            output.append(f"  UNIT: {unit}")
            output.append("  KEYWORDS: " + ", ".join(keywords))
    return "\n".join(output)


def classify_questions_with_gemini(questions):
    if not gemini_client:
        raise RuntimeError("Gemini API key not found. Set GEMINI_API_KEY in your .env file.")

    syllabus_text = build_syllabus_text()
    question_text = "\n".join(f"{i + 1}. {question}" for i, question in enumerate(questions))

    prompt = f"""
You are an academic question-paper analysis system.

Your task is to classify every question in a question paper.

You MUST use the subjects and units supplied below.

SYLLABUS:

{syllabus_text}

For each question determine:

1. subject
2. unit
3. topic
4. difficulty
5. question_type
6. confidence

Allowed difficulty values:
- Easy
- Medium
- Hard

Allowed question types:
- Conceptual
- Numerical
- Programming
- Derivation
- Proof
- Algorithm
- Definition
- Comparison
- Application
- Other

Important rules:

- Do not invent subjects that are not present in the syllabus.
- Select the closest matching subject.
- Select the closest matching unit.
- Topic may be more specific than the unit.
- Understand the meaning of the question instead of relying only on exact keywords.
- Confidence must be between 0 and 1.
- Classify EVERY question exactly once.

QUESTIONS:

{question_text}
"""

    schema = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question_number": {"type": "integer"},
                        "subject": {"type": "string"},
                        "unit": {"type": "string"},
                        "topic": {"type": "string"},
                        "difficulty": {"type": "string"},
                        "question_type": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "question_number",
                        "subject",
                        "unit",
                        "topic",
                        "difficulty",
                        "question_type",
                        "confidence",
                    ],
                },
            }
        },
        "required": ["questions"],
    }

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    result = json.loads(response.text)
    return result["questions"]


def run_ml_analysis(questions):
    if len(questions) == 0:
        return pd.DataFrame(), [], {}

    if len(questions) == 1:
        return pd.DataFrame([{"Question": questions[0], "ML Cluster": 1}]), [], {}

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(questions)

    num_clusters = min(5, max(2, len(questions) // 2))
    num_clusters = min(num_clusters, len(questions))

    model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    for cluster_id in range(num_clusters):
        center = model.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-5:][::-1]
        keywords = [feature_names[index] for index in top_indices]
        cluster_keywords[cluster_id + 1] = keywords

    ml_df = pd.DataFrame({"Question": questions, "ML Cluster": cluster_labels + 1})
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarities = []
    for i in range(len(questions)):
        other_scores = [similarity_matrix[i][j] for j in range(len(questions)) if i != j]
        similarities.append(round(max(other_scores) * 100, 1) if other_scores else 0)

    ml_df["Closest Question Similarity (%)"] = similarities
    return ml_df, cluster_keywords, similarity_matrix


def create_analysis_dataframe(questions, gemini_results, ml_df):
    rows = []
    ml_clusters = {}
    if not ml_df.empty:
        for _, row in ml_df.iterrows():
            ml_clusters[row["Question"]] = row["ML Cluster"]

    for i, question in enumerate(questions):
        if i < len(gemini_results):
            result = gemini_results[i]
        else:
            result = {
                "question_number": i + 1,
                "subject": "Unknown",
                "unit": "Unknown",
                "topic": "Unknown",
                "difficulty": "Unknown",
                "question_type": "Other",
                "confidence": 0,
            }

        rows.append({
            "Question No": i + 1,
            "Question": question,
            "Subject": result.get("subject", "Unknown"),
            "Unit": result.get("unit", "Unknown"),
            "Topic": result.get("topic", "Unknown"),
            "Difficulty": result.get("difficulty", "Unknown"),
            "Question Type": result.get("question_type", "Other"),
            "AI Confidence": round(float(result.get("confidence", 0)) * 100, 1),
            "ML Cluster": ml_clusters.get(question, "-"),
        })

    return pd.DataFrame(rows)


def perform_pareto_analysis(df):
    if df.empty:
        return pd.DataFrame(), []

    topic_counts = df["Topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    topic_counts = topic_counts.sort_values("Count", ascending=False).reset_index(drop=True)
    total = topic_counts["Count"].sum()
    topic_counts["Percentage"] = (topic_counts["Count"] / total * 100).round(1)
    topic_counts["Cumulative %"] = topic_counts["Percentage"].cumsum().round(1)

    pareto_topics = []
    cumulative_before = 0
    for _, row in topic_counts.iterrows():
        if cumulative_before < 80:
            pareto_topics.append(row["Topic"])
        cumulative_before = row["Cumulative %"]
    return topic_counts, pareto_topics


def create_charts(topic_df, pareto_topics):
    if topic_df.empty:
        return None, None

    plot_df = topic_df.head(12).copy()
    priority_set = set(pareto_topics)
    colors = ["#27ae60" if topic in priority_set else "#95a5a6" for topic in plot_df["Topic"]]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    y_pos = range(len(plot_df))
    ax1.barh(y_pos, plot_df["Count"], color=colors, edgecolor="black")
    for i, (count, percentage) in enumerate(zip(plot_df["Count"], plot_df["Percentage"])):
        ax1.text(count + 0.1, i, f"{count} ({percentage}%)", va="center", fontsize=9)
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(plot_df["Topic"], fontsize=9)
    ax1.set_xlabel("Number of Questions")
    ax1.set_title("AI Topic Distribution")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    freq_path = "freq_chart.png"
    plt.savefig(freq_path, dpi=120, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(11, 6))
    x_pos = range(len(plot_df))
    ax2.bar(x_pos, plot_df["Count"], color=colors, edgecolor="black")
    ax2.set_ylabel("Number of Questions")
    ax2.set_xlabel("Topics")
    ax3 = ax2.twinx()
    ax3.plot(x_pos, plot_df["Cumulative %"], "r-o", linewidth=2, markersize=6)
    ax3.axhline(y=80, color="orange", linestyle="--", linewidth=2)
    ax3.set_ylabel("Cumulative Percentage")
    ax3.set_ylim(0, 105)
    ax2.set_xticks(list(x_pos))
    ax2.set_xticklabels(plot_df["Topic"], rotation=45, ha="right", fontsize=9)
    ax2.set_title("Pareto Analysis - 80/20 Rule")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    pareto_path = "pareto_chart.png"
    plt.savefig(pareto_path, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    return freq_path, pareto_path


def generate_ai_recommendations(analysis_df, topic_df, pareto_topics):
    if not gemini_client:
        return "AI recommendations unavailable because the Gemini API key is missing."

    topic_summary = topic_df.to_dict(orient="records")
    difficulty_counts = analysis_df["Difficulty"].value_counts().to_dict()
    type_counts = analysis_df["Question Type"].value_counts().to_dict()

    prompt = f"""
You are an academic study-planning assistant.

Analyze this question-paper analysis.

TOPIC FREQUENCIES:
{json.dumps(topic_summary, indent=2)}

PRIORITY TOPICS:
{json.dumps(pareto_topics, indent=2)}

DIFFICULTY DISTRIBUTION:
{json.dumps(difficulty_counts, indent=2)}

QUESTION TYPE DISTRIBUTION:
{json.dumps(type_counts, indent=2)}

Give a concise study recommendation.

Include:

1. Most important topics to study first
2. Topics that appear frequently
3. Difficulty observations
4. Question-type observations
5. A recommended study order
6. One short overall conclusion

Do not invent topics that are not present in the supplied analysis.
"""
    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


def create_report(analysis_df, topic_df, pareto_topics, ml_clusters, ai_recommendations):
    if analysis_df.empty:
        return "No analysis available."

    total_questions = len(analysis_df)
    total_topics = len(topic_df)
    priority_questions = topic_df[topic_df["Topic"].isin(pareto_topics)]["Count"].sum()
    coverage = (priority_questions / total_questions * 100) if total_questions else 0
    subjects = analysis_df["Subject"].value_counts()
    avg_confidence = analysis_df["AI Confidence"].mean()

    report = ["=" * 75, "          AI QUESTION PAPER ANALYSIS REPORT", "=" * 75, "", "ANALYSIS METHOD", "---------------", "• LLM: Gemini", "• ML: TF-IDF + K-Means clustering", "• Statistical Analysis: Frequency + Pareto", "• Question Extraction: Regex", "", "SUMMARY", "-------", f"• Total Questions       : {total_questions}", f"• Unique Topics         : {total_topics}", f"• Priority Topics       : {len(pareto_topics)}", f"• Priority Coverage     : {coverage:.1f}%", f"• Average AI Confidence : {avg_confidence:.1f}%", f"• ML Clusters           : {len(ml_clusters)}", "", "SUBJECT DISTRIBUTION", "--------------------"]
    for subject, count in subjects.items():
        percentage = (count / total_questions * 100) if total_questions else 0
        report.append(f"• {subject:<35} {count:>3} questions ({percentage:.1f}%)")
    report.extend(["", "=" * 75, "PRIORITY TOPICS", "---------------"])
    for _, row in topic_df.iterrows():
        if row["Topic"] in pareto_topics:
            report.append(f"★ {row['Topic']:<35} {int(row['Count']):>3} questions ({row['Percentage']:.1f}%)")
    report.extend(["", "=" * 75, "COMPLETE TOPIC BREAKDOWN", "------------------------"])
    for _, row in topic_df.iterrows():
        marker = "★" if row["Topic"] in pareto_topics else "○"
        report.append(f"{marker} {row['Topic']:<35} {int(row['Count']):>3} ({row['Percentage']:>5.1f}%) Cum: {row['Cumulative %']:>5.1f}%")
    report.extend(["", "=" * 75, "ML CLUSTER ANALYSIS", "-------------------"])
    for cluster_id, keywords in ml_clusters.items():
        report.append(f"Cluster {cluster_id}: {', '.join(keywords)}")
    report.extend(["", "=" * 75, "AI STUDY RECOMMENDATIONS", "------------------------", ai_recommendations, "", "=" * 75])
    return "\n".join(report)


def process_file(file, file_type):
    if file is None:
        return "❌ Please upload a file.", None, None, None

    if not GEMINI_API_KEY:
        return (
            "❌ Gemini API key not found.\n\nCreate a .env file containing:\n\nGEMINI_API_KEY=your_api_key",
            None,
            None,
            None,
        )

    text = extract_text_from_file(file, file_type)
    if text.startswith("Error"):
        return f"❌ {text}", None, None, None
    if len(text.strip()) < 20:
        return "❌ No usable text extracted from file.", None, None, None

    questions = extract_questions(text)
    if not questions:
        return "❌ No questions found. Make sure the uploaded document contains numbered questions or question marks.", None, None, None

    try:
        gemini_results = classify_questions_with_gemini(questions)
    except Exception as e:
        return f"❌ Gemini analysis failed.\n\nError: {str(e)}", None, None, None

    try:
        ml_df, ml_clusters, _ = run_ml_analysis(questions)
    except Exception as e:
        return f"❌ ML analysis failed.\n\nError: {str(e)}", None, None, None

    analysis_df = create_analysis_dataframe(questions, gemini_results, ml_df)
    topic_df, pareto_topics = perform_pareto_analysis(analysis_df)
    freq_chart, pareto_chart = create_charts(topic_df, pareto_topics)

    try:
        ai_recommendations = generate_ai_recommendations(analysis_df, topic_df, pareto_topics)
    except Exception as e:
        ai_recommendations = f"Unable to generate AI recommendations: {str(e)}"

    report = create_report(analysis_df, topic_df, pareto_topics, ml_clusters, ai_recommendations)
    detailed_table = analysis_df[["Question No", "Question", "Subject", "Unit", "Topic", "Difficulty", "Question Type", "AI Confidence", "ML Cluster"]]
    return report, freq_chart, pareto_chart, detailed_table


with gr.Blocks(title="AI Question Paper Analyzer") as demo:
    gr.Markdown("""
    # 🎓 AI Question Paper Analyzer

    ### Gemini LLM + Machine Learning + Pareto Analysis
    """)

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload Question Paper", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"])
            file_type = gr.Radio(choices=["PDF", "Image", "Text"], value="PDF", label="File Type")
            analyze_button = gr.Button("🔍 Analyze Question Paper", variant="primary", size="lg")

    gr.Markdown("## 📊 Analysis Report")
    output_text = gr.Textbox(label="AI Analysis Report", lines=35)

    with gr.Row():
        chart1 = gr.Image(label="Topic Distribution")
        chart2 = gr.Image(label="Pareto Analysis")

    gr.Markdown("## 🧠 Question-Level AI + ML Analysis")
    detailed_output = gr.Dataframe(headers=["Question No", "Question", "Subject", "Unit", "Topic", "Difficulty", "Question Type", "AI Confidence", "ML Cluster"], interactive=False, wrap=True)

    gr.Markdown("""
    ---

    ### How the system works

    **1. File Processing**

    PDF, image or text question papers are converted into text.

    **2. Question Extraction**

    Regular expressions identify individual questions.

    **3. Gemini LLM Analysis**

    Gemini identifies the subject, unit, topic, difficulty and question type for each question.

    **4. Machine Learning Analysis**

    TF-IDF converts questions into numerical feature vectors. K-Means groups similar questions into clusters.

    **5. Statistical Analysis**

    Question frequencies are calculated and Pareto analysis identifies the topics responsible for approximately 80% of the question volume.

    **6. AI Recommendations**

    Gemini generates study recommendations based on the resulting topic, difficulty and question-type distribution.
    """)

    analyze_button.click(fn=process_file, inputs=[file_input, file_type], outputs=[output_text, chart1, chart2, detailed_output])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
