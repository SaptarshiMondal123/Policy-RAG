# 🤖 Policy RAG Assistant (Hybrid Architecture)

A robust Retrieval-Augmented Generation (RAG) system designed to answer questions from internal policy documents with high precision. This project demonstrates a Hybrid Architecture using local embeddings for cost-efficiency and Google's Gemini Flash model for reasoning, complete with a self-correction evaluation loop.

## 🚀 Key Features (Engineering Highlights)

*   **Hybrid RAG Architecture**: Combines HuggingFace Local Embeddings (Free, Unlimited, Private) with Gemini 2.5 Flash-Lite (High-Performance Reasoning).
*   **🛡️ Self-Correction & Evaluation**: Includes an `evaluate.py` script that uses an "LLM-as-a-Judge" pattern to grade answers against ground-truth facts, ensuring reliability.
*   **🔍 Two-Stage Retrieval (Reranking)**: Implements a score-based filtering step (score < 1.5) to discard irrelevant chunks and reduce hallucinations.
*   **🎭 Dynamic Prompt Templating**: Uses LangChain to switch between a "Basic" (Chatty) persona and an "Advanced" (Strict, Analytical) persona.
*   **✅ Structured JSON Output**: Enforces strict output schemas using Pydantic parsers to ensure downstream application compatibility.
*   **⚡ Dual Interface**:
    *   **CLI (`main.py`)**: For automated, headless interaction (fulfills "no fancy UI" requirement).
    *   **Streamlit Dashboard (`app.py`)**: A developer tool to visualize latency, confidence scores, and source documents.

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/SaptarshiMondal123/Policy-RAG.git
cd Policy-RAG
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up Environment Variables:
Create a `.env` file in the root directory and add your Google API Key:

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
```

## 🏃‍♂️ Usage Guide

### 1. Ingest Data (Build the Database)

First, process the PDF documents in the `data/` folder and build the local ChromaDB vector store.

```bash
python main.py --ingest
```

### 2. CLI Mode (Production / Headless)

Ask questions directly from your terminal. This mode is optimized for speed and logging.

```bash
python main.py --query "What is the deadline for a refund?"
```

### 3. Developer UI (Streamlit)

Launch the interactive dashboard to visualize the "Advanced" vs "Basic" prompt differences, check confidence scores, and see the raw source text.

```bash
streamlit run app.py
```

*Pro Tip: If you want to test this on your phone or share it live, use `ngrok http 8501`.*

### 4. Run Automated Evaluation

Run the test suite to benchmark the system's accuracy and safety using the LLM-as-a-Judge pattern.

```bash
python evaluate.py
```

## 🏗️ Project Structure

```plaintext
📂 Policy_RAG_Assignment/
├── 📄 main.py            # CLI entry point for Ingestion & Querying
├── 📄 app.py             # Streamlit Dashboard (Developer UI)
├── 📄 rag_engine.py      # CORE LOGIC: RAG Pipeline, Reranking, Prompting
├── 📄 evaluate.py        # TEST SUITE: LLM-as-a-Judge benchmarking
├── 📄 requirements.txt   # Python dependencies
├── 📄 README.md          # Project Documentation
├── 📂 data/              # Place your PDF documents here
└── 📂 chroma_db/         # Local Vector Database (Auto-generated)
```

## 🧠 Engineering Decisions

### Why Streamlit?

While the requirement stated "no fancy UI," I included a Streamlit interface as a Developer Tool. It allows for real-time debugging of Latency, Confidence Scores, and Source Retrieval verification, which is critical for maintaining RAG reliability.

### Why Local Embeddings + Gemini?

*   **Cost & Privacy**: Local embeddings (`all-MiniLM-L6-v2`) run on-device, meaning document chunks are not sent to an external API for vectorization. This reduces cost to zero and improves data privacy.
*   **Performance**: Gemini Flash-Lite is used specifically for its high throughput and large context window, preventing 429 Resource Exhausted errors during intensive evaluation loops.

### Why Reranking?

Standard vector search often retrieves "nearest neighbors" that are mathematically close but contextually irrelevant. The added Distance Threshold Filtering acts as a lightweight reranker, ensuring only high-quality context reaches the LLM, drastically reducing hallucinations.

## 📦 Requirements

*   Python 3.9+
*   langchain
*   streamlit
*   chromadb
*   google-generativeai
*   pydantic
*   python-dotenv

---

**Author**: Saptarshi Mondal
**Status**: Completed & Tested
