# 🔬 Autonomous Scientific Experiment Planner

An AI-powered research agent that analyzes scientific literature and proposes new experiments.

## Features
- 📥 Fetches real papers from arXiv
- 🔍 Semantic search using TF-IDF vector store
- 🕳️ Detects research gaps using LLM reasoning
- 🧪 Generates detailed experiment plans with dataset requirements
- ⬇️ Download full report as text file

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/experiment-planner
cd experiment-planner
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run locally
```bash
streamlit run app.py
```

### 4. Add your Groq API key
Get a free key at [console.groq.com](https://console.groq.com)

Enter it in the sidebar when running the app.

## Deployment on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file as `app.py`
5. Add secret: `GROQ_API_KEY = "your_key_here"`

## Tech Stack
- **Frontend**: Streamlit
- **LLM**: Groq API (Llama3, Mixtral)
- **Literature**: arXiv API
- **Search**: TF-IDF + Cosine Similarity

## Industry Track
This project uses Groq-hosted LLMs for cloud deployment. The offline model variant uses Ollama locally.
