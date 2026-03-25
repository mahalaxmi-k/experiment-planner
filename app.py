import streamlit as st
import os
from fetch_papers import fetch_arxiv_papers
from embed_search import build_vector_store, search_similar
from gap_detector import detect_gaps
from experiment_planner import generate_experiment_plan

# Page config
st.set_page_config(
    page_title="Autonomous Scientific Experiment Planner",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1e3a5f;
        padding: 1rem 1.5rem;
        border-radius: 4px;
        margin-bottom: 1.5rem;
    }
    .paper-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .gap-item {
        background: #fff8e1;
        border-left: 3px solid #f9a825;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 4px;
    }
    .stButton>button {
        background-color: #1e3a5f;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2d5499;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🔬 Autonomous Scientific Experiment Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze literature → Detect gaps → Generate experiment plans</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", 
                                   value=os.environ.get("GROQ_API_KEY", ""),
                                   help="Get free key at console.groq.com")
    
    st.markdown("---")
    st.subheader("Search Settings")
    num_papers = st.slider("Number of papers to fetch", 3, 15, 6)
    model_choice = st.selectbox("LLM Model", [
        "llama3-8b-8192",
        "llama3-70b-8192", 
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ])
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. 📥 Fetch papers from arXiv")
    st.markdown("2. 🔍 Build semantic search index")
    st.markdown("3. 🕳️ Detect research gaps")
    st.markdown("4. 🧪 Generate experiment plan")
    
    st.markdown("---")
    st.markdown("*Powered by Groq API + arXiv*")

# Main input
col1, col2 = st.columns([3, 1])
with col1:
    research_topic = st.text_input(
        "🔎 Research Topic",
        placeholder="e.g., transformer models for drug discovery, CRISPR gene editing efficiency...",
        label_visibility="collapsed"
    )
with col2:
    run_button = st.button("🚀 Analyze", use_container_width=True)

# Validation
if run_button:
    if not groq_api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar.")
        st.stop()
    if not research_topic:
        st.error("⚠️ Please enter a research topic.")
        st.stop()
    
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Step 1: Fetch Papers
    with st.spinner("📥 Fetching papers from arXiv..."):
        try:
            papers = fetch_arxiv_papers(research_topic, max_results=num_papers)
            if not papers:
                st.error("No papers found. Try a different topic.")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching papers: {e}")
            st.stop()

    st.success(f"✅ Found {len(papers)} papers")

    # Step 2: Show Papers
    with st.expander("📄 Fetched Papers", expanded=False):
        for i, paper in enumerate(papers, 1):
            st.markdown(f"""
            <div class="paper-card">
                <strong>{i}. {paper['title']}</strong><br>
                <small>👥 {paper.get('authors', 'N/A')} &nbsp;|&nbsp; 📅 {paper.get('published', 'N/A')}</small><br>
                <small>{paper['summary'][:300]}...</small>
            </div>
            """, unsafe_allow_html=True)

    # Step 3: Build Vector Store & Search
    with st.spinner("🔍 Building semantic index..."):
        try:
            vectorstore = build_vector_store(papers)
            relevant_docs = search_similar(vectorstore, research_topic, k=min(5, len(papers)))
        except Exception as e:
            st.error(f"Error building index: {e}")
            st.stop()

    # Step 4: Detect Gaps
    with st.spinner("🕳️ Detecting research gaps..."):
        try:
            gaps = detect_gaps(relevant_docs, research_topic, model_choice)
        except Exception as e:
            st.error(f"Error detecting gaps: {e}")
            st.stop()

    st.markdown("### 🕳️ Identified Research Gaps")
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    for gap in gaps.split('\n'):
        if gap.strip():
            st.markdown(f'<div class="gap-item">• {gap.strip()}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 5: Generate Experiment Plan
    with st.spinner("🧪 Generating experiment plan..."):
        try:
            plan = generate_experiment_plan(gaps, research_topic, model_choice)
        except Exception as e:
            st.error(f"Error generating plan: {e}")
            st.stop()

    st.markdown("### 🧪 Proposed Experiment Plan")
    
    tabs = st.tabs(["📋 Full Plan", "📊 Dataset Requirements", "📝 Raw Output"])
    
    with tabs[0]:
        st.markdown(plan.get("full_plan", "No plan generated"))
    
    with tabs[1]:
        dataset_info = plan.get("dataset_requirements", "")
        if dataset_info:
            st.markdown(dataset_info)
        else:
            st.info("Dataset requirements are included in the full plan above.")
    
    with tabs[2]:
        st.code(str(plan), language="text")

    # Download button
    full_output = f"""
AUTONOMOUS SCIENTIFIC EXPERIMENT PLANNER
==========================================
Research Topic: {research_topic}

PAPERS ANALYZED:
{chr(10).join([f"- {p['title']}" for p in papers])}

RESEARCH GAPS:
{gaps}

EXPERIMENT PLAN:
{plan.get('full_plan', '')}
"""
    st.download_button(
        label="⬇️ Download Full Report",
        data=full_output,
        file_name=f"experiment_plan_{research_topic[:30].replace(' ', '_')}.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("<center><small>Built for Industry Track | Uses arXiv API + Groq LLM</small></center>", unsafe_allow_html=True)
