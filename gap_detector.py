import os
from groq import Groq


def detect_gaps(papers: list[dict], research_topic: str, model: str = "llama3-8b-8192") -> str:
    """Use Groq LLM to identify research gaps from fetched papers."""
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Build context from papers
    paper_summaries = "\n\n".join([
        f"Paper {i+1}: {p['title']}\nSummary: {p['summary'][:400]}"
        for i, p in enumerate(papers[:6])
    ])
    
    prompt = f"""You are an expert scientific researcher analyzing literature on: "{research_topic}"

Here are the relevant papers found:

{paper_summaries}

Based on these papers, identify 4-6 clear research gaps and limitations. For each gap:
- Be specific and actionable
- Reference what current studies are missing
- Suggest what kind of investigation is needed

Format each gap on a new line starting with a number. Be concise and scientific."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()
