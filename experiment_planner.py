import os
from groq import Groq


def generate_experiment_plan(gaps: str, research_topic: str, model: str = "llama3-8b-8192") -> dict:
    """Generate a detailed experiment plan based on identified research gaps."""
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    prompt = f"""You are an expert research scientist. Based on the following research gaps identified in "{research_topic}", create a detailed experiment plan.

RESEARCH GAPS:
{gaps}

Generate a comprehensive experiment plan with these sections:

## 1. Research Hypothesis
State the primary hypothesis to test.

## 2. Objectives
List 3-4 specific, measurable objectives.

## 3. Methodology
Describe the experimental approach step by step.

## 4. Dataset Requirements
Specify:
- Data types needed
- Sample size recommendations
- Data collection methods
- Preprocessing steps

## 5. Expected Outcomes
What results are anticipated and how they address the gaps.

## 6. Evaluation Metrics
How will success be measured?

## 7. Limitations & Risks
Potential challenges and mitigation strategies.

Be specific, scientific, and practical."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7
    )
    
    full_plan = response.choices[0].message.content.strip()
    
    # Extract dataset requirements section
    dataset_section = ""
    lines = full_plan.split('\n')
    in_dataset = False
    dataset_lines = []
    
    for line in lines:
        if "dataset requirements" in line.lower() or "4." in line:
            in_dataset = True
        elif in_dataset and line.startswith("## ") and "dataset" not in line.lower():
            in_dataset = False
        
        if in_dataset:
            dataset_lines.append(line)
    
    dataset_section = '\n'.join(dataset_lines) if dataset_lines else ""
    
    return {
        "full_plan": full_plan,
        "dataset_requirements": dataset_section
    }
