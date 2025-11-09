# Sleep Lab — Agentic AI for Optimal Sleep Quality

"We built a research team that never sleeps… so you finally can."

Sleep Lab is a multi-agent research system designed to accelerate scientific understanding in the domain of sleep quality. Instead of relying on a single LLM to summarize papers, Sleep Lab orchestrates several specialized AI agents that work together like a miniature research lab: ingesting studies, critiquing claims, synthesizing insights, and surfacing follow-up questions worth investigating.

The goal is simple: show how multi-agent reasoning can produce clearer, faster, and more verifiable scientific insights compared to single-model approaches.

---

## Key Features

### Multi-Agent Reasoning
- **Researcher**: ingests studies and extracts key findings  
- **Reviewer**: challenges claims and evaluates evidence quality  
- **Synthesizer**: reconciles disagreements into unified insights  
- **Explorer**: proposes follow-up questions and new angles  

### Robust Ingestion Pipeline
- Fetches sources from **arXiv**, **OpenAlex**, and general web pages  
- HTML extraction via **Trafilatura**, **requests_html**, and custom fallbacks  
- Claim mining using regex heuristics, study-type weighting, and confidence scoring  

### LLM-Optional Architecture
- With a Gemini API key: agents use **Gemini Flash or Pro**  
- Without a key: deterministic rule-based agents provide fully offline reasoning  

### Performance Optimizations
- Early-stop heuristic to avoid unnecessary downloads  
- Lightweight summarization for long documents  
- Caching and prioritization based on study strength  

### Interactive GUI
- Modern **Gradio 4.x** interface that streams real-time agent debate and evidence synthesis  

---

## Tech Stack
- Python  
- Gemini Flash/Pro (optional)  
- Gradio 4.x  
- NetworkX  
- Matplotlib  
- Trafilatura, requests_html  

---

## How to Run

1. Install dependencies:
   ```bash
    pip install -r requirements.txt
    ```

2. (Optional) Set your Gemini key:
    ```bash
    export GEMINI_API_KEY="your-key"
    ```

3. Launch the interface:
    ```bash
    python gui.py
    ```

---

## Project Goal

To demonstrate how coordinated AI agents can collaborate meaningfully to generate more transparent, reliable, and scientifically grounded insights than a single LLM running alone.
