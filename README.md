# AI Knowledge Assistant Agent

A **local, privacy-friendly AI reasoning agent** built in Python.  
It runs entirely on your machine, using **LM Studio** or any local OpenAI-compatible endpoint —  
no paid API keys required.

The agent follows a **ReAct (Reason + Act + Observe)** pattern:
- Thinks step by step   
- Executes external tools (e.g. search, summarize) 
- Evaluates relevance   
- Learns from memory 
- Streams its reasoning live in a clean Streamlit UI 

---

## Features

### ReAct-style reasoning loop
- **THINK → ACT → OBSERVE → REFLECT → ANSWER**
- Each step is displayed live with reasoning trace.

### Tool calling
- **DuckDuckGo search** for external knowledge  
- **Summarize** for condensing long text  
- Easily extendable with your own tools (e.g. Wikipedia, calculator).

### Confidence scoring
- After each tool call, the agent uses the LLM to score **relevance (0–1)**  
  and explain why an observation is or isn’t useful.

### Persistent memory
- Stores short-term Q&A in `memory.json`
- Logs full reasoning traces in `memory.traces.json` for retraining or debugging

### Tool chaining
- Automatically chains tools (e.g. `search → summarize → final`)
- Prevents infinite loops and repeated actions

### Streamlit UI
- Interactive chat-style interface  
- Live reasoning trace (streamed step by step)

### How to run
`poetry run streamlit run app/ui.py`