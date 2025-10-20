import debugpy
import os

if os.getenv("DEBUG_ATTACH", "false").lower() == "true":
    debugpy.listen(("localhost", 5678))
    print("🪲 Waiting for debugger to attach on port 5678...")
    debugpy.wait_for_client()
    print("✅ Debugger attached.")

import streamlit as st

# Add project root to Python path
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from core.agent import Agent  # noqa: E402


st.set_page_config(page_title="AI Knowledge Assistant")
st.title("🧠 AI Knowledge Assistant (ReAct Reasoning Agent)")

if "agent" not in st.session_state:
    st.session_state.agent = Agent()

question = st.text_area("Ask me anything:", height=120)
go = st.button("Run")

if go and question.strip():
    area = st.container()
    with st.spinner("Thinking..."):

        def draw(step):
            with area:
                st.markdown(f"### Step {step.idx}")
                st.write(f"**Thought:** {step.thought}")
                if step.tool:
                    st.code(
                        f"Action: {step.tool}('{step.tool_input}')", language="text"
                    )
                if step.observation:
                    st.info(step.observation[:1200])
                if step.confidence is not None:
                    st.caption(
                        f"Relevance: {step.confidence:.2f} — {step.confidence_reason or ''}"
                    )
                if step.final_answer:
                    st.success(step.final_answer)

        # stream steps
        for _ in st.session_state.agent.run_stream(question, on_step=draw):
            pass
