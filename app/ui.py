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

query = st.text_area("💬 Ask me anything:", height=120)
if st.button("Run"):
    with st.spinner("Thinking..."):
        result = st.session_state.agent.run(query)

    # Split reasoning steps visually
    for line in result.split("\n"):
        if line.startswith("🧠"):
            st.markdown(f"### {line}")
        elif line.startswith("🔍"):
            st.info(line)
        elif "✅ Final Answer" in line:
            st.success(line)
        else:
            st.write(line)
