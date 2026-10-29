"""
Day 2, Hours 1-3: Streamlit chat UI with a live "Agent steps" panel.

Run:  streamlit run app.py
"""
import os
import subprocess
import sys
import uuid

import streamlit as st

st.set_page_config(page_title="Sarkari Saathi", page_icon="🇮🇳")

# Fresh deploy (e.g. Hugging Face): build the vector DB once at startup.
if not os.path.exists("chroma_db"):
    with st.spinner("First start: preparing the scheme documents (about a minute)..."):
        subprocess.run([sys.executable, "ingest_all.py"], check=True)

from agent import agent, to_text  # must come after the DB exists



st.title("🇮🇳 Sarkari Saathi")
st.caption("Find government schemes you qualify for, in Hindi or English.")

# ---------- session state ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []   # [{"role", "content", "steps"}]

# ---------- sidebar ----------
EXAMPLES = [
    "I am a farmer. Am I eligible for PM-KISAN?",
    "मैं एक छात्र हूँ, क्या मुझे छात्रवृत्ति मिल सकती है?",
    "What documents do I need for Ayushman Bharat?",
]
with st.sidebar:
    st.header("Try asking")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()
    st.caption("Eligibility is decided by fixed rules, not by the AI. "
               "Final decisions rest with the official authorities.")

# ---------- show past messages ----------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        if m.get("steps"):
            with st.expander("🔍 Agent steps"):
                for s in m["steps"]:
                    st.write(s)
        st.markdown(m["content"])


def run_agent(prompt: str):
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    steps, final = [], ""
    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        try:
            for step in agent.stream({"messages": [("user", prompt)]}, config,
                                     stream_mode="updates"):
                for node, update in step.items():
                    if not update:
                        continue
                    for msg in update.get("messages", []):
                        calls = getattr(msg, "tool_calls", None)
                        if calls:
                            for c in calls:
                                line = f"🔧 **Tool call:** `{c['name']}` with `{c['args']}`"
                                steps.append(line)
                                status.write(line)
                        elif node == "tools":
                            txt = to_text(msg.content).replace("\n", " ")[:220]
                            line = f"👀 **Observed:** {txt}..."
                            steps.append(line)
                            status.write(line)
                        else:
                            final = to_text(msg.content)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as e:  # never crash the demo
            status.update(label="Something went wrong", state="error")
            final = ("Sorry, I hit a problem reaching the AI service. "
                     "Please try again in a moment.")
            steps.append(f"⚠️ {type(e).__name__}")
        st.markdown(final)
    st.session_state.history.append(
        {"role": "assistant", "content": final, "steps": steps})


# ---------- input ----------
prompt = st.chat_input("Ask in Hindi or English...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    run_agent(prompt)
