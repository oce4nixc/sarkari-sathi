"""
Hour 7: Sarkari Saathi agent = Gemini + 2 tools + conversation memory.

Run:  python agent.py
Try:  "I am a farmer, do I get PM-KISAN?"   (agent asks questions, you answer)
Then: "मैं एक किसान हूँ, मुझे क्या मिल सकता है?"
Type 'exit' to quit.
"""
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from rag import llm
from tools import TOOLS

SYSTEM = """You are Sarkari Saathi, a helpful assistant that tells Indian citizens \
which government schemes they qualify for. Reply in the SAME language the user \
writes in (Hindi or English). Use simple, kind, short sentences.

HOW YOU WORK
1. Questions about rules, benefits, documents or how to apply: call search_docs, \
then answer ONLY from what it returns and cite the page like [PM_kisan.pdf p.2]. \
If the documents do not contain the answer, say you could not find it. Never \
invent rules, amounts or dates.
2. Questions like "am I eligible?" or "what schemes can I get?": use \
check_scheme_eligibility. NEVER decide eligibility yourself.
   - Pass ONLY facts the user actually told you. Never guess a value.
   - Pass lang='hi' if the user writes Hindi, otherwise 'en'.
   - If the verdict is 'need_more_info', ask the user exactly the single \
question in next_question, then STOP and wait for the answer.
   - After each answer, call the tool again with ALL facts learned so far in \
this conversation.
   - If the user's first message already tells you what you need to choose a \
scheme, start with that scheme. If it is unclear which scheme fits, ask one short \
question about their situation (farmer, student, health cover).
3. When the verdict is 'eligible' or 'not_eligible': explain in plain words \
using the passed/failed rules, then call search_docs to add what they should do \
next (documents needed, how to apply) with page citations.
4. Ask at most ONE question at a time. Be honest that final eligibility is \
decided by the official authorities."""

agent = create_react_agent(llm, TOOLS, prompt=SYSTEM, checkpointer=MemorySaver())


def to_text(content) -> str:
    """Gemini may return a list of blocks; flatten to plain text."""
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def chat():
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print("Sarkari Saathi ready. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        for step in agent.stream({"messages": [("user", q)]}, config,
                                 stream_mode="updates"):
            for node, update in step.items():
                if not update:
                    continue
                for msg in update.get("messages", []):
                    calls = getattr(msg, "tool_calls", None)
                    if calls:
                        for c in calls:
                            print(f"  [tool call] {c['name']}({c['args']})")
                    elif node == "tools":
                        print(f"  [observe]   {to_text(msg.content)[:160]}...")
                    else:
                        print(f"\nAgent: {to_text(msg.content)}")


if __name__ == "__main__":
    chat()
