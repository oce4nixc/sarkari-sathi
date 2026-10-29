"""
Hour 4: score your RAG pipeline on tests.json.

Each test: {"q": question, "scheme": "pm kisan", "any_of": ["6,000", "6000"]}
  - Retrieval PASS: a chunk from the right scheme is in the top 4
  - Answer PASS:    the answer contains at least one keyword from any_of
                    (for "not in the documents" tests use scheme "none" and
                     any_of like ["could not find", "not find", "नहीं"])

Run:  python eval.py
"""
import json
import time

from rag import emb, llm, DB_DIR
from langchain_chroma import Chroma

PAUSE = 4  # seconds between questions (free-tier rate limits)

db = Chroma(persist_directory=DB_DIR, embedding_function=emb)


def answer(question: str):
    hits = db.similarity_search(question, k=4)
    context = "\n\n".join(
        f"[{h.metadata.get('source')} p.{h.metadata.get('page', 0) + 1}]\n{h.page_content}"
        for h in hits
    )
    prompt = (
        "You answer questions about Indian government schemes using ONLY the "
        "context below. Cite the source and page like [file p.3]. If the answer "
        "is not in the context, say you could not find it. Reply in the same "
        "language as the question.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    )
    content = llm.invoke(prompt).content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return content, hits


with open("tests.json", encoding="utf-8") as f:
    tests = json.load(f)

retr_ok = ans_ok = 0
failures = []

for i, t in enumerate(tests, 1):
    try:
        text, hits = answer(t["q"])
    except Exception as e:  # keep going if one call fails
        print(f"{i:2d}. ERROR  {t['q'][:50]}  ({type(e).__name__})")
        failures.append((t["q"], "error", str(e)[:80]))
        time.sleep(PAUSE)
        continue

    schemes = {h.metadata.get("scheme") for h in hits}
    r = t["scheme"] == "none" or t["scheme"] in schemes
    a = any(k.lower() in text.lower() for k in t["any_of"])
    retr_ok += r
    ans_ok += a
    print(f"{i:2d}. retrieval={'OK ' if r else 'MISS'}  answer={'OK ' if a else 'MISS'}  {t['q'][:55]}")
    if not (r and a):
        failures.append((t["q"], "retrieval" if not r else "answer", text[:150].replace("\n", " ")))
    time.sleep(PAUSE)

n = len(tests)
print(f"\nRetrieval accuracy: {retr_ok}/{n} = {100 * retr_ok / n:.0f}%")
print(f"Answer accuracy:    {ans_ok}/{n} = {100 * ans_ok / n:.0f}%")
if failures:
    print("\nFailures to investigate:")
    for q, kind, info in failures:
        print(f" - [{kind}] {q}\n     -> {info}")
