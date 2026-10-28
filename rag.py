"""
Hour 2: RAG over one PDF with Gemini (no torch needed).

Usage:
  python rag.py ingest data/pm_kisan.pdf
  python rag.py ask "Who is eligible for this scheme?"
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # reads GOOGLE_API_KEY from .env

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# If a model name gives a 404 error, check current names in Google AI Studio.
CHAT_MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "models/gemini-embedding-001"
DB_DIR = "chroma_db"

emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0, timeout=60, max_retries=1)


def ingest(pdf_path: str):
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata["source"] = os.path.basename(pdf_path)
    Chroma.from_documents(chunks, emb, persist_directory=DB_DIR)
    print(f"Stored {len(chunks)} chunks from {len(pages)} pages.")


def ask(question: str):
    db = Chroma(persist_directory=DB_DIR, embedding_function=emb)
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
    print("\n--- RETRIEVED ---")
    for h in hits:
        print(f"* {h.metadata.get('source')} p.{h.metadata.get('page', 0) + 1}: "
              f"{h.page_content[:100].strip()!r}...")
    print("\n--- ANSWER ---")
    content = llm.invoke(prompt).content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    print(content)

if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("ingest", "ask"):
        print(__doc__)
        sys.exit(1)
    cmd, arg = sys.argv[1], " ".join(sys.argv[2:])
    ingest(arg) if cmd == "ingest" else ask(arg)
