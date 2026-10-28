"""
Hour 3: load ALL PDFs in data/, tag every chunk with its scheme, rebuild the DB.

Scheme name comes from the file name:  pm_kisan.pdf -> "pm kisan"
Run:  python ingest_all.py
"""
import glob
import os
import shutil
import time

from rag import emb, DB_DIR  # reuses your embedding setup from rag.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

DATA_DIR = "data"
BATCH = 40          # chunks per embedding call (keeps free-tier limits happy)
PAUSE = 2           # seconds between batches

# Start clean so PM_kisan.pdf is not stored twice
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
db = Chroma(persist_directory=DB_DIR, embedding_function=emb)

pdfs = sorted(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
if not pdfs:
    raise SystemExit("No PDFs found in data/")

for path in pdfs:
    fname = os.path.basename(path)
    scheme = os.path.splitext(fname)[0].replace("_", " ").replace("-", " ").lower()
    pages = PyPDFLoader(path).load()
    chunks = splitter.split_documents(pages)
    for c in chunks:
        c.metadata["source"] = fname
        c.metadata["scheme"] = scheme
    for i in range(0, len(chunks), BATCH):
        db.add_documents(chunks[i : i + BATCH])
        time.sleep(PAUSE)
    print(f"{fname}: {len(pages)} pages -> {len(chunks)} chunks  [scheme: {scheme}]")

print("\nDone. Now sanity-check retrieval:")
tests = [
    "Who is eligible for farmer income support?",
    "scholarship for students",
    "health insurance cover per family",
]
for q in tests:
    hits = db.similarity_search(q, k=3)
    print(f"\nQ: {q}")
    for h in hits:
        print(f"  - [{h.metadata['scheme']}] p.{h.metadata.get('page', 0) + 1}")
