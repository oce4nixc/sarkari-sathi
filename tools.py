"""
Hour 6: the agent's tools.

  search_docs                -> RAG over the scheme PDFs (optionally scoped to one scheme)
  check_scheme_eligibility   -> rule-based check; tells the agent WHAT to ask the user next

Test both without any agent:  python tools.py
"""
import json
from typing import Optional

from langchain_core.tools import tool
from langchain_chroma import Chroma

from rag import emb, DB_DIR
from eligibility import check_eligibility, FIELD_QUESTIONS, SCHEMES

_db = Chroma(persist_directory=DB_DIR, embedding_function=emb)


@tool
def search_docs(query: str, scheme: str = "") -> str:
    """Search the official scheme documents. Use this for any question about
    rules, benefits, documents needed or how to apply. Always cite the page.
    scheme (optional) restricts the search; valid values: 'pm kisan',
    'scholarship', 'ayushman yojana'. Leave empty to search all schemes."""
    kwargs = {"k": 4}
    if scheme.strip():
        kwargs["filter"] = {"scheme": scheme.lower().strip()}
    hits = _db.similarity_search(query, **kwargs)
    if not hits:
        return "No matching text found in the documents."
    return "\n\n".join(
        f"[{h.metadata.get('source')} p.{h.metadata.get('page', 0) + 1}] {h.page_content}"
        for h in hits
    )


@tool
def check_scheme_eligibility(
    scheme: str,
    owns_cultivable_land: Optional[bool] = None,
    is_income_tax_payer: Optional[bool] = None,
    holds_constitutional_post: Optional[bool] = None,
    is_govt_employee: Optional[bool] = None,
    monthly_pension: Optional[float] = None,
    is_registered_professional: Optional[bool] = None,
    is_student: Optional[bool] = None,
    annual_family_income: Optional[float] = None,
    last_exam_percent: Optional[float] = None,
    category: Optional[str] = None,
) -> str:
    """Check whether the user is eligible for ONE scheme using fixed rules.
    scheme must be 'pm kisan', 'scholarship' or 'ayushman yojana'.
    Pass ONLY the facts the user actually told you; leave everything else empty.
    NEVER guess a value. The result lists 'ask_user': questions you must ask
    the user (one or two at a time) before the answer can be final."""
    args = dict(locals())
    profile = {k: v for k, v in args.items() if k != "scheme" and v is not None}

    result = check_eligibility(scheme, profile)
    result["ask_user"] = [
        {"field": f,
         "question_en": FIELD_QUESTIONS[f][0],
         "question_hi": FIELD_QUESTIONS[f][1]}
        for f in result.get("missing", []) if f in FIELD_QUESTIONS
    ]
    return json.dumps(result, ensure_ascii=False)


TOOLS = [search_docs, check_scheme_eligibility]


if __name__ == "__main__":
    print("--- search_docs (scoped to pm kisan) ---")
    print(search_docs.invoke({"query": "who is excluded", "scheme": "pm kisan"})[:500])

    print("\n--- check: partial info (should ask questions) ---")
    print(check_scheme_eligibility.invoke(
        {"scheme": "pm kisan", "owns_cultivable_land": True}))

    print("\n--- check: tax payer (should be NOT eligible) ---")
    print(check_scheme_eligibility.invoke(
        {"scheme": "pm kisan", "owns_cultivable_land": True,
         "is_income_tax_payer": True}))
