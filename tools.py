"""
Hour 6: the agent's tools (matched to YOUR eligibility.py and ask_user.py).

  search_docs               -> RAG over the scheme PDFs (optionally scoped to one scheme)
  check_scheme_eligibility  -> runs your RULES and returns ONE next question to ask

Test both without any agent:  python tools.py
"""
import json
from typing import Optional

from langchain_core.tools import tool
from langchain_chroma import Chroma

from rag import emb, DB_DIR
from eligibility import check_eligibility
from ask_user import ask_user

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
    owns_land: Optional[bool] = None,
    pays_income_tax: Optional[bool] = None,
    occupation: Optional[str] = None,
    monthly_pension: Optional[float] = None,
    age: Optional[int] = None,
    enrolled_in_college: Optional[bool] = None,
    family_income: Optional[float] = None,
    in_eligible_list: Optional[bool] = None,
    lang: str = "en",
) -> str:
    """Check whether the user is eligible for ONE scheme using fixed rules.
    scheme must be 'pm kisan', 'scholarship' or 'ayushman yojana'.
    Pass ONLY facts the user actually told you; leave everything else empty.
    NEVER guess a value. lang is 'hi' if the user writes Hindi, else 'en'.
    If verdict is 'need_more_info', ask the user the single question in
    'next_question', wait for the answer, then call this tool again with
    everything learned so far."""
    args = dict(locals())
    lang = args.pop("lang")
    scheme = args.pop("scheme")
    user = {k: v for k, v in args.items() if v is not None}

    result = check_eligibility(scheme, user)
    if "error" not in result:
        nxt = ask_user(result["missing_fields"], lang)
        if nxt["field"] is None and result["missing_fields"]:
            # field has no entry in FIELD_QUESTIONS yet: add one in ask_user.py
            f = result["missing_fields"][0]
            nxt = {"field": f, "question": f"Please tell me: {f}"}
        result["next_question"] = nxt
    return json.dumps(result, ensure_ascii=False)


TOOLS = [search_docs, check_scheme_eligibility]


if __name__ == "__main__":
    print("--- search_docs (scoped to pm kisan) ---")
    print(search_docs.invoke({"query": "who is excluded", "scheme": "pm kisan"})[:500])

    print("\n--- partial info: should return need_more_info + next_question ---")
    print(check_scheme_eligibility.invoke(
        {"scheme": "pm kisan", "owns_land": True}))

    print("\n--- tax payer: should be not_eligible ---")
    print(check_scheme_eligibility.invoke(
        {"scheme": "pm kisan", "owns_land": True, "pays_income_tax": True,
         "occupation": "farmer", "monthly_pension": 0}))

    print("\n--- Hindi question for missing info ---")
    print(check_scheme_eligibility.invoke(
        {"scheme": "pm kisan", "owns_land": True, "lang": "hi"}))
