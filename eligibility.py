"""Hour 5: check_eligibility - the first real agent tool.

Rules are starter values. VERIFY each one against your PDFs before the demo.
"""

EXCLUDED_PROFESSIONS = {"doctor", "engineer", "lawyer", "chartered accountant", "architect"}

# Each rule: (name, fields it needs, function(user) -> bool)
RULES = {
    "pm kisan": [
        ("owns cultivable land", ["owns_land"],
         lambda u: u["owns_land"] is True),
        ("not an income tax payer", ["pays_income_tax"],
         lambda u: u["pays_income_tax"] is False),
        ("not in an excluded profession", ["occupation"],
         lambda u: u["occupation"].lower() not in EXCLUDED_PROFESSIONS),
        ("not a high-pension retiree", ["monthly_pension"],
         lambda u: u["monthly_pension"] < 10000),
    ],
    "scholarship": [
        ("age 18-25", ["age"],
         lambda u: 18 <= u["age"] <= 25),
        ("enrolled in college", ["enrolled_in_college"],
         lambda u: u["enrolled_in_college"] is True),
        ("family income below Rs 4.5 lakh", ["family_income"],
         lambda u: u["family_income"] < 450000),
    ],
    "ayushman yojana": [
        ("age 70+ or in eligible household list", ["age"],
         lambda u: u["age"] >= 70 or u.get("in_eligible_list") is True),
    ],
}


def check_eligibility(scheme: str, user: dict) -> dict:
    """Return which rules passed/failed and which details are still missing."""
    scheme = scheme.lower().strip()
    if scheme not in RULES:
        return {"error": f"Unknown scheme '{scheme}'. Choose from: {list(RULES)}"}

    passed, failed, missing = [], [], set()
    for name, fields, rule in RULES[scheme]:
        absent = [f for f in fields if user.get(f) is None]
        if absent:
            missing.update(absent)
            continue
        (passed if rule(user) else failed).append(name)
        # Under 70 and list status unknown: ask, don't reject
    if (scheme == "ayushman yojana" and user.get("age") is not None
            and user["age"] < 70 and user.get("in_eligible_list") is None):
        failed, missing = [], {"in_eligible_list"}

    if failed:
        verdict = "not_eligible"
    elif missing:
        verdict = "need_more_info"
    else:
        verdict = "eligible"

    return {"scheme": scheme, "verdict": verdict, "passed": passed,
            "failed": failed, "missing_fields": sorted(missing)}


if __name__ == "__main__":
    print(check_eligibility("pm kisan", {"owns_land": True, "pays_income_tax": False,
                                         "occupation": "farmer", "monthly_pension": 0}))
    print(check_eligibility("pm kisan", {"owns_land": True, "pays_income_tax": True,
                                         "occupation": "doctor", "monthly_pension": 0}))
    print(check_eligibility("scholarship", {"age": 20}))
    print(check_eligibility("ayushman yojana", {"age": 72}))
