"""Hour 6: ask_user - the agent asks for missing details, one question at a time."""

FIELD_QUESTIONS = {
    "owns_land": (
        "Does your family own cultivable agricultural land?",
        "क्या आपके परिवार के नाम पर खेती की ज़मीन है?",
    ),
    "pays_income_tax": (
        "Do you pay income tax?",
        "क्या आप आयकर (इनकम टैक्स) भरते हैं?",
    ),
    "occupation": (
        "What is your occupation? (e.g. farmer, doctor, teacher)",
        "आप क्या काम करते हैं? (जैसे किसान, डॉक्टर, शिक्षक)",
    ),
    "monthly_pension": (
        "What is your monthly pension in rupees? (enter 0 if none)",
        "आपकी मासिक पेंशन कितनी है? (नहीं है तो 0 लिखें)",
    ),
    "age": (
        "How old are you?",
        "आपकी उम्र कितनी है?",
    ),
    "enrolled_in_college": (
        "Are you currently enrolled in college?",
        "क्या आप अभी कॉलेज में पढ़ रहे हैं?",
    ),
    "family_income": (
        "What is your family's yearly income in rupees?",
        "आपके परिवार की सालाना आय कितनी है (रुपये में)?",
    ),
    "in_eligible_list": (
        "Is your family on the Ayushman Bharat eligible list?",
        "क्या आपके परिवार का नाम आयुष्मान की पात्र सूची में है?",
    ),
}


def detect_lang(text: str) -> str:
    """'hi' if the text contains Devanagari characters, else 'en'."""
    return "hi" if any("\u0900" <= ch <= "\u097F" for ch in text) else "en"


def ask_user(missing_fields: list, lang: str = "en") -> dict:
    """Pick the next missing field and return the question to show the user."""
    for field in missing_fields:
        if field in FIELD_QUESTIONS:
            en, hi = FIELD_QUESTIONS[field]
            return {"field": field, "question": hi if lang == "hi" else en}
    return {"field": None, "question": None}


if __name__ == "__main__":
    print(ask_user(["enrolled_in_college", "family_income"], "en"))
    print(ask_user(["pays_income_tax"], detect_lang("क्या मैं पात्र हूँ?")))
    print(ask_user([]))
