import re


IKRAM_NAMES = [
    "ikram",
    "ikram ullah",
    "ikramullah",
    "ikram wazir",
    "ikram ullah wazir"
]


def contains_ikram_name(text):
    text = text.lower()

    for name in IKRAM_NAMES:
        if name in text:
            return True

    return False


def is_clearly_unrelated(question):
    """
    Detect questions that are clearly about another person
    or an unrelated general topic.
    """

    question = question.lower().strip()

    unrelated_patterns = [
        r"\btell me about elon musk\b",
        r"\bwho is elon musk\b",
        r"\btell me about bill gates\b",
        r"\bwho is bill gates\b",
        r"\btell me about donald trump\b",
        r"\bwho is donald trump\b",
        r"\bwhat is the capital of\b",
        r"\bsolve this math\b",
        r"\bwhat is machine learning\b",
        r"\bwhat is artificial intelligence\b",
        r"\bwhat is python\b"
    ]

    for pattern in unrelated_patterns:
        if re.search(pattern, question):
            return True

    return False


def is_ikram_question(question, conversation_history=None):

    # Direct Ikram reference
    if contains_ikram_name(question):
        return True

    # Clearly unrelated question
    if is_clearly_unrelated(question):
        return False

    # If it is not clearly unrelated, allow it to reach RAG.
    #
    # This is important because Ikrambot has only one
    # knowledge domain: Ikram Ullah.
    return True


def get_rejection_message():
    return (
        "I can only answer questions about Ikram Ullah. "
        "Please ask me something related to Ikram Ullah."
    )