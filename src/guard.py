import re


# --------------------------------------------------
# IKRAM IDENTIFIERS
# --------------------------------------------------

IKRAM_NAMES = [
    "ikram",
    "ikram ullah",
    "ikramullah",
    "ikram wazir",
    "ikram ullah wazir"
]


# --------------------------------------------------
# PRONOUN / FOLLOW-UP REFERENCES
# --------------------------------------------------

IKRAM_CONTEXT_WORDS = [
    "his",
    "him",
    "he",
    "he's",
    "himself",
    "ikram's",
    "ikram’s",
    "the person",
    "this person"
]


# --------------------------------------------------
# CHECK FOR IKRAM NAME
# --------------------------------------------------

def contains_ikram_name(text):

    text = text.lower()

    for name in IKRAM_NAMES:

        if name in text:
            return True

    return False


# --------------------------------------------------
# CHECK FOR IKRAM REFERENCE
# --------------------------------------------------

def contains_ikram_reference(text):

    text = text.lower()

    for word in IKRAM_CONTEXT_WORDS:

        pattern = rf"\b{re.escape(word)}\b"

        if re.search(pattern, text):
            return True

    return False


# --------------------------------------------------
# CONVERT HISTORY TO TEXT
# --------------------------------------------------

def history_to_text(conversation_history):

    if not conversation_history:
        return ""

    history_parts = []

    for message in conversation_history:

        # Streamlit / RAG history format
        if isinstance(message, dict):

            content = message.get("content", "")

            if content:
                history_parts.append(str(content))

        # Also support normal strings
        else:

            history_parts.append(str(message))

    return " ".join(history_parts).lower()


# --------------------------------------------------
# CHECK WHETHER QUESTION IS ABOUT IKRAM
# --------------------------------------------------

def is_ikram_question(question, conversation_history=None):

    # Directly mentions Ikram
    if contains_ikram_name(question):
        return True

    # Follow-up question
    if conversation_history:

        if contains_ikram_reference(question):

            previous_text = history_to_text(
                conversation_history
            )

            if contains_ikram_name(previous_text):
                return True

    return False


# --------------------------------------------------
# OUT-OF-SCOPE MESSAGE
# --------------------------------------------------

def get_rejection_message():

    return (
        "I don't know about that. "
        "I can only answer questions about Ikram Ullah. "
        "Please ask me something about Ikram Ullah."
    )