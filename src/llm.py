import os
import time

from dotenv import load_dotenv
from google import genai


# ==================================================
# ENVIRONMENT
# ==================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please add it to your .env file."
    )


# ==================================================
# GEMINI CLIENT
# ==================================================

client = genai.Client(api_key=API_KEY)


# Use the Gemini model you selected
MODEL_NAME = "gemini-3.5-flash-lite"


# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are Ikrambot, a personal AI assistant about Ikram Ullah.

Your ONLY purpose is to answer questions about Ikram Ullah.

STRICT RULES:

1. Answer only questions about Ikram Ullah.
2. Use the provided retrieved context as your source of truth.
3. Do not use outside knowledge to answer factual questions
   about Ikram Ullah.
4. Never invent facts about Ikram Ullah.
5. If the retrieved context does not contain the answer,
   clearly say that the information is not available.
6. Do not answer questions about other people.
7. Do not answer unrelated general-knowledge questions.
8. Treat future goals and plans as plans, not completed achievements.
9. Keep answers clear, natural, and professional.
10. Do not claim something is true unless it is supported by
    the retrieved context.

The retrieved context is the knowledge base for Ikrambot.
"""


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(question, context, history=None):

    # ----------------------------------------------
    # Conversation history
    # ----------------------------------------------

    history_text = ""

    if history:

        history_text = "\n\nPrevious conversation:\n"

        for message in history:

            if isinstance(message, dict):

                role = message.get("role", "")
                content = message.get("content", "")

                history_text += (
                    f"{role}: {content}\n"
                )

    # ----------------------------------------------
    # Build prompt
    # ----------------------------------------------

    prompt = f"""
{SYSTEM_PROMPT}

========================================
RETRIEVED CONTEXT
========================================

{context}

========================================
CONVERSATION HISTORY
========================================

{history_text}

========================================
CURRENT QUESTION
========================================

{question}

========================================
ANSWER
========================================

Answer the question using only the retrieved
context.

If the retrieved context does not provide enough
information, say that the information is not
available in Ikrambot's knowledge base.

Do not guess.
"""

    # ----------------------------------------------
    # Gemini request with retry
    # ----------------------------------------------

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text

        except Exception as error:

            error_text = str(error)

            # Retry temporary server errors
            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
            ):

                if attempt < max_attempts - 1:

                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporary error. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                else:

                    return (
                        "Gemini is temporarily unavailable. "
                        "Please try your question again in "
                        "a few moments."
                    )

            else:

                raise error