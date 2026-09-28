import os
import time

from dotenv import load_dotenv
from google import genai


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


# ==================================================
# CHECK API KEY
# ==================================================

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please add it to your .env file."
    )


# ==================================================
# GEMINI CLIENT
# ==================================================

client = genai.Client(
    api_key=API_KEY
)


# ==================================================
# GEMINI MODEL
# ==================================================

MODEL_NAME = "gemini-3.5-flash-lite"


# ==================================================
# SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are Ikrambot, a personal AI assistant about Ikram Ullah.

Your ONLY purpose is to answer questions about Ikram Ullah.

STRICT RULES:

1. Answer only questions about Ikram Ullah.

2. Use the provided retrieved context as your
   primary source of truth.

3. Do not use outside knowledge to answer factual
   questions about Ikram Ullah.

4. Never invent facts about Ikram Ullah.

5. If the retrieved context does not contain the answer,
   clearly say that the information is not available.

6. Do not answer questions about other people.

7. Do not answer unrelated general-knowledge questions.

8. Treat future goals and plans as plans, not completed
   achievements.

9. Keep answers clear, natural, and professional.

10. Do not claim something is true unless it is supported
    by the retrieved context.

11. Use conversation history to understand follow-up
    references such as:
    - he
    - him
    - his
    - this
    - that
    - these
    - those

12. Even when the user uses a short follow-up question,
    answer it according to the established conversation
    about Ikram Ullah.

The retrieved context is the knowledge base for Ikrambot.
"""


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(
    question,
    context,
    history=None
):

    # --------------------------------------------------
    # CONVERSATION HISTORY
    # --------------------------------------------------

    history_text = ""

    if history:

        history_parts = []

        for message in history:

            if isinstance(message, dict):

                role = message.get(
                    "role",
                    ""
                )

                content = message.get(
                    "content",
                    ""
                )

                if content:

                    history_parts.append(
                        f"{role.upper()}: {content}"
                    )

        history_text = "\n\n".join(
            history_parts
        )


    # --------------------------------------------------
    # PROMPT
    # --------------------------------------------------

    prompt = f"""
{SYSTEM_PROMPT}

========================================
RETRIEVED CONTEXT
========================================

{context}

========================================
PREVIOUS CONVERSATION
========================================

{history_text}

========================================
CURRENT QUESTION
========================================

{question}

========================================
INSTRUCTIONS
========================================

Use the retrieved context to answer the
current question.

Use the previous conversation to understand
references such as:

- he
- him
- his
- this
- that
- these
- those

Because Ikrambot is a personal assistant
about Ikram Ullah, interpret an ambiguous
personal reference according to the established
conversation context.

However:

- Use the retrieved context as the factual source.
- Do not invent information.
- Do not answer unrelated questions.
- Do not answer questions about other people.
- Do not turn future plans into completed achievements.
- If the retrieved context does not contain the answer,
  clearly say that the information is not available.

========================================
ANSWER
========================================
"""


    # --------------------------------------------------
    # GEMINI REQUEST WITH RETRY
    # --------------------------------------------------

    max_retries = 3

    for attempt in range(max_retries):

        try:

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )

            return response.text.strip()


        except Exception as e:

            error_text = str(e)

            print(
                f"Gemini error "
                f"(attempt {attempt + 1}/{max_retries}):"
            )

            print(error_text)


            # Retry temporary server/rate errors
            if (
                "503" in error_text
                or "429" in error_text
                or "UNAVAILABLE" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                if attempt < max_retries - 1:

                    wait_time = 2 ** attempt

                    print(
                        f"Retrying in "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )

                    continue


            # Other errors should not be hidden
            return (
                "Sorry, I encountered an error "
                "while generating the answer."
            )


    return (
        "Sorry, I could not generate an answer "
        "at this time."
    )