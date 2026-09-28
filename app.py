import streamlit as st
import sys
from pathlib import Path


# ==================================================
# PATH SETUP
# ==================================================

SRC_DIR = Path(__file__).parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from rag import IkrambotRAG


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Ikrambot",
    page_icon="🤖",
    layout="centered"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #666;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🤖 Ikrambot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Personal AI Assistant for Ikram Wazir'
    '</div>',
    unsafe_allow_html=True
)


# ==================================================
# LOAD RAG SYSTEM
# ==================================================

@st.cache_resource
def load_chatbot():
    return IkrambotRAG()


try:
    chatbot = load_chatbot()

except Exception as e:

    st.error("Unable to load Ikrambot.")
    st.exception(e)
    st.stop()


# ==================================================
# SESSION STATE
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("Ikrambot")

    st.write(
        """
        Ikrambot is a personal RAG-based AI assistant
        that answers questions about **Ikram Wazir**.
        """
    )

    st.divider()

    st.subheader("Knowledge Areas")

    st.write("🎓 Education")
    st.write("💻 Skills")
    st.write("📚 Courses")
    st.write("🚀 Projects")
    st.write("💼 Career Goals")
    st.write("📄 CV & Cover Letter")
    st.write("🏢 Business Interests")
    st.write("🌍 Personal Background")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ==================================================
# QUESTION SUGGESTIONS
# ==================================================

SUGGESTION_QUESTIONS = [
    "Who is Ikram Ullah?",
    "Tell me about Ikram Ullah.",
    "Where was Ikram Ullah born?",
    "When was Ikram Ullah born?",
    "Tell me about Ikram's background.",
    "Where does Ikram study?",
    "Which university does Ikram attend?",
    "What degree is Ikram pursuing?",
    "What is Ikram's field of study?",
    "Which courses has Ikram studied?",
    "What AI courses has Ikram studied?",
    "What programming courses has Ikram studied?",
    "What machine learning courses has Ikram studied?",
    "What deep learning courses has Ikram studied?",
    "What NLP courses has Ikram studied?",
    "Has Ikram studied generative AI?",
    "What are Ikram's technical skills?",
    "What programming languages does Ikram know?",
    "What AI skills does Ikram have?",
    "What machine learning skills does Ikram have?",
    "What deep learning skills does Ikram have?",
    "What NLP skills does Ikram have?",
    "What tools and technologies does Ikram use?",
    "What projects has Ikram worked on?",
    "Tell me about Ikram's AI Resume Screening project.",
    "What is Ikram's Formula 1 project?",
    "Tell me about Ikram's NLP project.",
    "What robotics projects has Ikram worked on?",
    "What is Ikrambot?",
    "What technologies has Ikram used in his projects?",
    "What are Ikram's career goals?",
    "What are Ikram's future plans?",
    "What does Ikram want to learn?",
    "What type of internship is Ikram looking for?",
    "What are Ikram's AI career interests?",
    "What businesses is Ikram interested in?",
    "Tell me about Waziri Socks.",
    "What business ideas has Ikram explored?",
    "Tell me about Ikram's CV.",
    "Tell me about Ikram's professional experience.",
    "What is Ikram's professional profile?"
]


# ==================================================
# WELCOME MESSAGE
# ==================================================


if not st.session_state.messages:

    st.info(
        """
        👋 **Welcome to Ikrambot!**

        I can answer questions about **Ikram Wazir**
        using his personal knowledge base.

        You can ask about his education, courses,
        skills, projects, career goals, CV,
        business interests, and personal background.
        """
    )


# ==================================================
# DISPLAY PREVIOUS MESSAGES
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"• {source}"
                    )


# ==================================================
# CHAT INPUT
# ==================================================

question = st.chat_input(
    "Ask something about Ikram Wazir..."
)


# ==================================================
# PROCESS QUESTION
# ==================================================


if question:

    question = question.strip()


    if question:

        # ------------------------------------------
        # USER MESSAGE
        # ------------------------------------------

        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ------------------------------------------
        # GENERATE ANSWER
        # ------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Thinking..."
            ):

                try:

                    result = chatbot.ask(
                        question,
                        history=st.session_state.messages
                    )

                    answer = result["answer"]

                    sources = result["sources"]

                except Exception as e:

                    answer = (
                        "Sorry, I encountered "
                        "an error while processing "
                        "your question."
                    )

                    sources = []

                    st.error(
                        str(e)
                    )


            # --------------------------------------
            # ANSWER
            # --------------------------------------

            st.markdown(
                answer
            )


            # --------------------------------------
            # SOURCES
            # --------------------------------------

            if sources:

                with st.expander(
                    "📚 Sources"
                ):

                    for source in sources:

                        st.write(
                            f"• {source}"
                        )


        # ------------------------------------------
        # SAVE ASSISTANT MESSAGE
        # ------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources
            }
        )


        # ------------------------------------------
        # RERUN
        # ------------------------------------------

        st.rerun()