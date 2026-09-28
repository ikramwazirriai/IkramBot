import streamlit as st
import sys
from pathlib import Path

# Add src folder to Python path
SRC_DIR = Path(__file__).parent / "src"
sys.path.append(str(SRC_DIR))

from rag import IkrambotRAG


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Ikrambot",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

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

    .source-box {
        background-color: #f5f5f5;
        padding: 10px 15px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🤖 Ikrambot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Personal AI Assistant for Ikram Ullah'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# LOAD RAG SYSTEM
# --------------------------------------------------

@st.cache_resource
def load_chatbot():
    return IkrambotRAG()


try:
    chatbot = load_chatbot()

except Exception as e:
    st.error("Unable to load Ikrambot.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("Ikrambot")

    st.write(
        """
        Ikrambot is a personal RAG-based AI assistant
        that answers questions about **Ikram Ullah**.
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

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# WELCOME MESSAGE
# --------------------------------------------------

if not st.session_state.messages:

    st.info(
        """
        👋 **Welcome to Ikrambot!**

        I can answer questions about **Ikram Ullah**
        using his personal knowledge base.

        **Try asking:**

        • Who is Ikram Ullah?  
        • Where does Ikram study?  
        • What projects has Ikram worked on?  
        • What are Ikram's skills?  
        • What are Ikram's career goals?
        """
    )


# --------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Show sources for assistant messages
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources"):

                for source in message["sources"]:
                    st.write(f"• {source}")


# --------------------------------------------------
# CHAT INPUT
# --------------------------------------------------

question = st.chat_input(
    "Ask something about Ikram Ullah..."
)


# --------------------------------------------------
# PROCESS QUESTION
# --------------------------------------------------

if question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                result = chatbot.ask(
                    question,
                    history=st.session_state.messages
                )

                answer = result["answer"]
                sources = result["sources"]

            except Exception as e:

                answer = (
                    "Sorry, I encountered an error "
                    "while processing your question."
                )

                sources = []

                st.error(str(e))

        st.markdown(answer)

        # Show sources
        if sources:

            with st.expander("📚 Sources"):

                for source in sources:
                    st.write(f"• {source}")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )