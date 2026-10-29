from pathlib import Path
import re

import faiss
import pickle

from embeddings import load_embedding_model
from guard import is_ikram_question, get_rejection_message
from llm import generate_answer


# ==================================================
# FILE PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_DIR / "vectorstore" / "ikram.index"
METADATA_PATH = PROJECT_DIR / "vectorstore" / "metadata.pkl"


QUERY_EXPANSIONS = (
    (
        ("study", "studies", "studying", "university", "college", "degree", "education", "attend"),
        "university education college degree institution Lahore UMT",
    ),
    (
        ("born", "birth", "birthplace"),
        "birthplace village hometown",
    ),
    (
        ("programming language", "programming languages"),
        "programming language Python coding",
    ),
)

STOP_WORDS = {
    "a", "about", "after", "all", "an", "and", "are", "as", "at", "be",
    "been", "before", "being", "but", "by", "can", "could", "did", "do",
    "does", "for", "from", "he", "her", "him", "his", "how", "i", "in",
    "is", "it", "me", "of", "on", "or", "she", "tell", "that", "the",
    "their", "them", "they", "this", "to", "uh", "ullah", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "with", "would",
    "you", "your", "ikram", "know", "knows",
}


def format_source_reference(result):
    source = result["metadata"]["source"]
    pages = result["metadata"].get("pages") or sorted(
        {
            int(page)
            for page in re.findall(r"--- Page (\d+) ---", result["text"])
        }
    )

    if not pages:
        return source

    page_label = "page" if len(pages) == 1 else "pages"
    page_numbers = ", ".join(str(page) for page in pages)
    return f"{source} ({page_label} {page_numbers})"


def get_query_keywords(text):
    keywords = set()

    for token in re.findall(r"[a-z0-9]+", text.lower()):
        if token in STOP_WORDS:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = f"{token[:-3]}y"
        elif token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        keywords.add(token)

    return keywords


# ==================================================
# IKRAMBOT RAG CLASS
# ==================================================

class IkrambotRAG:

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def __init__(self):

        print("Loading Ikrambot RAG system...")

        # Load FAISS index
        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        # Load chunks and metadata
        with open(
            METADATA_PATH,
            "rb"
        ) as file:

            self.data = pickle.load(file)

        self.chunks = self.data["chunks"]

        self.metadata = self.data["metadata"]

        pages_by_source = {}
        for chunk, metadata in zip(self.chunks, self.metadata):
            source = metadata["source"]
            pages = [
                int(page)
                for page in re.findall(r"--- Page (\d+) ---", chunk)
            ]
            if pages:
                pages_by_source[source] = pages[-1]
            inherited_page = pages_by_source.get(source)
            metadata["pages"] = sorted(
                set(pages or ([inherited_page] if inherited_page else []))
            )

        # Load embedding model
        self.embedding_model = (
            load_embedding_model()
        )

        print("Ikrambot RAG system ready.")

    # --------------------------------------------------
    # PREPARE RETRIEVAL QUERY
    # --------------------------------------------------

    def prepare_retrieval_query(
        self,
        question,
        history=None
    ):
        """
        Improve short or pronoun-based questions
        before sending them to the embedding model.

        Example:

            Original:
            "Where does he study?"

            Retrieval query:
            "Ikram Ullah Where does he study?"

        The original question is still sent to Gemini.
        """

        question_lower = question.lower().strip()

        # Pronouns that may refer to Ikram
        pronouns = [
            "he",
            "him",
            "his",
            "himself"
        ]

        words = question_lower.split()

        # Check whether the question contains
        # a personal pronoun
        has_pronoun = any(
            word in words
            for word in pronouns
        )

        # If a pronoun is present and Ikram is not
        # explicitly mentioned, add Ikram Ullah
        # to improve semantic retrieval.
        if (
            has_pronoun
            and "ikram" not in question_lower
        ):

            question = f"Ikram Ullah {question}"

        query_lower = question.lower()
        expansions = [
            expansion
            for terms, expansion in QUERY_EXPANSIONS
            if any(term in query_lower for term in terms)
        ]

        return " ".join([question, *expansions])

    # --------------------------------------------------
    # RETRIEVE RELEVANT CHUNKS
    # --------------------------------------------------

    def retrieve(
        self,
        question,
        history=None,
        top_k=5,
        similarity_threshold=0.25
    ):
        """
        Retrieve relevant chunks from FAISS.
        """

        # ----------------------------------------------
        # 1. PREPARE BETTER RETRIEVAL QUERY
        # ----------------------------------------------

        retrieval_query = (
            self.prepare_retrieval_query(
                question,
                history
            )
        )

        print(
            "Original question:",
            question
        )

        print(
            "Retrieval query:",
            retrieval_query
        )

        # ----------------------------------------------
        # 2. CREATE QUERY EMBEDDING
        # ----------------------------------------------

        query_embedding = (
            self.embedding_model.encode(
                [retrieval_query],
                convert_to_numpy=True
            )
        )

        # ----------------------------------------------
        # 3. NORMALIZE EMBEDDING
        # ----------------------------------------------

        faiss.normalize_L2(
            query_embedding
        )

        # ----------------------------------------------
        # 4. SEARCH FAISS
        # ----------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        relevance_threshold = max(
            similarity_threshold,
            0.30,
            float(scores[0][0]) * 0.85
        )

        # ----------------------------------------------
        # 5. COLLECT RELEVANT RESULTS
        # ----------------------------------------------

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):
            score = float(score)

            if score < relevance_threshold:
                continue

            results.append(
                {
                    "score": score,
                    "text": self.chunks[index_id],
                    "metadata": self.metadata[index_id]
                }
            )

        question_keywords = get_query_keywords(question)
        if question_keywords and results:
            relevant_results = [
                result
                for result in results
                if len(
                    question_keywords
                    & get_query_keywords(result["text"])
                )
                >= (len(question_keywords) + 1) // 2
            ]
            if relevant_results:
                results = relevant_results

        return results

    # --------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------

    def build_context(
        self,
        results
    ):
        """
        Combine retrieved chunks into a single
        context string for Gemini.
        """

        context_parts = []

        for result in results:

            source = format_source_reference(result)

            text = result["text"]

            context_parts.append(
                f"[Source: {source}]\n{text}"
            )

        return "\n\n".join(
            context_parts
        )

    # --------------------------------------------------
    # ASK IKRAMBOT
    # --------------------------------------------------

    def ask(
        self,
        question,
        history=None,
        top_k=5
    ):
        """
        Main RAG pipeline:

        Question
            ↓
        Guard
            ↓
        Query preparation
            ↓
        Embedding
            ↓
        FAISS retrieval
            ↓
        Context
            ↓
        Gemini
            ↓
        Answer + Sources
        """

        # ----------------------------------------------
        # 1. CHECK QUESTION SCOPE
        # ----------------------------------------------

        if not is_ikram_question(
            question,
            history
        ):

            return {
                "answer": get_rejection_message(),
                "sources": []
            }

        # ----------------------------------------------
        # 2. RETRIEVE RELEVANT INFORMATION
        # ----------------------------------------------

        results = self.retrieve(
            question,
            history=history,
            top_k=top_k,
            similarity_threshold=0.30
        )

        # ----------------------------------------------
        # 3. CHECK WHETHER INFORMATION WAS FOUND
        # ----------------------------------------------

        if not results:

            return {
                "answer": (
                    "I don't have enough information "
                    "about that in Ikram Ullah's "
                    "knowledge base."
                ),
                "sources": []
            }

        # ----------------------------------------------
        # 4. BUILD RETRIEVED CONTEXT
        # ----------------------------------------------

        context = self.build_context(
            results
        )

        # ----------------------------------------------
        # 5. SEND CONTEXT + QUESTION TO GEMINI
        # ----------------------------------------------

        answer = generate_answer(
            question=question,
            context=context,
            history=history
        )

        # ----------------------------------------------
        # 6. COLLECT SOURCES
        # ----------------------------------------------

        sources = []

        for result in results:

            source = format_source_reference(result)

            if source not in sources:

                sources.append(source)

        # ----------------------------------------------
        # 7. RETURN FINAL RESULT
        # ----------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }
