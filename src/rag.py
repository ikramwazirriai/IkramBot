import faiss
import pickle

from embeddings import load_embedding_model
from guard import is_ikram_question, get_rejection_message
from llm import generate_answer


# ==================================================
# FILE PATHS
# ==================================================

INDEX_PATH = "vectorstore/ikram.index"
METADATA_PATH = "vectorstore/metadata.pkl"


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
            INDEX_PATH
        )

        # Load chunks and metadata
        with open(
            METADATA_PATH,
            "rb"
        ) as file:

            self.data = pickle.load(file)

        self.chunks = self.data["chunks"]

        self.metadata = self.data["metadata"]

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

            return f"Ikram Ullah {question}"

        # Otherwise keep the original query
        return question

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

        # ----------------------------------------------
        # 5. COLLECT RELEVANT RESULTS
        # ----------------------------------------------

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            score = float(score)

            # Ignore results below threshold
            if score < similarity_threshold:
                continue

            results.append(
                {
                    "score": score,
                    "text": self.chunks[index_id],
                    "metadata": self.metadata[index_id]
                }
            )

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

            source = result["metadata"]["source"]

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
            similarity_threshold=0.25
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

            source = result["metadata"]["source"]

            if source not in sources:

                sources.append(source)

        # ----------------------------------------------
        # 7. RETURN FINAL RESULT
        # ----------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }