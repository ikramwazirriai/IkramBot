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
    # RETRIEVE RELEVANT CHUNKS
    # --------------------------------------------------

    def retrieve(
        self,
        question,
        top_k=5,
        similarity_threshold=0.25
    ):

        # Create embedding for user question
        query_embedding = self.embedding_model.encode(
            [question],
            convert_to_numpy=True
        )

        # Normalize embedding
        faiss.normalize_L2(
            query_embedding
        )

        # Search FAISS
        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        # Process search results
        for score, index_id in zip(
            scores[0],
            indices[0]
        ):

            score = float(score)

            # Ignore weakly related results
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
        # 5. SEND CONTEXT TO GEMINI
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
        # 7. RETURN RESULT
        # ----------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }