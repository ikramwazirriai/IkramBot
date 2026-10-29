import unittest

import numpy as np

from rag import IkrambotRAG, format_source_reference


class FakeEmbeddingModel:
    def encode(self, texts, convert_to_numpy=True):
        return np.array([[1.0, 0.0]], dtype=np.float32)


class FakeIndex:
    def search(self, query_embedding, top_k):
        return (
            np.array([[0.8, 0.72, 0.51]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )


class RagReferenceTests(unittest.TestCase):
    def test_expands_education_queries(self):
        chatbot = object.__new__(IkrambotRAG)

        query = chatbot.prepare_retrieval_query("Where does Ikram study?")

        self.assertIn("university education college", query)

    def test_formats_pdf_page_references(self):
        result = {
            "metadata": {"source": "Introduction_Travel.pdf"},
            "text": "--- Page 2 --- current education --- Page 3 --- university",
        }

        self.assertEqual(
            format_source_reference(result),
            "Introduction_Travel.pdf (pages 2, 3)",
        )

    def test_formats_sources_without_page_markers(self):
        result = {
            "metadata": {"source": "CV.pdf"},
            "text": "Professional summary",
        }

        self.assertEqual(format_source_reference(result), "CV.pdf")

    def test_excludes_low_relevance_chunks(self):
        chatbot = object.__new__(IkrambotRAG)
        chatbot.embedding_model = FakeEmbeddingModel()
        chatbot.index = FakeIndex()
        chatbot.chunks = [
            "He is studying at UMT",
            "He studies at the university",
            "His career goals include an AI job",
        ]
        chatbot.metadata = [
            {"source": "Education.pdf"},
            {"source": "Introduction_Travel.pdf"},
            {"source": "Goals.pdf"},
        ]

        results = chatbot.retrieve("Where does Ikram study?")

        self.assertEqual(
            [result["text"] for result in results],
            [
                "He is studying at UMT",
                "He studies at the university",
            ],
        )


if __name__ == "__main__":
    unittest.main()
