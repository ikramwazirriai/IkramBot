import faiss
import numpy as np


def create_faiss_index(embeddings):
    """
    Create a FAISS index using cosine similarity.
    """

    # Convert to float32 because FAISS expects float32
    embeddings = np.asarray(embeddings).astype("float32")

    # Normalize vectors
    faiss.normalize_L2(embeddings)

    # Number of dimensions in each embedding
    dimension = embeddings.shape[1]

    # Inner product on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    return index