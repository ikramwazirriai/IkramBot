import faiss
import pickle
from embeddings import load_embedding_model


INDEX_PATH = "vectorstore/ikram.index"
METADATA_PATH = "vectorstore/metadata.pkl"


def search(query, top_k=5):

    # ---------------------------------------------
    # Load FAISS index
    # ---------------------------------------------

    index = faiss.read_index(INDEX_PATH)

    # ---------------------------------------------
    # Load chunks and metadata
    # ---------------------------------------------

    with open(METADATA_PATH, "rb") as file:
        data = pickle.load(file)

    chunks = data["chunks"]
    metadata = data["metadata"]

    # ---------------------------------------------
    # Load embedding model
    # ---------------------------------------------

    model = load_embedding_model()

    # ---------------------------------------------
    # Convert query into embedding
    # ---------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # ---------------------------------------------
    # Normalize query
    # ---------------------------------------------

    faiss.normalize_L2(query_embedding)

    # ---------------------------------------------
    # Search FAISS
    # ---------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    # ---------------------------------------------
    # Display results
    # ---------------------------------------------

    print("\n")
    print("=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        print(f"\nRESULT {rank}")
        print("-" * 70)

        print("Similarity:", round(float(score), 4))

        print(
            "Source:",
            metadata[index_id]["source"]
        )

        print(
            "Chunk:",
            metadata[index_id]["chunk_number"]
        )

        print("\nText:")
        print(chunks[index_id])


if __name__ == "__main__":

    question = input(
        "\nAsk something about Ikram Ullah: "
    )

    search(question)