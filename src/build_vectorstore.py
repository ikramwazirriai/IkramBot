from document_loader import load_all_pdfs
from preprocessing import clean_text
from chunking import split_text
from embeddings import load_embedding_model, create_embeddings
from vector_store import create_faiss_index


def main():

    print("=" * 60)
    print("IKRAMBOT VECTOR DATABASE BUILDER")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load PDFs
    # --------------------------------------------------

    print("\n[1] Loading documents...")

    documents = load_all_pdfs()

    print(f"Documents loaded: {len(documents)}")

    # --------------------------------------------------
    # 2. Clean and chunk documents
    # --------------------------------------------------

    print("\n[2] Cleaning and chunking documents...")

    all_chunks = []
    all_metadata = []

    for document in documents:

        source = document["source"]

        cleaned_text = clean_text(document["text"])

        chunks = split_text(
            cleaned_text,
            chunk_size=800,
            chunk_overlap=150
        )

        for chunk_number, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            all_metadata.append(
                {
                    "person": "Ikram Ullah",
                    "source": source,
                    "chunk_number": chunk_number
                }
            )

    print(f"Total chunks created: {len(all_chunks)}")

    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    print("\n[3] Loading embedding model...")

    model = load_embedding_model()

    # --------------------------------------------------
    # 4. Create embeddings
    # --------------------------------------------------

    print("\n[4] Creating embeddings...")

    embeddings = create_embeddings(
        model,
        all_chunks
    )

    print("Embedding shape:", embeddings.shape)

    # --------------------------------------------------
    # 5. Create FAISS index
    # --------------------------------------------------

    print("\n[5] Creating FAISS index...")

    index = create_faiss_index(embeddings)

    print("FAISS index created.")
    print("Vectors stored:", index.ntotal)

    # --------------------------------------------------
    # 6. Save everything
    # --------------------------------------------------

    print("\n[6] Saving vector database...")

    import os
    import pickle

    os.makedirs("vectorstore", exist_ok=True)

    # Save FAISS index
    import faiss

    faiss.write_index(
        index,
        "vectorstore/ikram.index"
    )

    # Save chunks and metadata
    with open(
        "vectorstore/metadata.pkl",
        "wb"
    ) as file:

        pickle.dump(
            {
                "chunks": all_chunks,
                "metadata": all_metadata
            },
            file
        )

    print("\nVector database saved successfully.")

    print("\nFiles created:")

    print("vectorstore/ikram.index")
    print("vectorstore/metadata.pkl")

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
