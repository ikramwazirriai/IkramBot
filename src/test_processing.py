from document_loader import load_all_pdfs
from preprocessing import clean_text
from chunking import split_text


def main():

    print("Loading documents...\n")

    documents = load_all_pdfs()

    total_chunks = 0

    for document in documents:

        source = document["source"]
        raw_text = document["text"]

        print("=" * 60)
        print("SOURCE:", source)

        # Step 1: clean
        cleaned_text = clean_text(raw_text)

        print("Original characters:", len(raw_text))
        print("Cleaned characters:", len(cleaned_text))

        # Step 2: chunk
        chunks = split_text(
            cleaned_text,
            chunk_size=800,
            chunk_overlap=150
        )

        print("Number of chunks:", len(chunks))

        total_chunks += len(chunks)

        # Show first chunk
        if chunks:

            print("\nFirst chunk:")
            print("-" * 60)
            print(chunks[0])
            print("-" * 60)

    print("\n")
    print("=" * 60)
    print("PROCESSING COMPLETE")
    print("Total chunks:", total_chunks)
    print("=" * 60)


if __name__ == "__main__":
    main()