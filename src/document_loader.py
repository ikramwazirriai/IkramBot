from pathlib import Path
from pypdf import PdfReader


DATA_DIR = Path("data")


def load_pdf(file_path):
    """Extract text from a single PDF file."""

    reader = PdfReader(file_path)

    text = ""

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()

        if page_text:
            text += f"\n--- Page {page_number} ---\n"
            text += page_text

    return text


def load_all_pdfs():
    """Load all PDF files from the data directory."""

    documents = []

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    for pdf_file in pdf_files:

        print(f"Loading: {pdf_file.name}")

        text = load_pdf(pdf_file)

        documents.append(
            {
                "source": pdf_file.name,
                "text": text,
            }
        )

    return documents


if __name__ == "__main__":

    documents = load_all_pdfs()

    print("\n==============================")
    print("Ikrambot Knowledge Base")
    print("==============================")

    print(f"Documents loaded: {len(documents)}")

    for document in documents:

        print("\nSource:", document["source"])
        print("Characters:", len(document["text"]))

        preview = document["text"][:300]

        print("Preview:")
        print(preview)