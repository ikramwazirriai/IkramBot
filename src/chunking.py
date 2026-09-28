def split_text(text, chunk_size=800, chunk_overlap=150):
    """
    Split text into overlapping chunks.

    chunk_size:
        Maximum approximate number of characters per chunk.

    chunk_overlap:
        Number of characters shared between neighboring chunks.
    """

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping overlap
        start = end - chunk_overlap

    return chunks