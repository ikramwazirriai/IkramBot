import re


def clean_text(text):
    """
    Clean extracted PDF text while preserving
    meaningful information.
    """

    # Replace multiple spaces/tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove spaces at the beginning/end of lines
    lines = [line.strip() for line in text.splitlines()]

    # Remove completely empty lines
    lines = [line for line in lines if line]

    # Rebuild text
    text = "\n".join(lines)

    return text.strip()