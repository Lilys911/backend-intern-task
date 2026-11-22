def chunk_text(text: str, size: int = 50, overlap: int = 10) -> list[str]:
    """Split `text` into chunks of words.

    Rules (see README for examples):
    - Words are separated by whitespace (use str.split()).
    - Each chunk has at most `size` words, joined with a single space.
    - Consecutive chunks share exactly `overlap` words.
    - Stop when a chunk reaches the last word. No extra trailing chunk.
    - Empty or whitespace-only text returns [].
    - Raise ValueError if size <= 0, overlap < 0, or overlap >= size.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= size:
        raise ValueError("overlap cannot be more or equal than size")
    word = text.split()
    step = size-overlap
    start = 0
    chunks = []
    if not word:
        return []
    while True:
        chunk = word[start:start+size]
        chunks.append(" ".join(chunk))
        if start +size-1 >= len(word) - 1:
            break

        start += step
    return chunks

