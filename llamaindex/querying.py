import re


def is_garbage_line(line: str) -> bool:
    if not line or len(line) < 3:
        return False

    # High ratio of non-ASCII / non-printable characters
    bad = sum(1 for c in line if ord(c) > 127 or (ord(c) < 32 and c not in '\t'))
    if bad / len(line) > 0.10:
        return True

    # Repeating substring pattern e.g. "sFxAbsFxAbsFxAbs"
    for size in range(3, 9):
        if len(line) >= size * 4 and line[:size] * 4 in line:
            return True

    # File paths, metadata, binary refs
    if re.search(r'[A-Za-z]:\\[A-Za-z]', line):           # Windows path
        return True
    if re.search(r'_params\.|hyperlink|\.pdf\b', line):    # PDF metadata
        return True
    if re.search(r'\.(h|so|dll|exe|obj|pyc)\b', line):     # Binary files
        return True
    if re.search(r'(@[A-Za-z0-9_]{4,}|%[0-9A-Fa-f]{2}){2,}', line):
        return True

    return False


def clean_context(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line and not is_garbage_line(line):
            lines.append(line)
    return "\n".join(lines)


def query_index(index, query, llm):
    """
    Retrieve context from Pinecone, clean it BEFORE sending to LLM,
    then call LLM directly with the clean context.
    """
    # Step 1: retrieve raw chunks from Pinecone
    try:
        retriever = index.as_retriever(similarity_top_k=4)
        nodes = retriever.retrieve(query)
        raw_chunks = [node.get_content() for node in nodes]
    except Exception:
        raw_chunks = []

    # Step 2: clean every chunk before LLM sees it
    clean_chunks = []
    for chunk in raw_chunks:
        cleaned = clean_context(chunk)
        if len(cleaned.strip()) > 50:          # skip near-empty cleaned chunks
            clean_chunks.append(cleaned)

    context = "\n\n".join(clean_chunks) if clean_chunks else ""

    # Step 3: build prompt and call LLM directly
    if context:
        prompt = (
            "You are an expert mechanical engineering assistant.\n"
            "Use the context below to answer the question. "
            "If context is not relevant, use your own knowledge.\n"
            "Write all math in LaTeX using $$ delimiters. "
            "Give only the direct answer — no preamble, no commentary.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )
    else:
        prompt = (
            "You are an expert mechanical engineering assistant. "
            "Answer the following question directly using your own knowledge. "
            "Write all math in LaTeX using $$ delimiters. "
            "Give only the direct answer — no preamble.\n\n"
            f"Question: {query}\nAnswer:"
        )

    result = llm.complete(prompt)

    # Wrap in a simple response object so home.py stays unchanged
    class Response:
        def __init__(self, text):
            self.response = text

    return Response(result.text)
