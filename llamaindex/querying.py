import re
from llama_index.core import PromptTemplate


def is_garbage_line(line: str) -> bool:
    """Detect if a single line is corrupted/binary garbage."""
    if not line or len(line) < 3:
        return False

    # High ratio of non-ASCII or non-printable characters
    bad = sum(1 for c in line if ord(c) > 127 or (ord(c) < 32 and c not in '\t'))
    if len(line) > 0 and bad / len(line) > 0.10:
        return True

    # Repeating substring pattern — e.g. "sFxAbsFxAbsFxAbs"
    for size in range(3, 9):
        if len(line) >= size * 4 and line[:size] * 4 in line:
            return True

    # File path / binary reference patterns
    if re.search(r'\.(h|so|dll|exe|obj|pyc)\b', line):
        return True
    if re.search(r'(@[A-Za-z0-9_]{4,}|%[0-9A-Fa-f]{2}){2,}', line):
        return True

    return False


def is_garbage_response(text: str) -> bool:
    """Detect if the full LLM response is garbage or corrupted."""
    if not text:
        return False

    # Repeating pattern across full response
    for size in range(3, 9):
        chunk = text[:size]
        if len(chunk) == size and chunk * 5 in text:
            return True

    # Non-ASCII garbage chars
    if re.search(r'[ɑɪ΢νωαβγδεζηθλμξπρστφχψ]', text):
        return True

    # File references in response
    if re.search(r'\w+\.(h|so|dll|exe|obj)\b', text):
        return True

    return False


def clean_context(raw: str) -> str:
    """Strip garbage lines from PDF-extracted context."""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line and not is_garbage_line(line):
            lines.append(line)
    return "\n".join(lines)


def query_index(index, query, llm):
    """Queries the index with the given query, with automatic garbage recovery."""

    qa_prompt_tmpl_str = (
        "You are an expert mechanical engineering assistant.\n"
        "Relevant context from the knowledge base is provided below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Instructions:\n"
        "- If the context above contains relevant information, use it to answer.\n"
        "- If the context is empty or not relevant, answer from your own mechanical engineering knowledge.\n"
        "- Write all mathematical expressions in LaTeX using $$ delimiters.\n"
        "- Give only the direct answer. No preamble, no commentary.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)
    query_engine = index.as_query_engine(text_qa_template=qa_prompt_tmpl, llm=llm)
    response = query_engine.query(query)

    # If response is garbage, bypass Pinecone entirely and ask LLM directly
    if is_garbage_response(response.response):
        direct_prompt = (
            "You are an expert mechanical engineering assistant. "
            "Answer the following question directly and concisely using your own knowledge. "
            "Use LaTeX ($$...$$) for any mathematical expressions. "
            "Give only the answer, no preamble.\n\n"
            f"Question: {query}\nAnswer:"
        )
        clean = llm.complete(direct_prompt)
        response.response = clean.text

    return response
