"""
Grounded prompt templates for Gemini.

Constructs prompts that explicitly instruct the LLM to:
- Answer ONLY from supplied document context
- Never use outside knowledge or fabricate information
- Cite sources with file name and page number
- Return 'insufficient information' when context doesn't support an answer
- Use conversation history only for understanding intent, not as evidence
"""


SYSTEM_INSTRUCTION = """You are a document question-answering assistant. You MUST follow these rules strictly:

1. Answer ONLY based on the DOCUMENT CONTEXT provided below.
2. Do NOT use any outside knowledge, training data, or general knowledge.
3. Do NOT fabricate, infer, or guess information not present in the document context.
4. If the answer is not supported by the provided document context, respond with EXACTLY: insufficient information
5. When answering, cite your sources using the format: [filename, page X]
6. CONVERSATION HISTORY is provided only to help you understand the user's intent (e.g., what "it" or "that" refers to). Do NOT treat conversation history as factual evidence.
7. If a follow-up question refers to something from conversation history, you must still find supporting evidence in the DOCUMENT CONTEXT to answer it.
8. Be concise and accurate. Quote or paraphrase the document context directly."""


def build_grounded_prompt(
    question: str,
    context_chunks: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """Build a grounded prompt for Gemini.

    Args:
        question: The user's current question.
        context_chunks: List of dicts with keys: text, file_name, page_number.
        conversation_history: Optional list of previous messages with keys:
            role ('user' or 'assistant'), content.

    Returns:
        Formatted prompt string.
    """
    parts = []

    # --- Conversation History ---
    if conversation_history:
        parts.append("=== CONVERSATION HISTORY ===")
        parts.append(
            "(Use this ONLY to understand the user's intent. "
            "Do NOT treat this as factual evidence.)\n"
        )
        for msg in conversation_history:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("")

    # --- Document Context ---
    parts.append("=== DOCUMENT CONTEXT ===")
    if context_chunks:
        for i, chunk in enumerate(context_chunks, 1):
            source = f"[{chunk['file_name']}, page {chunk['page_number']}]"
            parts.append(f"\n--- Chunk {i} {source} ---")
            parts.append(chunk["text"])
    else:
        parts.append("No relevant document context was found.")
    parts.append("")

    # --- Current Question ---
    parts.append("=== CURRENT QUESTION ===")
    parts.append(question)
    parts.append("")

    # --- Instructions ---
    parts.append("=== INSTRUCTIONS ===")
    parts.append(
        "Answer the question using ONLY the document context above. "
        "Cite sources as [filename, page X]. "
        "If the answer is not supported by the document context, "
        "respond with exactly: insufficient information"
    )

    return "\n".join(parts)
