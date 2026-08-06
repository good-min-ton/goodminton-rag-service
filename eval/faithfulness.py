"""Disabled hook for E2+G2 (corrective-RAG + citation). Returns None (off)."""


async def evaluate_faithfulness(query: str, answer: str, contexts: list[str]):
    # Intentionally disabled in B1. E2+G2 will implement + enable.
    return None
