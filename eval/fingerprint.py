"""Deterministic corpus fingerprint so before/after runs compare the same DB."""

import hashlib


def corpus_fingerprint(source_ids) -> str:
    uniq = sorted({str(s) for s in source_ids})
    digest = hashlib.sha256(",".join(uniq).encode()).hexdigest()[:12]
    return f"{len(uniq)}:{digest}"
