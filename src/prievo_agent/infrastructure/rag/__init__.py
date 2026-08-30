"""Literature RAG adapter：BM25、embedding、RRF 与可选 cross-encoder rerank。"""

from .literature_hybrid import (
    DeterministicHashingVectorizer,
    LiteratureRetrievalResult,
    LocalHybridLiteratureRAG,
    RerankerPort,
    VectorPort,
    build_literature_rag_from_env,
)

__all__ = [
    "DeterministicHashingVectorizer",
    "LiteratureRetrievalResult",
    "LocalHybridLiteratureRAG",
    "RerankerPort",
    "VectorPort",
    "build_literature_rag_from_env",
]
