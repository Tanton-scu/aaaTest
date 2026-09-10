"""基于 FlagEmbedding 的 BGE embedding / reranker 适配器。

核心 RAG 管线负责 chunk、粗召回、RRF 融合和 provenance；本文件只负责把
FlagEmbedding 模型包装成项目内部端口，避免把第三方 SDK 细节散落到业务流程里。
"""

from __future__ import annotations

import os
from typing import Sequence


DEFAULT_BGE_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BGE_QUERY_INSTRUCTION = "为这个查询生成表示以用于检索相关文献："


class FlagEmbeddingUnavailable(RuntimeError):
    """FlagEmbedding 未安装或无法导入。"""


class BGEEmbeddingVectorizer:
    """BGE-M3 embedding adapter。

    BGE 检索约定：query embedding 加检索指令前缀，document chunk 作为普通语料编码。
    """

    production_semantic_embedding = True

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_EMBEDDING_MODEL,
        *,
        query_instruction: str = DEFAULT_BGE_QUERY_INSTRUCTION,
        use_fp16: bool = True,
        device: str | None = None,
    ) -> None:
        try:
            from FlagEmbedding import FlagModel
        except Exception as exc:  # pragma: no cover - CI 不强制下载大模型依赖
            raise FlagEmbeddingUnavailable(
                "FlagEmbedding is not installed. Run: pip install FlagEmbedding"
            ) from exc

        self.name = model_name
        self.query_instruction = query_instruction
        kwargs = {"use_fp16": bool(use_fp16)}
        if device:
            kwargs["device"] = device
        self.model = FlagModel(model_name, **kwargs)

    def encode(self, text: str) -> Sequence[float]:
        return self.encode_document(text)

    def encode_query(self, text: str) -> Sequence[float]:
        query_text = f"{self.query_instruction}{text}" if self.query_instruction else text
        if hasattr(self.model, "encode_queries"):
            return _first_vector(self.model.encode_queries([query_text]))
        return _first_vector(self.model.encode([query_text]))

    def encode_document(self, text: str) -> Sequence[float]:
        if hasattr(self.model, "encode_corpus"):
            return _first_vector(self.model.encode_corpus([text]))
        return _first_vector(self.model.encode([text]))


class BGEReranker:
    """BGE cross-encoder reranker。

    它只接收 RAG 管线粗召回后的 Top-N chunks，不对全库逐条打分。
    """

    production_cross_encoder = True

    def __init__(
        self,
        model_name: str = DEFAULT_BGE_RERANKER_MODEL,
        *,
        use_fp16: bool = True,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except Exception as exc:  # pragma: no cover - CI 不强制下载大模型依赖
            raise FlagEmbeddingUnavailable(
                "FlagEmbedding is not installed. Run: pip install FlagEmbedding"
            ) from exc

        self.name = model_name
        self.normalize = bool(normalize)
        kwargs = {"use_fp16": bool(use_fp16)}
        if device:
            kwargs["device"] = device
        self.model = FlagReranker(model_name, **kwargs)

    def score(self, query_text: str, documents: Sequence[str]) -> tuple[float, ...]:
        if not documents:
            return ()
        pairs = [[query_text, document] for document in documents]
        try:
            raw = self.model.compute_score(pairs, normalize=self.normalize)
        except TypeError:
            raw = self.model.compute_score(pairs)
        if isinstance(raw, (float, int)):
            return (float(raw),)
        return tuple(float(item) for item in list(raw))


def build_bge_vectorizer_from_env():
    backend = os.getenv("RAG_EMBEDDING_BACKEND", "hash").strip().lower()
    if backend not in {"bge", "bge-m3", "flagembedding"}:
        return None
    return BGEEmbeddingVectorizer(
        os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_BGE_EMBEDDING_MODEL).strip()
        or DEFAULT_BGE_EMBEDDING_MODEL,
        query_instruction=os.getenv(
            "RAG_QUERY_INSTRUCTION", DEFAULT_BGE_QUERY_INSTRUCTION
        ),
        use_fp16=_env_bool("RAG_BGE_USE_FP16", True),
        device=os.getenv("RAG_BGE_DEVICE", "").strip() or None,
    )


def build_bge_reranker_from_env():
    backend = os.getenv("RAG_RERANKER_BACKEND", "none").strip().lower()
    if backend not in {"bge", "bge-reranker", "flagembedding"}:
        return None
    return BGEReranker(
        os.getenv("RAG_RERANKER_MODEL", DEFAULT_BGE_RERANKER_MODEL).strip()
        or DEFAULT_BGE_RERANKER_MODEL,
        use_fp16=_env_bool("RAG_BGE_USE_FP16", True),
        device=os.getenv("RAG_BGE_DEVICE", "").strip() or None,
        normalize=_env_bool("RAG_RERANKER_NORMALIZE", True),
    )


def _first_vector(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not value:
        return ()
    first = value[0]
    if hasattr(first, "tolist"):
        first = first.tolist()
    return tuple(float(item) for item in first)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false")


__all__ = [
    "BGEEmbeddingVectorizer",
    "BGEReranker",
    "DEFAULT_BGE_EMBEDDING_MODEL",
    "DEFAULT_BGE_QUERY_INSTRUCTION",
    "DEFAULT_BGE_RERANKER_MODEL",
    "FlagEmbeddingUnavailable",
    "build_bge_reranker_from_env",
    "build_bge_vectorizer_from_env",
]
