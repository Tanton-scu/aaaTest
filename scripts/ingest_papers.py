"""把本地 PDF 转成产品 Literature retriever 可直接读取的统一 paper schema。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_ROOT = PROJECT_ROOT / "data" / "papers"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "literature" / "pdf_corpus.json"
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
NUMBERED_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*[.)]?\s+|(?:abstract|introduction|background|method(?:s|ology)?|"
    r"experiment(?:s|al results)?|results?|discussion|conclusion(?:s)?|references)\b)",
    re.IGNORECASE,
)
UNIT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+/_-]*|[\u4e00-\u9fff]|[^\s]")


@dataclass(frozen=True)
class PDFIngestResult:
    papers: tuple[dict[str, Any], ...]
    skipped: tuple[dict[str, str], ...]
    output_path: str


def ingest_pdfs(
    paper_root: Path = DEFAULT_PAPER_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    reader_factory: Callable[[str], Any] | None = None,
    chunk_words: int = 180,
    overlap_words: int = 30,
) -> PDFIngestResult:
    """逐 PDF 导入；单个坏文件会进入 skipped，不破坏其他论文和已有输出。"""

    if chunk_words <= 0 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("chunk_words/overlap_words 参数无效")
    root = Path(paper_root)
    output = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.glob("*.pdf"), key=lambda item: item.name.casefold())
    if not paths:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text("[]\n", encoding="utf-8")
        temporary.replace(output)
        return PDFIngestResult((), (), str(output))
    if reader_factory is None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("请先安装 PDF 可选依赖：pip install -e .[papers]") from exc
        reader_factory = PdfReader

    papers = []
    skipped = []
    used_ids = set()
    for path in paths:
        try:
            paper = _ingest_one(
                path,
                reader_factory,
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
            paper_id = paper["paper_id"]
            if paper_id in used_ids:
                paper_id = "{}-{}".format(paper_id, paper["source_digest"][:8])
                paper["paper_id"] = paper_id
                _rewrite_chunk_ids(paper)
            used_ids.add(paper_id)
            papers.append(paper)
        except Exception as exc:
            skipped.append(
                {"source_path": str(path), "error": "{}: {}".format(type(exc).__name__, exc)}
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    return PDFIngestResult(tuple(papers), tuple(skipped), str(output))


def _ingest_one(path, reader_factory, chunk_words, overlap_words):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = reader_factory(str(path))
    metadata = _metadata_dict(getattr(reader, "metadata", None))
    pages = []
    for number, page in enumerate(getattr(reader, "pages", ()), 1):
        text = str(page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    if not pages:
        raise ValueError("PDF 没有可提取文本")

    title = _metadata_value(metadata, "title") or path.stem.replace("_", " ").strip()
    author_text = _metadata_value(metadata, "author")
    authors = _authors(author_text)
    metadata_text = " ".join(str(value) for value in metadata.values())
    doi = DOI_PATTERN.search(metadata_text + " " + pages[0][1][:3000])
    identifier = "DOI:{}".format(doi.group(0).rstrip(".,;")) if doi else "sha256:{}".format(digest)
    year_match = YEAR_PATTERN.search(
        " ".join(
            [
                _metadata_value(metadata, "creationdate"),
                _metadata_value(metadata, "moddate"),
                _metadata_value(metadata, "creation_date"),
                _metadata_value(metadata, "modification_date"),
                path.stem,
            ]
        )
    )
    year = int(year_match.group(0)) if year_match else 0
    paper_id = _paper_id(title, year, digest)
    sections = _extract_sections(
        paper_id, pages, chunk_words=chunk_words, overlap_words=overlap_words
    )
    try:
        source_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_path = path.resolve().as_posix()
    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "identifier": identifier,
        "source_path": source_path,
        "primary": True,
        "sections": sections,
        "source_digest": digest,
        "ingest": {
            "format": "pdf",
            "page_count": len(getattr(reader, "pages", ())),
            "section_aware": True,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
        },
    }


def _extract_sections(paper_id, pages, chunk_words, overlap_words):
    grouped = OrderedDict()
    current = "document"
    for page_number, text in pages:
        buffer = []

        def flush():
            if buffer:
                grouped.setdefault(current, []).append(
                    {"page": page_number, "text": " ".join(buffer).strip()}
                )
                buffer.clear()

        for raw_line in text.splitlines():
            line = " ".join(raw_line.split()).strip()
            if not line:
                continue
            if _is_heading(line):
                flush()
                current = _section_name(line)
                grouped.setdefault(current, [])
            else:
                buffer.append(line)
        flush()

    sections = []
    for section_name, segments in grouped.items():
        if not segments:
            continue
        units = []
        for segment in segments:
            units.extend(
                (token, int(segment["page"]))
                for token in UNIT_PATTERN.findall(segment["text"])
            )
        chunks = []
        start = 0
        while start < len(units):
            end = min(len(units), start + chunk_words)
            selected = units[start:end]
            chunk_index = len(chunks)
            section_slug = _slug(section_name) or "section"
            chunks.append(
                {
                    "chunk_id": "{}:{}:{}".format(paper_id, section_slug, chunk_index),
                    "chunk_index": chunk_index,
                    "section": section_name,
                    "page_start": min(page for _, page in selected),
                    "page_end": max(page for _, page in selected),
                    "content": " ".join(token for token, _ in selected),
                }
            )
            if end == len(units):
                break
            start = max(start + 1, end - overlap_words)
        for index, chunk in enumerate(chunks):
            previous = chunks[index - 1]["chunk_id"] if index > 0 else ""
            following = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else ""
            chunk["previous_chunk_id"] = previous
            chunk["next_chunk_id"] = following
            chunk["adjacent_chunk_ids"] = [
                item for item in (previous, following) if item
            ]
        sections.append(
            {
                "section": section_name,
                "page_start": min(item["page"] for item in segments),
                "page_end": max(item["page"] for item in segments),
                "content": "\n".join(item["text"] for item in segments),
                "chunks": chunks,
            }
        )
    if not sections:
        raise ValueError("PDF 未形成有效 section/chunk")
    return sections


def _is_heading(line):
    if len(line) > 100 or len(line.split()) > 14:
        return False
    if NUMBERED_HEADING.match(line):
        return True
    letters = [item for item in line if item.isalpha()]
    if letters and all(not item.islower() for item in letters):
        return True
    return line.istitle() and not line.endswith((".", "?", "!", ":", ";"))


def _section_name(line):
    value = re.sub(r"^\d+(?:\.\d+)*[.)]?\s*", "", line).strip()
    return value or "document"


def _metadata_dict(metadata):
    if metadata is None:
        return {}
    try:
        return {str(key).lstrip("/").lower(): value for key, value in dict(metadata).items()}
    except (TypeError, ValueError):
        return {}


def _metadata_value(metadata, key):
    return str(metadata.get(key, "") or "").strip()


def _authors(value):
    if not value:
        return ["Unknown"]
    parts = re.split(r"\s*(?:;|\band\b|\|)\s*", value, flags=re.IGNORECASE)
    authors = [item.strip() for item in parts if item.strip()]
    return authors or ["Unknown"]


def _paper_id(title, year, digest):
    words = _slug(title).split("-")
    short_title = "-".join(words[:8]) or "paper"
    return "{}-{}-{}".format(year or "unknown", short_title, digest[:8])


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")


def _rewrite_chunk_ids(paper):
    all_chunks = [chunk for section in paper["sections"] for chunk in section["chunks"]]
    mapping = {}
    for chunk in all_chunks:
        old = chunk["chunk_id"]
        suffix = old.split(":", 1)[1] if ":" in old else old
        mapping[old] = "{}:{}".format(paper["paper_id"], suffix)
    for chunk in all_chunks:
        chunk["chunk_id"] = mapping[chunk["chunk_id"]]
        chunk["previous_chunk_id"] = mapping.get(chunk["previous_chunk_id"], "")
        chunk["next_chunk_id"] = mapping.get(chunk["next_chunk_id"], "")
        chunk["adjacent_chunk_ids"] = [
            mapping[item] for item in chunk["adjacent_chunk_ids"] if item in mapping
        ]


def main(argv=None):
    parser = argparse.ArgumentParser(description="导入本地 PDF Literature corpus")
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-words", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=30)
    args = parser.parse_args(argv)
    if not list(args.paper_root.glob("*.pdf")):
        print("{} 中没有 PDF；写入空的统一 corpus，不下载外部论文。".format(args.paper_root))
    try:
        result = ingest_pdfs(
            args.paper_root,
            args.output,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
    except RuntimeError as exc:
        print(str(exc))
        return 2
    except OSError as exc:
        print(
            "PDF ingest 无法写入 {}：{}。Linux bind mount 请确保当前容器用户"
            "对 data/literature 有写权限，或改在宿主机执行该脚本。".format(
                args.output, exc
            ),
            file=sys.stderr,
        )
        return 3
    print(
        "已导入 {} 篇 PDF，跳过 {} 篇：{}".format(
            len(result.papers), len(result.skipped), result.output_path
        )
    )
    for item in result.skipped:
        print("跳过 {source_path}：{error}".format(**item))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
