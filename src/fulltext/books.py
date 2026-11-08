"""Shared helpers for the book full-text pipeline.

File discovery, filename parsing, hashing, reading EPUB/MOBI/AZW3/PDF content,
and language detection. Used by inventory, prescreen_books and extract_ebook.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import unicodedata
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from src.common.paths import DATA_INTERIM, DATA_RAW

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

SOURCE_DIR = DATA_INTERIM / "creative_writing"
RAW_BOOKS = DATA_RAW / "books"
FULLTEXT_DIR = DATA_INTERIM / "fulltext"
MANIFEST_PATH = FULLTEXT_DIR / "manifest.csv"
JSON_DIR = FULLTEXT_DIR / "json"
EXCERPTS_DIR = FULLTEXT_DIR / "excerpts"
SAMPLES_DIR = EXCERPTS_DIR / "samples"

BOOK_EXTS = {"epub", "pdf", "mobi", "azw3", "docx", "txtzip", "rtf"}
# Round 1 formats: everything whose text is available without layout analysis or OCR
EBOOK_EXTS = {"epub", "mobi", "azw3", "docx", "txtzip"}
FOLDER_LANG = {"english": "en", "german": "de"}
CORPUS_LANGS = {"en", "de"}

FILENAME_RE = re.compile(r"^(?P<stem>.*?)\s*\[(?P<md5>[0-9a-f]{32})\]\.(?P<ext>[A-Za-z0-9]+)$")
WORD_RE = re.compile(r"\w+", re.UNICODE)


def parse_filename(name: str) -> dict:
    """Split 'Title - Author [md5].ext' into parts. Title may itself contain ' - '."""
    match = FILENAME_RE.match(name)
    if not match:
        return {"file_title": "", "file_author": "", "md5": "", "file_ext": Path(name).suffix.lstrip(".").lower()}
    stem = match.group("stem")
    title, sep, author = stem.rpartition(" - ")
    if not sep:
        title, author = stem, ""
    return {
        "file_title": title.strip(),
        "file_author": author.replace(" _ ", "; ").strip(),
        "md5": match.group("md5"),
        "file_ext": match.group("ext").lower(),
    }


def detect_format(path: Path) -> str:
    """Real format from magic bytes; many '.epub' downloads are actually PDF or MOBI."""
    ext = path.suffix.lstrip(".").lower()
    with open(path, "rb") as fh:
        head = fh.read(68)
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK"):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        if "word/document.xml" in names:
            return "docx"
        if names and all(n.endswith((".txt", "/")) for n in names):
            return "txtzip"  # Internet Archive page-text bundle
        return "epub"
    if head[60:68] == b"BOOKMOBI":
        return "azw3" if ext == "azw3" else "mobi"
    if head.startswith(b"{\\rtf"):
        return "rtf"
    return ext


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def discover_books() -> list[tuple[Path, str]]:
    """Return (path, folder_lang) for every book under the source and raw folders."""
    found = []
    for path in sorted(SOURCE_DIR.glob("*/*/*")):
        if path.is_file() and path.suffix.lstrip(".").lower() in BOOK_EXTS:
            found.append((path, FOLDER_LANG.get(path.parts[-3], "")))
    for path in sorted(RAW_BOOKS.glob("*/*")):
        if path.is_file() and path.suffix.lstrip(".").lower() in BOOK_EXTS:
            found.append((path, ""))
    return found


def normalize_text(text: str) -> str:
    """NFC, soft hyphens and zero-width characters removed, whitespace collapsed."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


# ---------------------------------------------------------------------------
# EPUB
# ---------------------------------------------------------------------------

def _opf_path(zf: zipfile.ZipFile) -> str | None:
    try:
        container = zf.read("META-INF/container.xml")
    except KeyError:
        return None
    soup = BeautifulSoup(container, "xml")
    rootfile = soup.find("rootfile")
    return rootfile.get("full-path") if rootfile else None


def _resolve(base: str, href: str) -> str:
    href = href.split("#", 1)[0]
    parts = []
    for part in (PurePosixPath(base).parent / href).parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


class EncryptedBookError(Exception):
    """Content documents are DRM-encrypted (font obfuscation alone does not count)."""


FONT_OBFUSCATION = ("http://www.idpf.org/2008/embedding", "http://ns.adobe.com/pdf/enc#RC")


def check_epub_encryption(zf: zipfile.ZipFile) -> None:
    try:
        xml = zf.read("META-INF/encryption.xml")
    except KeyError:
        return
    soup = BeautifulSoup(xml, "xml")
    for data in soup.find_all("EncryptedData"):
        method = data.find("EncryptionMethod")
        ref = data.find("CipherReference")
        uri = (ref.get("URI") if ref else "") or ""
        if method and method.get("Algorithm") in FONT_OBFUSCATION:
            continue
        if uri.lower().endswith((".html", ".xhtml", ".htm", ".xml")):
            raise EncryptedBookError(f"DRM-encrypted content ({uri})")


def epub_spine(path: Path) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    """Return spine documents in reading order and a {doc_name: toc_title} map.

    Reads the OPF directly from the zip, which tolerates more malformed EPUBs
    than ebooklib; falls back to all (x)html files in archive order.
    """
    from urllib.parse import unquote

    with zipfile.ZipFile(path) as zf:
        check_epub_encryption(zf)
        names = set(zf.namelist())
        opf = _opf_path(zf)
        docs: list[tuple[str, bytes]] = []
        toc: dict[str, str] = {}
        if opf and opf in names:
            soup = BeautifulSoup(zf.read(opf), "xml")
            manifest = {}
            for item in soup.find_all("item"):
                manifest[item.get("id")] = (
                    _resolve(opf, unquote(item.get("href", ""))),
                    item.get("media-type", ""),
                    item.get("properties", "") or "",
                )
            for ref in soup.find_all("itemref"):
                entry = manifest.get(ref.get("idref"))
                if entry and entry[0] in names and "html" in entry[1]:
                    docs.append((entry[0], zf.read(entry[0])))
            toc = _epub_toc(zf, opf, soup, manifest, names)
        if not docs:
            docs = [(n, zf.read(n)) for n in sorted(names) if n.lower().endswith((".html", ".xhtml", ".htm"))]
    return docs, toc


def _epub_toc(zf, opf, soup, manifest, names) -> dict[str, str]:
    """First TOC label per document, from NCX (EPUB2) or nav (EPUB3)."""
    from urllib.parse import unquote

    toc: dict[str, str] = {}
    spine = soup.find("spine")
    ncx_id = spine.get("toc") if spine else None
    ncx = manifest.get(ncx_id, (None,))[0] if ncx_id else None
    if not ncx:
        ncx = next((v[0] for v in manifest.values() if v[1] == "application/x-dtbncx+xml"), None)
    if ncx and ncx in names:
        nsoup = BeautifulSoup(zf.read(ncx), "xml")
        for point in nsoup.find_all("navPoint"):
            label, content = point.find("text"), point.find("content")
            if label and content and content.get("src"):
                doc = _resolve(ncx, unquote(content["src"]))
                toc.setdefault(doc, normalize_text(label.get_text()))
    if toc:
        return toc
    nav = next((v[0] for v in manifest.values() if "nav" in v[2].split()), None)
    if nav and nav in names:
        nsoup = BeautifulSoup(zf.read(nav), "lxml")
        for link in nsoup.find_all("a", href=True):
            doc = _resolve(nav, unquote(link["href"]))
            toc.setdefault(doc, normalize_text(link.get_text()))
    return toc


def html_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    return normalize_text(soup.get_text(" "))


# ---------------------------------------------------------------------------
# MOBI / AZW3
# ---------------------------------------------------------------------------

@contextmanager
def unpacked_kindle(path: Path, fmt: str):
    """Unpack MOBI/AZW3 with the `mobi` package; yields the EPUB or HTML path."""
    import mobi

    staged = None
    if path.suffix.lower() != f".{fmt}":
        # mobi.extract dispatches on the file extension
        staged = Path(tempfile.mkdtemp(prefix="cw_kindle_"))
        target = staged / f"book.{fmt}"
        shutil.copyfile(path, target)
        path = target
    tempdir, out = mobi.extract(str(path))
    try:
        yield Path(out)
    finally:
        shutil.rmtree(tempdir, ignore_errors=True)
        if staged:
            shutil.rmtree(staged, ignore_errors=True)


def kindle_documents(path: Path, fmt: str) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    with unpacked_kindle(path, fmt) as out:
        if out.suffix.lower() == ".epub":
            return epub_spine(out)
        html = out.read_bytes()
    # Single-HTML output (old MOBI): split on the MOBI page-break markers.
    chunks = re.split(rb"<mbp:pagebreak\s*/?>", html, flags=re.IGNORECASE)
    return [(f"part{i:04d}.html", c) for i, c in enumerate(chunks)], {}


def _html_doc(blocks: list[tuple[str, str]]) -> bytes:
    """Wrap (tag, text) blocks as minimal HTML so every format shares one parser."""
    from html import escape

    body = "".join(f"<{tag}>{escape(text)}</{tag}>" for tag, text in blocks)
    return f"<html><body>{body}</body></html>".encode("utf-8")


def docx_documents(path: Path) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        soup = BeautifulSoup(zf.read("word/document.xml"), "xml")
    blocks = []
    for para in soup.find_all("p"):
        text = "".join(t.get_text() for t in para.find_all("t"))
        style = para.find("pStyle")
        heading = style is not None and "heading" in (style.get("w:val") or style.get("val") or "").lower()
        if text.strip():
            blocks.append(("h2" if heading else "p", text))
    return [("document.xml", _html_doc(blocks))], {}


def txtzip_documents(path: Path) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    """One document per OCR page; paragraphs are blank-line separated blocks."""
    docs = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".txt")):
            raw = zf.read(name).decode("utf-8", errors="replace")
            raw = re.sub(r"(\w)-\n\s*(?=[a-zäöüß])", r"\1", raw)  # OCR line-break hyphenation
            blocks = [("p", b) for b in re.split(r"\n\s*\n", raw) if b.strip()]
            docs.append((name, _html_doc(blocks)))
    return docs, {}


def ebook_documents(path: Path, fmt: str | None = None) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    fmt = fmt or detect_format(path)
    if fmt == "epub":
        return epub_spine(path)
    if fmt in {"mobi", "azw3"}:
        return kindle_documents(path, fmt)
    if fmt == "docx":
        return docx_documents(path)
    if fmt == "txtzip":
        return txtzip_documents(path)
    raise ValueError(f"not an ebook ({fmt}): {path}")


# ---------------------------------------------------------------------------
# Text samples and language detection
# ---------------------------------------------------------------------------

def ebook_sample(path: Path, fmt: str, max_chars: int = 40000) -> str:
    """Body text sample: skip the first 15% of documents (cover, front matter)."""
    docs, _ = ebook_documents(path, fmt)
    start = int(len(docs) * 0.15)
    parts, total = [], 0
    for _, html in docs[start:] + docs[:start]:
        text = html_text(html)
        if not text:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def front_text(path: Path, fmt: str, max_chars: int = 15000) -> str:
    """Title pages / imprint: first documents of an ebook or first 6 PDF pages."""
    if fmt == "pdf":
        import pymupdf

        with pymupdf.open(path, filetype="pdf") as doc:
            return normalize_text(" ".join(doc[i].get_text() for i in range(min(6, doc.page_count))))[:max_chars]
    if fmt not in EBOOK_EXTS:
        return ""
    docs, _ = ebook_documents(path, fmt)
    parts, total = [], 0
    for _, html in docs[: max(int(len(docs) * 0.15), 3)]:
        text = html_text(html)
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars]


def pdf_page_texts(path: Path, max_pages: int = 25) -> tuple[int, list[str]]:
    """Return (page_count, texts of up to max_pages evenly spread body pages)."""
    import pymupdf

    with pymupdf.open(path, filetype="pdf") as doc:
        n = doc.page_count
        if n == 0:
            return 0, []
        lo, hi = int(n * 0.05), max(int(n * 0.95), 1)
        span = list(range(lo, hi)) or list(range(n))
        step = max(len(span) // max_pages, 1)
        picks = span[::step][:max_pages]
        return n, [doc[i].get_text() for i in picks]


_DETECTOR = None


def detector():
    global _DETECTOR
    if _DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder

        langs = [Language.ENGLISH, Language.GERMAN, Language.FRENCH, Language.SPANISH,
                 Language.ITALIAN, Language.DUTCH, Language.PORTUGUESE, Language.POLISH,
                 Language.SWEDISH, Language.DANISH, Language.LATIN]
        _DETECTOR = LanguageDetectorBuilder.from_languages(*langs).build()
    return _DETECTOR


def detect_language(text: str, chunk: int = 3000, n_chunks: int = 5) -> tuple[str, float]:
    """Majority vote over a few chunks; returns (iso639-1 code, share of chunks agreeing)."""
    text = text.strip()
    if word_count(text) < 50:
        return "", 0.0
    step = max(len(text) // n_chunks, chunk)
    votes: dict[str, int] = {}
    for start in range(0, len(text), step):
        piece = text[start:start + chunk]
        if word_count(piece) < 30:
            continue
        lang = detector().detect_language_of(piece)
        if lang is not None:
            code = lang.iso_code_639_1.name.lower()
            votes[code] = votes.get(code, 0) + 1
    if not votes:
        return "", 0.0
    best = max(votes, key=votes.get)
    return best, round(votes[best] / sum(votes.values()), 2)


def write_sample(book_id: str, text: str) -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLES_DIR / f"{book_id}.txt").write_text(text, encoding="utf-8")


def read_sample(book_id: str) -> str:
    path = SAMPLES_DIR / f"{book_id}.txt"
    return path.read_text(encoding="utf-8") if path.is_file() else ""
