"""Suggest and apply corrected titles and authors for the English books.

`suggest` collects candidate (title, author) pairs from the download filename,
the candidate lists, metadata embedded in the file and Open Library (ISBN from
the copyright page, else a title/author search). Every candidate is checked
against the book's own first pages; the best verified pair is proposed with a
confidence level and written to data/interim/fulltext/title_author_review.csv.

`apply` copies accepted rows into config/book_metadata_overrides.csv, which
reorganize_books and inventory use for filenames.

  high        title and author verified on the title page, every changed field backed by 2+ sources
  medium      verified by one source, or title page unreadable but sources agree
  conflict    a change or disagreement that the title page does not confirm
  wrong_book  the file's text is a different book than its current name

Acceptance: `high` rows are applied unless `accept=no`; all others need `accept=yes`.

Usage:
  python -m src.fulltext.fix_titles suggest [--offline] [--limit N] [--book-id ID ...]
  python -m src.fulltext.fix_titles apply
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import ROOT
from src.fulltext.books import (
    FULLTEXT_DIR,
    MANIFEST_PATH,
    METADATA_OVERRIDES,
    display_title_author,
    embedded_metadata,
    front_text,
    load_metadata_overrides,
    normalize_text,
    parse_filename,
    raw_files_by_md5,
)

REVIEW_PATH = FULLTEXT_DIR / "title_author_review.csv"
CACHE_PATH = FULLTEXT_DIR / "lookup_cache.json"
OL = "https://openlibrary.org"
USER_AGENT = "creative-writing-corpus/0.1 (research corpus metadata check)"

# Wording preference when several sources give the same words
SOURCES = ("list", "openlibrary", "embedded", "filename")
CONFIDENCE_ORDER = {"wrong_book": 0, "conflict": 1, "medium": 2, "high": 3}
OVERRIDE_COLS = ["book_id", "title", "author", "source", "confidence", "identity_changed", "reviewer", "note"]

VERIFIED = 0.8        # share of title words found on the first pages
UNVERIFIED = 0.5      # below this the current title is treated as absent from the text
MIN_FRONT_WORDS = 30  # fewer words: title page unreadable (scan, image cover)

WORD_RE = re.compile(r"\w+", re.UNICODE)
JUNK_TITLE = re.compile(r"^(untitled|test|unknown|title|cover|none|null|document\d*|book\d*|[\W\d_]*)$", re.I)
JUNK_AUTHOR = re.compile(
    r"^(unknown|anonymous|anon|user|owner|admin|administrator|author|test|various|n/?a|none|"
    r"windows user|microsoft|adobe|calibre|[\W\d_]*)$",
    re.I,
)
PUBLISHERS = {
    "scribner", "penguin", "routledge", "macmillan", "palgrave", "wiley", "sage", "bloomsbury",
    "harpercollins", "random house", "vintage", "anchor", "pearson", "longman", "mcgraw-hill",
    "oxford university press", "cambridge university press", "writer's digest books", "heinemann",
    "scholastic", "norton", "w. w. norton", "houghton mifflin", "simon & schuster", "taylor & francis",
}
ROLE_WORDS = re.compile(
    r"\(?\b(author|authors|editor|editors|eds?|edited by|compiler|comp|illustrator|illustrated by|"
    r"translator|translated by|trans|foreword|introduction|narrator|contributor|writer|"
    r"edt|trl|ill|ctb|aut|cmp|professor|prof|dr)\b\.?\)?",
    re.I,
)
EDITION = re.compile(
    r"[,:;.]?\s*\b(\d+(st|nd|rd|th)|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|new|revised)"
    r"( anniversary| revised| and expanded| updated)? edition\b",
    re.I,
)
DATES = re.compile(r"\(?\b\d{3,4}\s*-\s*(\d{3,4})?\)?\.?|\(?\bb\.\s*\d{4}\)?")
NAME_PARTICLES = {"de", "van", "von", "der", "da", "di", "du", "le", "la", "mac", "st"}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "phd", "md"}
TRAILING_NOTE = re.compile(
    r"\s*[\(\[][^)\]]*\b(edition|kindle|e-?book|epub|pdf|series|guides|library|classics|vol|volume|book \d+)\b[^)\]]*[\)\]]\s*$",
    re.I,
)
ISBN_RE = re.compile(r"ISBN[\s:-]*(?:1[03])?[\s:-]*((?:97[89][\s-]?)?(?:\d[\s-]?){9}[\dXx])", re.I)


# ---------------------------------------------------------------------------
# Cleaning and comparison
# ---------------------------------------------------------------------------

def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text or "").lower().replace("&", " and ")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return WORD_RE.findall(text.replace("_", " "))


def title_key(title: str) -> tuple[str, ...]:
    return tuple(words(title))


def main_key(title: str) -> tuple[str, ...]:
    return title_key(re.split(r"\s*[:;]\s+|\s+[-–—]\s+|\.\s+", title or "", maxsplit=1)[0])


def same_book(a: str, b: str) -> bool:
    """Main titles share most words, or one main title starts the other."""
    ka, kb = main_key(a), main_key(b)
    if not ka or not kb:
        return False
    if ka[: len(kb)] == kb or kb[: len(ka)] == ka:
        return True
    sa, sb = set(ka), set(kb)
    return len(sa & sb) / len(sa | sb) >= 0.6


def clean_title(title: str, source: str) -> str:
    title = normalize_text(title or "")
    if source == "filename":
        title = re.sub(r"(^|\s)_(\S[^_]*?)_(?=\s|$)", r"\1'\2'", title)  # _quoted_
        title = re.sub(r"\s+_\s+", ": ", title)
        title = re.sub(r"_$", "?", title)
        title = re.sub(r"_\s+", ": ", title)
        title = re.sub(r"(?<=\w)_(?=\w)", "/", title)
        title = title.replace("_", " ")
    title = re.sub(r"^microsoft (word|powerpoint)\s*-\s*", "", title, flags=re.I)
    title = re.sub(r"\.(pdf|docx?|epub|mobi|indd|qxd|rtf)$", "", title, flags=re.I)
    title = re.split(r"\s+=", title, maxsplit=1)[0]          # MARC parallel title "= ..."
    title = EDITION.sub("", title)
    while TRAILING_NOTE.search(title):
        title = TRAILING_NOTE.sub("", title)
    if ":" in title:                                         # series note after a subtitle
        title = re.sub(r"\s*\([^()]*\)\s*$", "", title)
    title = re.sub(r"\s+/\s*$", "", title)                   # MARC "title /"
    title = re.sub(r"\s*;\s+", ": ", title, count=1) if ":" not in title else title
    title = re.sub(r"\s+:\s*", ": ", title)                  # "title : subtitle"
    title = re.sub(r"\s+", " ", title).strip(" -:;,/")
    if JUNK_TITLE.match(title) or (" " not in title and len(title) > 15) or re.search(r"\d{2,}_|titelblatt", title, re.I):
        return ""
    return title


def _junk_author(name: str) -> bool:
    low = name.lower().strip(" .")
    if JUNK_AUTHOR.match(low) or low in PUBLISHERS or len(low) < 3:
        return True
    if re.search(r"\d|www|@|\.com|\bpress\b|\bpublish|\buniversity\b|mcgraw|\balliance\b", low):
        return True
    return " " not in name and name.islower()  # account names such as 'brockur'


def clean_author(author: str) -> str:
    author = normalize_text(author or "")
    author = re.sub(r"^by\s+", "", author, flags=re.I)
    names: list[str] = []
    for part in re.split(r"\s*;\s*|\s+&\s+|\s+and\s+|\s*\|\s*|\s+_\s+", author):
        part = DATES.sub("", ROLE_WORDS.sub("", part))
        pieces = [p.strip(" ,") for p in part.split(",") if p.strip(" .,")]
        if not pieces:
            continue
        first = pieces[0].split()
        if len(pieces) >= 2 and (len(first) == 1 or (len(first) == 2 and first[0].lower() in NAME_PARTICLES)):
            suffix = [p for p in pieces[2:] if p.lower().strip(".") in NAME_SUFFIXES]
            found = [" ".join([pieces[1], pieces[0], *suffix])]  # 'Miller, Walter James' → 'Walter James Miller'
        else:
            found = pieces
        for name in found:
            name = re.sub(r"\s+", " ", name).strip(" .,;:-")
            if name and not _junk_author(name) and surname(name) not in {surname(n) for n in names}:
                names.append(name)
    return "; ".join(names)


def surname(name: str) -> str:
    tokens = [t for t in words(name) if t not in NAME_SUFFIXES]
    return tokens[-1] if tokens else ""


def shares_person(a: str, b: str) -> bool:
    """Some name in a and b has the same surname and the same first initial."""
    def people(author: str) -> set[tuple[str, str]]:
        found = set()
        for name in author.split(";"):
            tokens = [t for t in words(name) if t not in NAME_SUFFIXES]
            if tokens:
                found.add((tokens[-1], tokens[0][0] if len(tokens) > 1 else ""))
        return found

    return any(sa == sb and (not ia or not ib or ia == ib) for sa, ia in people(a) for sb, ib in people(b))


def surnames(author: str) -> frozenset[str]:
    return frozenset(surname(n) for n in author.split(";") if surname(n))


def title_score(title: str, front: set[str]) -> float:
    key = [w for w in title_key(title) if len(w) > 1 or w.isdigit()]
    return sum(w in front for w in key) / len(key) if key else 0.0


def author_score(author: str, front: set[str]) -> float:
    """Share of authors whose surname is on the first pages."""
    names = surnames(author)
    return sum(s in front for s in names) / len(names) if names else 0.0


def name_fullness(author: str, front: set[str]) -> float:
    """Share of all name tokens (first names too) on the first pages."""
    tokens = [t for t in words(author) if len(t) > 1]
    return sum(t in front for t in tokens) / len(tokens) if tokens else 0.0


def isbn_ok(isbn: str) -> bool:
    if len(isbn) == 10:
        total = sum((10 - i) * (10 if c in "Xx" else int(c)) for i, c in enumerate(isbn))
        return isbn[:9].isdigit() and total % 11 == 0
    if len(isbn) == 13 and isbn.isdigit():
        return sum((3 if i % 2 else 1) * int(c) for i, c in enumerate(isbn)) % 10 == 0
    return False


def find_isbns(text: str) -> list[str]:
    found = []
    for match in ISBN_RE.finditer(text):
        isbn = re.sub(r"[\s-]", "", match.group(1)).upper()
        if isbn_ok(isbn) and isbn not in found:
            found.append(isbn)
    return found


# ---------------------------------------------------------------------------
# Open Library
# ---------------------------------------------------------------------------

class OpenLibrary:
    """Cached, rate-limited Open Library client (one request per second)."""

    def __init__(self, offline: bool, log):
        import requests

        self.offline, self.log = offline, log
        self.cache: dict[str, object] = json.loads(CACHE_PATH.read_text("utf-8")) if CACHE_PATH.is_file() else {}
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.last = 0.0
        self.requests = 0

    def get(self, url: str):
        if url in self.cache:
            return self.cache[url]
        if self.offline:
            return None
        for attempt in range(3):
            time.sleep(max(0.0, 1.0 - (time.time() - self.last)))
            self.last = time.time()
            try:
                resp = self.session.get(url, timeout=30)
            except Exception as exc:
                self.log.error(f"open library: {url}: {exc}")
                time.sleep(5 * (attempt + 1))
                continue
            self.requests += 1
            if resp.status_code == 404:
                self.cache[url] = None
                return None
            if resp.ok:
                self.cache[url] = resp.json()
                return self.cache[url]
            time.sleep(5 * (attempt + 1))
        return None

    def save(self) -> None:
        CACHE_PATH.write_text(json.dumps(self.cache, ensure_ascii=False, indent=0), encoding="utf-8")

    def _author_names(self, refs: list) -> list[str]:
        names = []
        for ref in refs[:4]:
            key = (ref.get("author") or ref).get("key") if isinstance(ref, dict) else None
            data = self.get(f"{OL}{key}.json") if key else None
            if isinstance(data, dict) and data.get("name"):
                names.append(data["name"])
        return names

    def by_isbn(self, isbn: str) -> tuple[str, str]:
        edition = self.get(f"{OL}/isbn/{isbn}.json")
        if not isinstance(edition, dict) or not edition.get("title"):
            return "", ""
        title = edition["title"] + (f": {edition['subtitle']}" if edition.get("subtitle") else "")
        refs = edition.get("authors") or []
        if not refs and edition.get("works"):
            work = self.get(f"{OL}{edition['works'][0]['key']}.json")
            refs = work.get("authors", []) if isinstance(work, dict) else []
        return title, "; ".join(self._author_names(refs))

    def search(self, title: str, author: str) -> tuple[str, str]:
        from urllib.parse import urlencode

        params = {"title": " ".join(main_key(title)), "limit": 5, "fields": "title,subtitle,author_name"}
        if surnames(author):
            params["author"] = sorted(surnames(author))[0]
        data = self.get(f"{OL}/search.json?{urlencode(params)}")
        for doc in (data or {}).get("docs", []) if isinstance(data, dict) else []:
            found = doc.get("title", "") + (f": {doc['subtitle']}" if doc.get("subtitle") else "")
            if same_book(found, title):
                return found, "; ".join(doc.get("author_name", [])[:3])
        return "", ""


# ---------------------------------------------------------------------------
# Decision per book
# ---------------------------------------------------------------------------

def collect(row: dict, path: Path, ol: OpenLibrary | None, front: str) -> tuple[list[dict], str]:
    """Cleaned candidates [{source, title, author}] and the ISBN used."""
    emb_title, emb_author = embedded_metadata(path, row["format"])
    raw = [
        ("filename", row["file_title"], row["file_author"]),
        ("list", row["title"], row["author_name"]),
        ("embedded", emb_title, emb_author),
    ]
    isbn = ""
    if ol is not None:
        ol_title = ol_author = ""
        for candidate in find_isbns(front)[:3]:
            ol_title, ol_author = ol.by_isbn(candidate)
            if ol_title:
                isbn = candidate
                break
        if not ol_title:
            queries = [(row["title"] or row["file_title"], row["author_name"] or row["file_author"])]
            if emb_title and not same_book(emb_title, queries[0][0]):
                queries.append((emb_title, emb_author))
            for q_title, q_author in queries:
                q_title = clean_title(q_title, "filename")
                if q_title:
                    ol_title, ol_author = ol.search(q_title, clean_author(q_author))
                if ol_title:
                    break
        raw.append(("openlibrary", ol_title, ol_author))
    cands = []
    for source, title, author in raw:
        title, author = clean_title(title, source), clean_author(author)
        if title or author:
            cands.append({"source": source, "title": title, "author": author})
    return cands, isbn


def pick_wording(options: list[tuple[str, str]], current: str) -> str:
    """Among same-key texts keep the current one, else follow SOURCES order, avoiding all-lower/upper."""
    texts = [t for _, t in options]
    if current in texts:
        return current
    ranked = sorted(options, key=lambda o: (o[1].islower() or o[1].isupper(), SOURCES.index(o[0])))
    return ranked[0][1]


def decide(row: dict, cands: list[dict], front_words: set[str], cur_title: str, cur_author: str) -> dict:
    readable = len(front_words) >= MIN_FRONT_WORDS
    titles = [(c["source"], c["title"]) for c in cands if c["title"]]
    authors = [(c["source"], c["author"]) for c in cands if c["author"]]
    cur_t_clean = clean_title(cur_title, "filename")
    cur_a_clean = clean_author(cur_author)
    reasons = []

    def agree_title(title: str) -> list[str]:
        return sorted({s for s, t in titles if title_key(t) == title_key(title)})

    def agree_author(author: str) -> list[str]:
        return sorted({s for s, a in authors if surnames(a) == surnames(author)})

    # --- title
    cur_score = title_score(cur_t_clean, front_words) if readable else 0.0
    prop_t, identity = cur_t_clean or cur_title, False
    if readable:
        verified = [(s, t) for s, t in titles if title_score(t, front_words) >= VERIFIED]
        if cur_score >= VERIFIED or (cur_t_clean and title_score(" ".join(main_key(cur_t_clean)), front_words) >= VERIFIED):
            cur_main = main_key(cur_t_clean)
            longer = [
                (s, t) for s, t in verified
                if len(title_key(t)) > len(title_key(cur_t_clean)) and title_key(t)[: len(cur_main)] == cur_main
            ]
            if longer:
                best_len = max(len(title_key(t)) for _, t in longer)
                best = [(s, t) for s, t in longer if len(title_key(t)) == best_len]
                prop_t = pick_wording(best, cur_t_clean)
                reasons.append(f"title extended ({','.join(agree_title(prop_t))})")
        elif cur_score < UNVERIFIED:
            other = [(s, t) for s, t in verified if not same_book(t, cur_t_clean)]
            if other:
                key = max(other, key=lambda o: (title_score(o[1], front_words), len(title_key(o[1]))))[1]
                prop_t = pick_wording([o for o in other if title_key(o[1]) == title_key(key)], cur_t_clean)
                identity = True
                reasons.append(f"current title not in text; text matches '{prop_t}' ({','.join(agree_title(prop_t))})")
    else:
        cur_main = main_key(cur_t_clean)
        backed = [
            (s, t) for s, t in titles
            if len(agree_title(t)) >= 2 and len(title_key(t)) > len(title_key(cur_t_clean)) and title_key(t)[: len(cur_main)] == cur_main
        ]
        if backed:
            best_len = max(len(title_key(t)) for _, t in backed)
            prop_t = pick_wording([o for o in backed if len(title_key(o[1])) == best_len], cur_t_clean)
            reasons.append(f"title extended, unverified ({','.join(agree_title(prop_t))})")
    if title_key(prop_t) == title_key(cur_t_clean):
        options = [(s, t) for s, t in titles if title_key(t) == title_key(cur_t_clean) and s != "filename"]
        if "_" in cur_title and options:
            prop_t = pick_wording(options, "")
        elif "_" not in cur_title:
            prop_t = cur_title if title_key(prop_t) == title_key(cur_title) else prop_t

    # --- author
    pool = authors
    if identity:  # authors must belong to the book the text actually is
        pool = [(s, a) for s, a in authors if any(same_book(t, prop_t) for s2, t in titles if s2 == s)] or [
            (s, a) for s, a in authors if author_score(a, front_words) == 1.0
        ]
    prop_a = cur_a_clean or cur_author
    if readable:
        cur_ok = bool(cur_a_clean) and author_score(cur_a_clean, front_words) == 1.0 and not identity
        verified_a = [(s, a) for s, a in pool if author_score(a, front_words) == 1.0]
        if cur_ok:
            better = [
                (s, a) for s, a in verified_a
                if surnames(a) > surnames(cur_a_clean)
                or (surnames(a) == surnames(cur_a_clean) and name_fullness(a, front_words) > name_fullness(cur_a_clean, front_words))
            ]
            if better:
                prop_a = max(better, key=lambda o: (len(surnames(o[1])), name_fullness(o[1], front_words), -SOURCES.index(o[0])))[1]
                reasons.append(f"author completed ({','.join(agree_author(prop_a))})")
        elif verified_a:
            prop_a = max(verified_a, key=lambda o: (len(surnames(o[1])), name_fullness(o[1], front_words), -SOURCES.index(o[0])))[1]
            reasons.append(f"author on title page ({','.join(agree_author(prop_a))})")
        elif identity and pool:
            prop_a = min(pool, key=lambda o: SOURCES.index(o[0]))[1]
            reasons.append(f"author from the same source as the title, not on title page ({','.join(agree_author(prop_a))})")
        elif identity:
            prop_a = ""
            reasons.append("author of the actual book not found")
    elif not cur_a_clean:
        backed = [(s, a) for s, a in pool if len(agree_author(a)) >= 2] or pool
        if backed:
            prop_a = min(backed, key=lambda o: SOURCES.index(o[0]))[1]
            reasons.append(f"author was missing, unverified ({','.join(agree_author(prop_a))})")
    if surnames(prop_a) == surnames(cur_a_clean) and name_fullness(prop_a, front_words) <= name_fullness(cur_a_clean, front_words):
        if cur_a_clean:  # same people: keep the current spelling, only fix 'Last, First' and list separators
            prop_a = cur_a_clean if "," in cur_author else cur_author
    if readable and not identity and cur_a_clean and prop_a and not shares_person(prop_a, cur_a_clean):
        identity = True  # same title, different author on the title page: another book with that title
        reasons.append("title page names a different author: probably another book with the same title")

    # --- confidence
    t_changed = prop_t != cur_title
    a_changed = prop_a != cur_author
    t_ok = readable and title_score(prop_t, front_words) >= VERIFIED
    a_ok = readable and author_score(prop_a, front_words) == 1.0
    disagree = readable and cur_score < UNVERIFIED and any(not same_book(t, cur_t_clean) for _, t in titles)
    if identity:
        confidence = "wrong_book"
    elif not (t_changed or a_changed):
        if not disagree:
            return {}
        confidence = "conflict"
        reasons.append("current title not in text; sources disagree, none verified")
    elif readable and not (t_ok and a_ok) and (t_changed and not t_ok or a_changed and not a_ok):
        confidence = "conflict"
        reasons.append("change not confirmed by title page")
    elif (
        t_ok and a_ok
        and (not t_changed or len(agree_title(prop_t)) >= 2 or title_key(prop_t) == title_key(cur_t_clean))
        and (not a_changed or len(agree_author(prop_a)) >= 2 or surnames(prop_a) == surnames(cur_a_clean) and cur_a_clean)
    ):
        confidence = "high"
    else:
        confidence = "medium"
    if t_changed and not reasons:
        reasons.append("punctuation restored")
    if a_changed and not any(r.startswith("author") for r in reasons):
        reasons.append("author name cleaned")
    return {
        "proposed_title": prop_t,
        "proposed_author": prop_a,
        "confidence": confidence,
        "title_score": round(title_score(prop_t, front_words), 2) if readable else "",
        "author_score": round(author_score(prop_a, front_words), 2) if readable else "",
        "title_sources": ",".join(agree_title(prop_t)),
        "author_sources": ",".join(agree_author(prop_a)),
        "reason": "; ".join(reasons),
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def suggest(args) -> None:
    log = setup_logger("fix_titles_suggest")
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    books = manifest[manifest["lang"] == "en"]
    if args.book_id:
        books = books[books["book_id"].isin(args.book_id)]
    if args.limit:
        books = books.head(args.limit)
    overrides = load_metadata_overrides()
    previous = {}
    if REVIEW_PATH.is_file():
        old = pd.read_csv(REVIEW_PATH, dtype=str).fillna("")
        previous = {r["book_id"]: r for r in old.to_dict(orient="records")}
    ol = OpenLibrary(args.offline, log)
    log.info(f"{len(books)} English books; open library {'offline (cache only)' if args.offline else 'online'}")

    files = raw_files_by_md5()
    rows = []
    for i, row in enumerate(books.to_dict(orient="records"), 1):
        path = files.get(row["md5"])
        if path is None:
            log.error(f"file not found under data/raw/books: {row['book_id']} ({row['raw_path']})")
            continue
        moved = path.name != Path(row["raw_path"]).name or path.parent.name != Path(row["raw_path"]).parent.name
        cur_title, cur_author = display_title_author(row, overrides)
        if moved and row["book_id"] not in overrides:  # renamed by hand: that name is the current one
            parsed = parse_filename(path.name)
            cur_title, cur_author = parsed["file_title"] or cur_title, parsed["file_author"] or cur_author
        try:
            front = front_text(path, row["format"])
        except Exception as exc:
            log.error(f"front text failed: {row['book_id']}: {exc}")
            front = ""
        cands, isbn = collect(row, path, ol, front)
        result = decide(row, cands, set(words(front)), cur_title, cur_author)
        if result:
            by_source = {c["source"]: c for c in cands}
            out = {
                "book_id": row["book_id"],
                "corpus_role": row["corpus_role"],
                "format": row["format"],
                "title_match": row["title_match"],
                "current_title": cur_title,
                "current_author": cur_author,
                **{f"{s}_{f}": by_source.get(s, {}).get(f, "") for s in SOURCES for f in ("title", "author")},
                "isbn": isbn,
                "front_words": len(front.split()),
                **result,
                "accept": "",
                "manually_moved": moved,
                "raw_path": str(path.relative_to(ROOT)),
            }
            prev = previous.get(row["book_id"])
            if prev and (prev["proposed_title"], prev["proposed_author"]) == (out["proposed_title"], out["proposed_author"]):
                out["accept"] = prev.get("accept", "")
            rows.append(out)
        if i % 25 == 0:
            ol.save()
            log.info(f"  {i}/{len(books)}  ({len(rows)} suggestions, {ol.requests} requests)")
    ol.save()

    review = pd.DataFrame(rows)
    if not review.empty:
        review = review.sort_values(["confidence", "book_id"], key=lambda s: s.map(CONFIDENCE_ORDER) if s.name == "confidence" else s)
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8")
    log.info(f"wrote {REVIEW_PATH.relative_to(ROOT)} ({len(review)} rows, {ol.requests} open library requests)")
    if not review.empty:
        log.info("by confidence:\n" + review["confidence"].value_counts().to_string())
    log.close()


def apply(_args) -> None:
    log = setup_logger("fix_titles_apply")
    review = pd.read_csv(REVIEW_PATH, dtype=str).fillna("")
    existing = pd.read_csv(METADATA_OVERRIDES, dtype=str).fillna("") if METADATA_OVERRIDES.is_file() else pd.DataFrame(columns=OVERRIDE_COLS)
    records = {r["book_id"]: r for r in existing.to_dict(orient="records")}

    counts = {"auto-high": 0, "manual": 0, "skipped": 0}
    for row in review.to_dict(orient="records"):
        accept = row["accept"].strip().lower()
        if accept in {"yes", "y"}:
            reviewer = "manual"
        elif accept == "" and row["confidence"] == "high":
            reviewer = "auto-high"
        else:
            counts["skipped"] += 1
            continue
        wrong = row["confidence"] == "wrong_book"
        records[row["book_id"]] = {
            "book_id": row["book_id"],
            "title": row["proposed_title"].strip(),
            "author": row["proposed_author"].strip(),
            "source": ",".join(dict.fromkeys(s for x in (row["title_sources"], row["author_sources"]) for s in x.split(",") if s)),
            "confidence": row["confidence"],
            "identity_changed": str(wrong),
            "reviewer": reviewer,
            "note": "file is a different book than its old name; re-check corpus_role" if wrong else row["reason"],
        }
        counts[reviewer] += 1

    out = pd.DataFrame(list(records.values()), columns=OVERRIDE_COLS).sort_values("book_id")
    out.to_csv(METADATA_OVERRIDES, index=False, encoding="utf-8")
    log.info(f"wrote {METADATA_OVERRIDES.relative_to(ROOT)} ({len(out)} overrides): {counts}")
    log.info("next: python -m src.fulltext.reorganize_books --dry-run")
    log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("suggest", help="write the review CSV")
    s.add_argument("--offline", action="store_true", help="use only cached Open Library responses")
    s.add_argument("--limit", type=int, help="first N books only (testing)")
    s.add_argument("--book-id", nargs="*", help="only these book_ids")
    sub.add_parser("apply", help="copy accepted rows into config/book_metadata_overrides.csv")
    args = parser.parse_args()
    {"suggest": suggest, "apply": apply}[args.command](args)


if __name__ == "__main__":
    main()
