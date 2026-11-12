"""Book-level statistics for the English primary and comparison books.

Builds one row per collected file (505 at the time of writing) with:

- what the manifest already knows (folder label, format, edition year, words),
- quality flags (needs OCR, short text, possible duplicate, label conflict),
- one flag `in_analysis_set` that says whether the book enters the headline figures,
- exploratory counts of point-of-view (POV) vocabulary per 10,000 words.

The output holds counts and catalogue metadata only, never book text, so it
can be tracked in git and the notebook can be re-run without the full texts:

  data/processed/en_book_stats.csv
  data/processed/en_book_stats.meta.json   (when and from what it was built)

Usage:
  python -m src.analysis.corpus_stats               # build the table (about 2 minutes)
  python -m src.analysis.corpus_stats --selftest    # check the POV matcher only
  python -m src.analysis.corpus_stats --spotcheck 40  # print matched snippets (not saved)

How POV vocabulary is counted (exploratory, lexicon v0.1, not validated):
for each dimension in config/pov_lexicon.yml we take the English `high`
terms and the English `regex` patterns, and count their occurrences in the
body text (front and back matter are skipped). Spaces and hyphens are
interchangeable ("first person" = "first-person"), matching ignores case and
a plural -s is allowed. An overall count merges overlapping matches, so
"third-person point of view" is one occurrence, not two.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache

import pandas as pd
import yaml

from src.common.logging_util import setup_logger
from src.common.normalize import normalize_title
from src.common.paths import CONFIG, DATA_INTERIM, DATA_PROCESSED, ROOT

MANIFEST_PATH = DATA_INTERIM / "fulltext" / "manifest.csv"
TITLE_REVIEW_PATH = DATA_INTERIM / "fulltext" / "title_author_review.csv"
METADATA_OVERRIDES_PATH = CONFIG / "book_metadata_overrides.csv"
PRIMARY_CHECK_PATH = DATA_INTERIM / "fulltext" / "primary_check.csv"  # src.fulltext.prescreen_books
JSON_DIR = DATA_INTERIM / "fulltext" / "json"
LEXICON_PATH = CONFIG / "pov_lexicon.yml"
STATS_PATH = DATA_PROCESSED / "en_book_stats.csv"
META_PATH = DATA_PROCESSED / "en_book_stats.meta.json"
BIBLIO_PATH = DATA_PROCESSED / "en_primary_comparison_biblio.csv"  # src.publication.build_en_biblio_timeline

LANG = "en"
ROLES = ["primary", "comparison"]
SHORT_TEXT_WORDS = 10_000  # below this the file is too short to be a whole manual

# Exclusion steps, applied in this order. A book is counted under the first that applies.
EXCLUSION_STEPS = {
    "needs_ocr": "Scanned PDF, no text yet (needs OCR)",
    "short_text": f"Text shorter than {SHORT_TEXT_WORDS:,} words (leaflet or partial file)",
    "role_conflict": "Same work filed under both primary and comparison",
    "surplus_duplicate": "Second file of a work already counted",
}

# "Discusses POV substantially" has no agreed cut-off yet. These two are illustrations only.
ILLUSTRATIVE_THRESHOLDS = [1, 5]
# Label sources that count as checked (see src/fulltext/README.md, "Labels").
CHECKED_ROLE_SOURCES = ["manual:folder", "review:taxonomy-2026-11"]

# Plain-language names for the tables and figures.
ROLE_LABELS = {"primary": "Primary", "comparison": "Comparison"}
# Robustness cut inside primary (both stay corpus_role=primary).
PRIMARY_GROUP_LABELS = {
    "fiction_manual": "Fiction-writing manuals",
    "pedagogy_genre_craft": "Pedagogy / genre craft",
    "contrast": "Contrast (narratology / academic)",
}
# Override notes that mark books moved into primary as pedagogy or genre craft (not core manuals).
_PEDAGOGY_MOVE_NOTES = {
    "narrative-writing primary definition (2026-11-12)",
    "narrative-writing def: screenwriting craft",
    "narrative-writing def: poetry craft",
}
_PEDAGOGY_NOTE_RE = re.compile(
    r"journal writing|non-fiction writing craft|Creative Nonfiction|CW pedagogy|"
    r"screenwriting craft|poetry craft",
    re.IGNORECASE,
)
_SCHOOL_TITLE_RE = re.compile(
    r"teach|classroom|pedagog|lesson|curriculum|for teachers|writing workshop|"
    r"grade \d|students?\b|pupils?|\bschool\b",
    re.IGNORECASE,
)
_THERAPY_TITLE_RE = re.compile(r"therap|expressive writing|journal writing|healing", re.IGNORECASE)
_GENRE_TITLE_RULES = ("screenwriting", "poetry", "playwriting", "creative_nonfiction")
ROLE_SOURCE_LABELS = {
    "prescreen": "Automatic prescreen, not checked",
    "review:agent-firstpass": "AI first-pass proposal, not confirmed",
    "review:taxonomy-2026-11": "Taxonomy review, November 2026",
    "review:agent-2026-11-12": "AI title check, 12 November 2026, not confirmed",
    "manual:folder": "Placed by hand",
}
IDENTITY_LABELS = {
    "": "File and catalogue record agree (as far as checked)",
    "renamed": "File renamed after checking it; record may describe another title",
    "unresolved": "Open doubt whether file and record are the same book",
}
YEAR_SOURCE_LABELS = {
    "candidate_list": "Year from the candidate-list record",
    "other_file_of_same_work": "Year from another file of the same work",
    "record_is_another_title": "Year not used: file was renamed, record may describe another title",
    "record_not_verified": "Year not used: open doubt about the record",
    "missing": "No catalogue record, no year",
}
# Work years we trust: the seed bibliography, or Open Library joined by work id (not by title).
CLEAN_WORK_YEAR_SOURCES = ["seed", "ol_search"]

# Informal reading of `--spotcheck 40` (seed 20261112, table built 2026-11-12): 20 random
# matches per corpus, each judged as "about narrative point of view" or "everyday sense"
# (an opinion, grammatical person in a grammar lesson, a camera direction in a script).
# One reader (the AI assistant that wrote this module), no second opinion: a rough
# indication only. Re-run the command and replace the numbers after your own reading.
# Read before the folders were re-sorted on the same day (505 files, not 494).
SPOTCHECK = {
    "date": "2026-11-12, before the re-sort of the folders",
    "reader": "AI assistant, single reading",
    "primary": {"read": 20, "not_narrative": 2},
    "comparison": {"read": 20, "not_narrative": 9},
}
# Comparison is the small contrast group (narratology + academic writing). Older reasons
# remain labelled for any leftover rows.
EXCLUSION_REASON_LABELS = {
    "narratology_theory": "Narratology / narrative theory",
    "academic_writing": "Academic / composition writing",
    "research_discipline": "Research on creative writing",
    "academic_study": "Scholarship on literature / writing",
    "discipline_writing": "Writing for law, history, science",
    "school_pedagogy": "School pedagogy (now primary)",
    "screenwriting_only": "Screenwriting (now primary)",
    "therapy": "Therapeutic writing (now primary)",
    "creative_nonfiction": "Creative nonfiction (now primary)",
    "poetry_only": "Poetry (now primary)",
    "playwriting_only": "Playwriting (now primary)",
    "journalism": "Journalism",
    "songwriting_only": "Songwriting",
    "other": "Other",
}
# One plain sentence per aspect, for readers who do not work with narratology every day.
ASPECT_IN_WORDS = {
    "general_pov": "The topic named as such: point of view, viewpoint.",
    "person": "Whether the story is told as I, you, or he/she.",
    "knowledge_focalization": "How much the narrator knows: omniscient, limited, focalization.",
    "thought_representation": "How a character's thoughts are written (interior monologue, free indirect style).",
    "consciousness_access": "Whether the text can enter a character's mind.",
    "multiplicity_switching": "More than one viewpoint, and changing between them.",
    "reliability": "Whether the narrator can be trusted.",
    "distance_depth": "How close the telling stays to a character (close third, deep POV).",
    "objective_mode": "Telling only what a camera could see.",
    "involvement_voice": "Whether the narrator steps in, and what the voice sounds like.",
    "information_control": "Who is allowed to know what, and when the reader learns it.",
    "narrator_role": "Who is telling: a witness, the protagonist, someone at the edge of the story.",
}
DIMENSION_LABELS = {
    "general_pov": "Point of view in general",
    "person": "Grammatical person (first / third ...)",
    "knowledge_focalization": "Narrator knowledge (omniscient, limited, focalization)",
    "thought_representation": "Thought representation (interior monologue ...)",
    "consciousness_access": "Access to a character's mind",
    "multiplicity_switching": "Multiple viewpoints and switching",
    "reliability": "Narrator reliability",
    "distance_depth": "Distance and depth (close third, deep POV)",
    "objective_mode": "Objective / camera mode",
    "involvement_voice": "Narrator involvement and voice",
    "information_control": "Information control (who knows what)",
    "narrator_role": "Narrator role (witness, protagonist ...)",
}

# Text clean-up before matching: curly quotes and Unicode hyphens become plain ones,
# soft hyphens disappear. Paragraphs are joined with a mark that no term can cross.
_TRANSLATE = str.maketrans({"\u2019": "'", "\u2018": "'", "\u2010": "-", "\u2011": "-",
                            "\u2012": "-", "\u00ad": None})
_PARAGRAPH_MARK = " \u00b6 "
_SEPARATOR = r"[\s\-]+"
_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "phd"}


# --------------------------------------------------------------------------- POV matcher

def _term_pattern(term: str) -> str:
    """Turn a lexicon term into a pattern where spaces and hyphens are interchangeable."""
    parts = [p for p in re.split(r"[\s\-]+", term.strip()) if p]
    return _SEPARATOR.join(re.escape(p) for p in parts)


def build_dimension_patterns(lexicon: dict, lang: str = LANG) -> dict[str, re.Pattern]:
    """One compiled pattern per POV dimension: high-precision terms plus the regex list."""
    patterns = {}
    for name, spec in lexicon["dimensions"].items():
        terms = sorted({t for t in (spec.get(lang, {}).get("high") or []) if t}, key=len, reverse=True)
        alternatives = []
        if terms:
            alternatives.append(r"(?<!\w)(?:" + "|".join(_term_pattern(t) for t in terms) + r")s?(?!\w)")
        alternatives += [f"(?:{rx})" for rx in (spec.get("regex", {}).get(lang) or [])]
        if alternatives:
            patterns[name] = re.compile("|".join(alternatives), re.IGNORECASE)
    return patterns


@lru_cache(maxsize=1)
def dimension_patterns() -> dict[str, re.Pattern]:
    with open(LEXICON_PATH, encoding="utf-8") as fh:
        return build_dimension_patterns(yaml.safe_load(fh))


def clean_text(text: str) -> str:
    """Normalize quotes, hyphens and whitespace so that matching is stable."""
    return " ".join(text.translate(_TRANSLATE).split())


def find_matches(text: str, patterns: dict[str, re.Pattern] | None = None):
    """Return (count per dimension, merged spans) for one cleaned piece of text.

    Merged spans are (start, end, dimensions): overlapping matches from different
    dimensions are fused, so each stretch of text is counted once overall.
    """
    patterns = patterns or dimension_patterns()
    per_dimension = {}
    spans = []
    for name, pattern in patterns.items():
        found = [(m.start(), m.end(), name) for m in pattern.finditer(text)]
        per_dimension[name] = len(found)
        spans += found
    spans.sort()
    merged = []
    for start, end, name in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]), merged[-1][2] | {name})
        else:
            merged.append((start, end, {name}))
    return per_dimension, merged


SELFTEST_POSITIVE = {
    "general_pov": ["Choose the point of view before you draft.", "Stay with one viewpoint character.",
                    "Two POVs are enough."],
    "person": ["A first person narrator reports only what she sees.", "Try the third-person point of view."],
    "narrator_role": ["Nick is a peripheral narrator.", "This is first person peripheral narration."],
    "knowledge_focalization": ["The limited-omniscient narrator knows one mind.", "Genette calls it internal focalisation."],
    "consciousness_access": ["We are inside the character\u2019s head.", "Interiority slows the scene."],
    "distance_depth": ["Gardner calls this psychic distance.", "Deep POV removes filter words."],
    "objective_mode": ["It is a fly-on-the-wall account.", "The camera eye records without comment."],
    "involvement_voice": ["An intrusive narrator comments on events.", "Listen to the narrator\u2019s voice."],
    "multiplicity_switching": ["Editors dislike head hopping.", "A novel with multiple points of view."],
    "reliability": ["Unreliable narrators hide things.", "Is she a reliable narrator?"],
    "thought_representation": ["Austen uses free-indirect discourse.", "A page of stream of consciousness."],
    "information_control": ["Decide what the reader knows at each point."],
}
SELFTEST_NEGATIVE = [
    "From a different perspective, voice and tone matter most.",
    "The point is that viewers like the distance between the characters.",
    "Tell the story in your own words.",
]


def selftest() -> list[str]:
    """Check the matcher on made-up sentences. Returns a list of problems (empty = fine)."""
    patterns = dimension_patterns()
    problems = []
    for name in patterns:
        if name not in SELFTEST_POSITIVE:
            problems.append(f"no self-test sentence for dimension {name}")
    for name, sentences in SELFTEST_POSITIVE.items():
        for sentence in sentences:
            counts, _ = find_matches(clean_text(sentence), patterns)
            if counts.get(name, 0) < 1:
                problems.append(f"{name}: no match in {sentence!r}")
    for sentence in SELFTEST_NEGATIVE:
        _, merged = find_matches(clean_text(sentence), patterns)
        if merged:
            problems.append(f"unexpected match in {sentence!r}")
    counts, merged = find_matches(clean_text("Point of view matters. POV is point-of-view."), patterns)
    if counts["general_pov"] != 3 or len(merged) != 3:
        problems.append("counting: expected 3 occurrences of general_pov")
    counts, merged = find_matches(clean_text("She chose the third-person point of view."), patterns)
    if not (counts["person"] == 1 and counts["general_pov"] == 1 and len(merged) == 1):
        problems.append("overlap: 'third-person point of view' must be one overall occurrence")
    return problems


# --------------------------------------------------------------------------- per-book counts

def _body_chapters(book_id: str):
    """Yield the cleaned text of each body chapter and its word count."""
    with open(JSON_DIR / LANG / f"{book_id}.json", encoding="utf-8") as fh:
        book = json.load(fh)
    for chapter in book["chapters"]:
        if chapter["kind"] != "body":
            continue
        paragraphs = [clean_text(p) for p in chapter["paragraphs"]]
        yield _PARAGRAPH_MARK.join(paragraphs), sum(len(p.split()) for p in paragraphs)


def count_book(book_id: str) -> dict:
    """POV counts for one book. Runs in a worker process."""
    patterns = dimension_patterns()
    row = {"book_id": book_id, "body_words": 0, "pov_n": 0, **{f"pov_{d}_n": 0 for d in patterns}}
    for text, words in _body_chapters(book_id):
        per_dimension, merged = find_matches(text, patterns)
        row["body_words"] += words
        row["pov_n"] += len(merged)
        for name, n in per_dimension.items():
            row[f"pov_{name}_n"] += n
    return row


# --------------------------------------------------------------------------- flags

def _surname(author) -> str:
    """Surname of the first author, lower-case; empty when unknown."""
    if not isinstance(author, str):
        return ""
    first = re.split(r";|_| and |&", author)[0]
    tokens = [t for t in normalize_title(first).split() if t not in _NAME_SUFFIXES]
    return "" if not tokens or tokens == ["unknown"] else tokens[-1]


def _work_key(title, author) -> str:
    """Matching key for 'probably the same work': title before a subtitle + author surname."""
    surname = _surname(author)
    base = normalize_title(re.split(r":| - ", str(title))[0])
    return f"{base}|{surname}" if base and surname else ""


def add_titles_and_identity(books: pd.DataFrame) -> pd.DataFrame:
    """Add the current title/author and say how far the catalogue record can be trusted.

    Some downloaded files turned out to be a different book than their catalogue
    record (see src/fulltext/README.md, "Titles and authors"). The current name
    comes from config/book_metadata_overrides.csv, exactly as for the filenames.

    identity_check:
      ""           file and catalogue record agree as far as we know
      "renamed"    the override table marks the title as changed (`identity_changed`). Some of
                   these files are a different book than their record, others only got a
                   fuller title; the table does not say which, so the record's year is not used
      "unresolved" an open "wrong book" suggestion, or the title words are not in the text
    """
    from src.fulltext.books import display_title_author, load_metadata_overrides  # light import, only for the build

    books = books.copy()
    names = load_metadata_overrides()
    pairs = [display_title_author(row, names) for row in books.to_dict("records")]
    books["title"] = [title for title, _ in pairs]
    books["author"] = [author for _, author in pairs]

    overrides = pd.read_csv(METADATA_OVERRIDES_PATH, dtype=str).fillna("")
    renamed = set(overrides.loc[overrides["identity_changed"].str.lower() == "true", "book_id"])
    review = pd.read_csv(TITLE_REVIEW_PATH, dtype=str).fillna("")
    open_suggestion = set(review.loc[(review["confidence"] == "wrong_book") & (review["accept"] == ""), "book_id"])
    title_mismatch = ~books["title_match"].astype(bool)
    books["identity_check"] = ""
    books.loc[books["book_id"].isin(open_suggestion) | title_mismatch, "identity_check"] = "unresolved"
    books.loc[books["book_id"].isin(renamed), "identity_check"] = "renamed"
    return books


def _load_genre_title_patterns() -> dict[str, re.Pattern]:
    with open(CONFIG / "book_prescreen.yml", encoding="utf-8") as fh:
        rules = yaml.safe_load(fh)["title_rules"]
    return {name: re.compile("|".join(rules[name]), re.IGNORECASE)
            for name in _GENRE_TITLE_RULES if name in rules}


def add_primary_group(books: pd.DataFrame) -> pd.DataFrame:
    """Split primary into fiction manuals vs pedagogy/genre craft (robustness cut, not comparison).

    Pedagogy/genre craft: screen / poetry / drama / CNF title, school or therapy title cues,
    or an override that moved the book from the old comparison reasons into primary.
    Comparison books are labelled `contrast`. Both primary groups keep corpus_role=primary.
    """
    from src.common.prescreen import load_screen_rules, prescreen

    books = books.copy()
    genre = _load_genre_title_patterns()
    screen_rules = load_screen_rules()
    overrides = pd.read_csv(CONFIG / "book_role_overrides.csv", dtype=str).fillna("")
    notes = overrides.set_index("book_id")["note"] if len(overrides) else pd.Series(dtype=str)

    groups, reasons = [], []
    for row in books.to_dict(orient="records"):
        if row["corpus_role"] != "primary":
            groups.append("contrast")
            reasons.append("comparison")
            continue
        title = row["title"] or ""
        status, rule = prescreen(title, "", screen_rules)
        hit = ""
        for name, pattern in genre.items():
            if pattern.search(title):
                hit = name
                break
        if not hit and status == "likely_contextual":
            hit = rule
        if not hit and _SCHOOL_TITLE_RE.search(title):
            hit = "school_title"
        if not hit and _THERAPY_TITLE_RE.search(title):
            hit = "therapy_title"
        note = notes.get(row["book_id"], "") if len(notes) else ""
        if not hit and (note in _PEDAGOGY_MOVE_NOTES or _PEDAGOGY_NOTE_RE.search(note or "")):
            hit = "override_move"
        if hit:
            groups.append("pedagogy_genre_craft")
            reasons.append(hit)
        else:
            groups.append("fiction_manual")
            reasons.append("")
    books["primary_group"] = groups
    books["primary_group_reason"] = reasons
    return books


def add_flags(books: pd.DataFrame) -> pd.DataFrame:
    """Add quality flags, the duplicate grouping and `in_analysis_set`."""
    books = books.copy()
    books["needs_ocr"] = books["extraction_status"].ne("ok")
    books["short_text"] = ~books["needs_ocr"] & books["n_words"].lt(SHORT_TEXT_WORDS)

    # Group files that look like the same work (same key, or identical text per the manifest).
    parent = {b: b for b in books["book_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = books.assign(key=[_work_key(t, a) for t, a in zip(books["title"], books["author"])])
    for _, ids in keys[keys["key"] != ""].groupby("key")["book_id"]:
        ids = list(ids)
        for other in ids[1:]:
            parent[find(other)] = find(ids[0])
    for book_id, twin in zip(books["book_id"], books["content_dup_of"]):
        if isinstance(twin, str) and twin in parent:
            parent[find(book_id)] = find(twin)
    books["work_group"] = [find(b) for b in books["book_id"]]
    group_size = books.groupby("work_group")["book_id"].transform("size")
    books["possible_duplicate"] = group_size > 1
    books["role_conflict"] = books["possible_duplicate"] & (
        books.groupby("work_group")["corpus_role"].transform("nunique") > 1)

    # Within each group keep the longest usable text; the others are surplus.
    usable = ~books["needs_ocr"] & ~books["short_text"] & ~books["role_conflict"]
    rank = books[usable].groupby("work_group")["n_words"].rank(method="first", ascending=False)
    books["surplus_duplicate"] = False
    books.loc[rank[rank > 1].index, "surplus_duplicate"] = True

    books["exclusion_step"] = ""
    for step in reversed(list(EXCLUSION_STEPS)):  # reversed so the first step wins
        books.loc[books[step], "exclusion_step"] = step
    books["in_analysis_set"] = books["exclusion_step"].eq("")

    # Edition year comes from the candidate-list record. It is only used when nothing speaks
    # against that record belonging to the file: not when the file was renamed (the year may
    # describe the old title), and not while a "wrong book" doubt is open.
    # A file without a year borrows it from another file of the same work.
    books["edition_year"] = books["publication_year"].astype("Int64")
    books["year_source"] = books["edition_year"].notna().map({True: "candidate_list", False: "missing"})
    for check, source in [("renamed", "record_is_another_title"), ("unresolved", "record_not_verified")]:
        drop = books["identity_check"].eq(check) & books["edition_year"].notna()
        books.loc[drop, "edition_year"] = pd.NA
        books.loc[drop, "year_source"] = source
    group_year = books.groupby("work_group")["edition_year"].transform("min")
    borrow = books["edition_year"].isna() & group_year.notna()
    books.loc[borrow, "edition_year"] = group_year[borrow]
    books.loc[borrow, "year_source"] = "other_file_of_same_work"
    books["work_group"] = books["work_group"].where(books["possible_duplicate"], "")
    return books


# --------------------------------------------------------------------------- build

def load_manifest_subset() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST_PATH, low_memory=False)
    return manifest[(manifest["lang"] == LANG) & manifest["corpus_role"].isin(ROLES)].reset_index(drop=True)


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build(workers: int) -> pd.DataFrame:
    log = setup_logger("corpus_stats")
    problems = selftest()
    if problems:
        for p in problems:
            log.error(f"matcher self-test: {p}")
        log.close()
        raise SystemExit("POV matcher self-test failed; no table written.")
    log.info(f"Matcher self-test passed ({len(dimension_patterns())} dimensions).")

    books = add_primary_group(add_flags(add_titles_and_identity(load_manifest_subset())))
    log.info(f"Files: {len(books)} " + str(books["corpus_role"].value_counts().to_dict()))
    primary = books[books["corpus_role"] == "primary"]
    log.info("Primary robustness split: " + str(primary["primary_group"].value_counts().to_dict()))

    extracted = books.loc[~books["needs_ocr"], "book_id"].tolist()
    log.info(f"Counting POV vocabulary in {len(extracted)} extracted books with {workers} workers ...")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        counts = pd.DataFrame(list(pool.map(count_book, extracted, chunksize=8)))
    books = books.merge(counts, on="book_id", how="left")
    books["pov_per_10k"] = (books["pov_n"] / books["body_words"] * 10_000).round(3)

    dims = list(dimension_patterns())
    for name in dims:
        if books[f"pov_{name}_n"].sum() == 0:
            log.error(f"WARNING: dimension {name} has zero matches in the whole corpus; check the lexicon/matcher.")
    if (books.loc[~books["needs_ocr"], "body_words"] == 0).any():
        log.error("WARNING: some extracted books have no body text.")

    candidate_source = books["candidate_id"].str.split(":").str[0].fillna("not_in_lists")
    out = pd.DataFrame({
        "book_id": books["book_id"],
        "corpus_role": books["corpus_role"],
        "primary_group": books["primary_group"],
        "primary_group_reason": books["primary_group_reason"],
        "title": books["title"],
        "author": books["author"],
        "format": books["format"],
        "edition_year": books["edition_year"],
        "year_source": books["year_source"],
        "candidate_source": candidate_source,
        "role_source": books["role_source"],
        "exclusion_reason": books["exclusion_reason"],
        "corpus_type": books["corpus_type"],
        "n_words": books["n_words"].astype("Int64"),
        "body_words": books["body_words"].astype("Int64"),
        "n_chapters": books["n_chapters"].astype("Int64"),
        "needs_ocr": books["needs_ocr"],
        "short_text": books["short_text"],
        "possible_duplicate": books["possible_duplicate"],
        "work_group": books["work_group"],
        "role_conflict": books["role_conflict"],
        "surplus_duplicate": books["surplus_duplicate"],
        "identity_check": books["identity_check"],
        "exclusion_step": books["exclusion_step"],
        "in_analysis_set": books["in_analysis_set"],
        "pov_n": books["pov_n"].astype("Int64"),
        "pov_per_10k": books["pov_per_10k"],
        **{f"pov_{d}_n": books[f"pov_{d}_n"].astype("Int64") for d in dims},
    }).sort_values(["corpus_role", "book_id"], ascending=[False, True])

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(STATS_PATH, index=False)
    with open(LEXICON_PATH, encoding="utf-8") as fh:
        lexicon_version = yaml.safe_load(fh).get("version")
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes_at_build": bool(_git("status", "--porcelain")),
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()[:16],
        "lexicon_version": lexicon_version,
        "lexicon_sha256": hashlib.sha256(LEXICON_PATH.read_bytes()).hexdigest()[:16],
        "short_text_words": SHORT_TEXT_WORDS,
        "files": int(len(out)),
        "in_analysis_set": int(out["in_analysis_set"].sum()),
    }
    META_PATH.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log.info(f"Wrote {STATS_PATH.relative_to(ROOT)} ({len(out)} rows, {int(out['in_analysis_set'].sum())} in analysis set)")
    log.info("Exclusions: " + str(out["exclusion_step"].replace("", "in_analysis_set").value_counts().to_dict()))
    log.close()
    return out


def spotcheck(n: int, seed: int, dimension: str | None) -> None:
    """Print n random matches with context, half per corpus. Nothing is written to disk."""
    stats = load_book_stats()
    column = f"pov_{dimension}_n" if dimension else "pov_n"
    rng = random.Random(seed)
    for role in ROLES:
        pool = stats[stats["in_analysis_set"] & (stats["corpus_role"] == role) & (stats[column] > 0)]
        total = int(pool[column].sum())
        picks = sorted(rng.sample(range(total), min(n // 2, total)))
        wanted: dict[str, list[int]] = {}
        offset = 0
        for book_id, k in zip(pool["book_id"], pool[column].astype(int)):
            local = [p - offset for p in picks if offset <= p < offset + k]
            if local:
                wanted[book_id] = local
            offset += k
        print(f"\n===== {role}: {len(picks)} of {total:,} matches ({dimension or 'all dimensions'}) =====")
        for book_id, local in wanted.items():
            i = 0
            for text, _ in _body_chapters(book_id):
                _, merged = find_matches(text)
                if dimension:
                    merged = [m for m in merged if dimension in m[2]]
                for start, end, names in merged:
                    if i in local:
                        left, right = text[max(0, start - 130):start], text[end:end + 130]
                        print(f"[{book_id} | {'+'.join(sorted(names))}] ...{left}>>>{text[start:end]}<<<{right}...")
                    i += 1


# --------------------------------------------------------------------------- helpers for the notebook

def load_book_stats() -> pd.DataFrame:
    """Read the book-level table with the right column types."""
    stats = pd.read_csv(STATS_PATH, dtype={"edition_year": "Int64"}, keep_default_na=True)
    for column in ["work_group", "exclusion_step", "exclusion_reason", "corpus_type", "identity_check", "author"]:
        stats[column] = stats[column].fillna("")
    return stats


def load_meta() -> dict:
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def dimension_columns(stats: pd.DataFrame) -> dict[str, str]:
    """Map dimension name -> count column, in the order of DIMENSION_LABELS."""
    return {d: f"pov_{d}_n" for d in DIMENSION_LABELS if f"pov_{d}_n" in stats.columns}


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion (works for small n and for 0% / 100%)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def flow_table(stats: pd.DataFrame) -> pd.DataFrame:
    """From collected files to the analysis set: one row per exclusion step, by corpus."""
    rows = [("Files collected", {r: int((stats["corpus_role"] == r).sum()) for r in ROLES})]
    for step, label in EXCLUSION_STEPS.items():
        rows.append((f"\u2212 {label}", {r: int(((stats["corpus_role"] == r) & (stats["exclusion_step"] == step)).sum())
                                         for r in ROLES}))
    rows.append(("= Books in the analysis set", {r: int(((stats["corpus_role"] == r) & stats["in_analysis_set"]).sum())
                                                 for r in ROLES}))
    table = pd.DataFrame({ROLE_LABELS[r]: [counts[r] for _, counts in rows] for r in ROLES},
                         index=[label for label, _ in rows])
    table["Total"] = table.sum(axis=1)
    return table


def median_iqr(values: pd.Series) -> tuple[float, float, float]:
    """Median with the first and third quartile (the middle half of the books lies between them)."""
    q1, median, q3 = values.astype(float).quantile([0.25, 0.5, 0.75])
    return median, q1, q3


def _median_text(values: pd.Series) -> str:
    median, q1, q3 = median_iqr(values)
    return f"{median:.1f} ({q1:.1f}\u2013{q3:.1f})"


def _share_text(hits: int, n: int) -> str:
    low, high = wilson_interval(hits, n)
    return f"{hits / n:.0%} ({low:.0%}\u2013{high:.0%})"


def key_figures(stats: pd.DataFrame) -> pd.DataFrame:
    """The numbers one would quote, per corpus. Shares come with their 95% interval in brackets."""
    rows: dict[str, dict[str, str]] = {}

    def add(label, value):
        rows[label] = {ROLE_LABELS[r]: value(stats[stats["corpus_role"] == r],
                                             stats[(stats["corpus_role"] == r) & stats["in_analysis_set"]]) for r in ROLES}

    add("Files collected", lambda files, books: f"{len(files)}")
    add("Folder label checked (placed by hand or taxonomy review)",
        lambda files, books: f"{files['role_source'].isin(CHECKED_ROLE_SOURCES).sum()} of {len(files)}")
    add("Books in the analysis set", lambda files, books: f"{len(books)}")
    add("Words in the analysis set (millions)", lambda files, books: f"{books['n_words'].sum() / 1e6:.1f}")
    time_books = load_time_books(stats)
    add("Books with a clean edition year",
        lambda files, books: f"{time_books.loc[books.index, 'edition_year_clean'].notna().sum()} of {len(books)}")
    add("Books with a clean year of first publication",
        lambda files, books: f"{time_books.loc[books.index, 'work_year_clean'].notna().sum()} of {len(books)}")
    add("Books with at least one POV term", lambda files, books: _share_text(int((books["pov_n"] > 0).sum()), len(books)))
    add("POV terms per 10,000 words: median (middle half)", lambda files, books: _median_text(books["pov_per_10k"]))
    for threshold in ILLUSTRATIVE_THRESHOLDS:
        add(f"Books with {threshold} or more POV terms per 10,000 words",
            lambda files, books, t=threshold: _share_text(int((books["pov_per_10k"] >= t).sum()), len(books)))
    return pd.DataFrame(rows).T


def robustness_primary_split(stats: pd.DataFrame) -> pd.DataFrame:
    """POV numbers for the within-primary cut: fiction manuals vs pedagogy/genre craft vs contrast.

    Pedagogy/genre craft stay in `primary`; this table is a robustness check, not a new corpus.
    """
    books = stats[stats["in_analysis_set"]]
    threshold = ILLUSTRATIVE_THRESHOLDS[-1]
    rows = {}
    order = ["fiction_manual", "pedagogy_genre_craft", "contrast"]
    for key in order:
        part = books[books["primary_group"] == key]
        rows[PRIMARY_GROUP_LABELS[key]] = {
            "books": len(part),
            "median": round(float(part["pov_per_10k"].median()), 1) if len(part) else float("nan"),
            "middle half": (
                f"{part['pov_per_10k'].quantile(0.25):.1f}\u2013{part['pov_per_10k'].quantile(0.75):.1f}"
                if len(part) else ""),
            f"{threshold} or more": f"{(part['pov_per_10k'] >= threshold).mean():.0%}" if len(part) else "",
        }
    return pd.DataFrame(rows).T


def _short_author(author: str) -> str:
    return str(author or "").split(";")[0].strip()


def example_titles(books: pd.DataFrame, group: str, n: int = 6) -> list[str]:
    """A few catalogue titles for one robustness group, spread across the alphabet (no book text)."""
    part = books[books["primary_group"] == group].sort_values("title")
    seen, lines = set(), []
    for row in part.itertuples(index=False):
        title = str(row.title).strip()
        key = title.casefold()
        if len(title) < 8 or key in seen:
            continue
        seen.add(key)
        lines.append(f"{title} — {_short_author(row.author)}")
    if len(lines) <= n:
        return lines
    pick = [round(i * (len(lines) - 1) / (n - 1)) for i in range(n)]
    return [lines[i] for i in pick]


def comparison_by_reason(books: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    """Comparison books grouped by the recorded reason, with a few example titles."""
    part = books[books["corpus_role"] == "comparison"]
    rows = []
    for reason, subset in part.groupby(part["exclusion_reason"].replace("", "other")):
        examples = []
        seen = set()
        ordered = subset.sort_values("title")
        candidates = []
        for row in ordered.itertuples(index=False):
            title = str(row.title).strip()
            if title.casefold() in seen or len(title) < 4:
                continue
            seen.add(title.casefold())
            candidates.append(f"{title} — {_short_author(row.author)}")
        if len(candidates) <= n:
            examples = candidates
        else:
            pick = [round(i * (len(candidates) - 1) / (n - 1)) for i in range(n)]
            examples = [candidates[i] for i in pick]
        rows.append({
            "Kind of book": EXCLUSION_REASON_LABELS.get(reason, reason or "Other"),
            "Books": len(subset),
            "Examples": "; ".join(examples),
        })
    return pd.DataFrame(rows).sort_values("Books", ascending=False).reset_index(drop=True)


def lexicon_table(books: pd.DataFrame) -> pd.DataFrame:
    """What the POV word list counts, and how often fiction manuals use each aspect."""
    with open(LEXICON_PATH, encoding="utf-8") as fh:
        lexicon = yaml.safe_load(fh)
    manuals = books[books["primary_group"] == "fiction_manual"]
    rows = []
    for dimension, block in lexicon["dimensions"].items():
        examples = ", ".join(block["en"]["high"][:2])
        column = f"pov_{dimension}_n"
        share = (manuals[column] > 0).mean() if len(manuals) and column in manuals else float("nan")
        rows.append({
            "Aspect": DIMENSION_LABELS.get(dimension, dimension),
            "Example terms": examples,
            "Fiction manuals with at least one": f"{share:.0%}" if len(manuals) else "",
            "In plain words": ASPECT_IN_WORDS.get(dimension, ""),
            "_key": dimension,
        })
    table = pd.DataFrame(rows)
    columns = [f"pov_{d}_n" for d in table["_key"] if f"pov_{d}_n" in manuals.columns]
    ranked = manuals[columns].gt(0).mean().sort_values(ascending=False).index
    order = [c.removeprefix("pov_").removesuffix("_n") for c in ranked]
    return table.set_index("_key").loc[order].reset_index(drop=True)


def sensitivity_table(stats: pd.DataFrame) -> pd.DataFrame:
    """The headline POV numbers under other choices of which books to include."""
    books = stats[stats["in_analysis_set"]]
    variants = {
        "Analysis set (used in all figures)": books,
        "Fiction manuals only vs contrast (robustness cut)": books[
            books["primary_group"].isin(["fiction_manual", "contrast"])],
        "All books with text, nothing excluded": stats[~stats["needs_ocr"]],
        "Analysis set without files whose identity is in doubt": books[books["identity_check"] == ""],
        "Analysis set, e-books only": books[books["format"] != "pdf"],
        "Analysis set, PDFs only": books[books["format"] == "pdf"],
    }
    threshold = ILLUSTRATIVE_THRESHOLDS[-1]
    rows = {}
    for name, subset in variants.items():
        row = {}
        for role in ROLES:
            part = subset[subset["corpus_role"] == role]
            median, q1, q3 = median_iqr(part["pov_per_10k"])
            row[(ROLE_LABELS[role], "books")] = len(part)
            row[(ROLE_LABELS[role], "median")] = round(median, 1)
            row[(ROLE_LABELS[role], "middle half")] = f"{q1:.1f}\u2013{q3:.1f}"
            row[(ROLE_LABELS[role], f"{threshold} or more")] = f"{(part['pov_per_10k'] >= threshold).mean():.0%}"
        rows[name] = row
    table = pd.DataFrame(rows).T
    for role in ROLES:  # mixed columns come back as text; restore the numbers
        table[(ROLE_LABELS[role], "books")] = table[(ROLE_LABELS[role], "books")].astype(int)
        table[(ROLE_LABELS[role], "median")] = table[(ROLE_LABELS[role], "median")].astype(float)
    return table


def load_time_books(stats: pd.DataFrame) -> pd.DataFrame:
    """The analysis set with clean years only (same index as `stats`).

    - `edition_year_clean`: the year of the book's own candidate-list record. Not used for renamed
      files, open identity doubts, or years borrowed from another file of the same work.
    - `work_year_clean`: year of first publication from the biblio table, only from the seed
      bibliography or from Open Library joined by work id, and only without an identity doubt.
    - `edition_lag`: edition year minus work year, where both are known.
    """
    biblio = pd.read_csv(BIBLIO_PATH, usecols=["book_id", "first_work_year", "first_work_year_source"],
                         dtype={"first_work_year": "Int64"})
    books = stats[stats["in_analysis_set"]].copy()
    work = books["book_id"].map(biblio.set_index("book_id")["first_work_year"])
    source = books["book_id"].map(biblio.set_index("book_id")["first_work_year_source"])
    trusted = source.isin(CLEAN_WORK_YEAR_SOURCES) & books["identity_check"].eq("")
    books["edition_year_clean"] = books["edition_year"].where(books["year_source"].eq("candidate_list")).astype("Int64")
    books["work_year_clean"] = work.where(trusted).astype("Int64")
    books["work_year_source"] = source.where(trusted).fillna("")
    books["edition_lag"] = (books["edition_year_clean"] - books["work_year_clean"]).astype("Int64")
    return books


def time_coverage_table(books: pd.DataFrame) -> pd.DataFrame:
    """How many books of the analysis set carry a clean edition year, work year, or both."""
    rows = {}
    for role in ROLES:
        part = books[books["corpus_role"] == role]
        both = part["edition_year_clean"].notna() & part["work_year_clean"].notna()
        lag = part.loc[both, "edition_lag"]
        rows[ROLE_LABELS[role]] = {
            "Books in the analysis set": len(part),
            "Clean edition year": int(part["edition_year_clean"].notna().sum()),
            "Clean year of first publication": int(part["work_year_clean"].notna().sum()),
            "\u2026 from the seed bibliography": int(part["work_year_source"].eq("seed").sum()),
            "\u2026 from Open Library (joined by work id)": int(part["work_year_source"].eq("ol_search").sum()),
            "Both years": int(both.sum()),
            "Of these, edition later than first publication": int((lag > 0).sum()),
        }
    table = pd.DataFrame(rows)
    table["Total"] = table.sum(axis=1)
    lag = books["edition_lag"].dropna()
    gaps = {ROLE_LABELS[r]: books.loc[(books["corpus_role"] == r) & (books["edition_lag"] > 0), "edition_lag"]
            for r in ROLES} | {"Total": lag[lag > 0]}
    table = table.astype(object)
    table.loc["Median gap in years, where later"] = {k: f"{v.median():.0f}" if len(v) else "\u2013" for k, v in gaps.items()}
    return table


def same_work_pairs(stats: pd.DataFrame) -> pd.DataFrame:
    """Works we hold twice, once as PDF and once as e-book: do the two files give the same count?"""
    usable = stats[stats["possible_duplicate"] & ~stats["needs_ocr"] & ~stats["short_text"]]
    rows = []
    for _, group in usable.groupby("work_group"):
        pdf, ebook = group[group["format"] == "pdf"], group[group["format"] != "pdf"]
        if len(pdf) == 1 and len(ebook) == 1:
            pdf, ebook = pdf.iloc[0], ebook.iloc[0]
            rows.append({
                "Work": f"{pdf['title'].split(':')[0]} ({pdf['author'].split(';')[0]})",
                "Words, PDF": int(pdf["n_words"]), "Words, e-book": int(ebook["n_words"]),
                "Length difference": abs(pdf["n_words"] - ebook["n_words"]) / max(pdf["n_words"], ebook["n_words"]),
                "POV per 10,000 words, PDF": pdf["pov_per_10k"], "POV per 10,000 words, e-book": ebook["pov_per_10k"],
            })
    return pd.DataFrame(rows).sort_values("Length difference").reset_index(drop=True)


def seed_placement() -> tuple[pd.Series, pd.DataFrame]:
    """Where the English seed titles (the benchmark of clearly relevant books) ended up.

    Returns (number of seed titles per folder, the seed titles that are not in `primary`).
    """
    from src.fulltext.books import display_title_author, load_metadata_overrides

    manifest = pd.read_csv(MANIFEST_PATH, low_memory=False).fillna("")
    seeds = manifest[(manifest["lang"] == LANG) & manifest["candidate_id"].str.startswith("seed:")]
    elsewhere = seeds[seeds["corpus_role"] != "primary"]
    overrides = load_metadata_overrides()
    wrong = wrong_book_ids()
    table = pd.DataFrame({
        "Seed": elsewhere["candidate_id"].str.replace("seed:", ""),
        "Seed title": elsewhere["title"], "Seed author": elsewhere["author_name"],
        "The file is": [display_title_author(row, overrides)[0] if row["book_id"] in wrong else "the seed title"
                        for row in elsewhere.to_dict(orient="records")],
        "Folder now": elsewhere["corpus_role"].replace("unreviewed", "review (not yet labelled)"),
        "Label from": elsewhere["role_source"].map(ROLE_SOURCE_LABELS),
    }).sort_values("Seed").reset_index(drop=True)
    return seeds.groupby("corpus_role")["candidate_id"].nunique(), table


def wrong_book_ids() -> set[str]:
    """Files that turned out to be another book than their catalogue record (`confidence` = wrong_book)."""
    overrides = pd.read_csv(METADATA_OVERRIDES_PATH, dtype=str).fillna("")
    return set(overrides.loc[overrides["confidence"] == "wrong_book", "book_id"])


def primary_check() -> pd.DataFrame:
    """English primary books whose label conflicts with the prescreen rules (written by prescreen_books)."""
    check = pd.read_csv(PRIMARY_CHECK_PATH, dtype=str).fillna("")
    return check[check["lang"] == LANG].reset_index(drop=True)


def files_by_language() -> pd.DataFrame:
    """All collected files by language and folder label (for context; only English is analysed here)."""
    manifest = pd.read_csv(MANIFEST_PATH, low_memory=False)
    return pd.crosstab(manifest["lang"], manifest["corpus_role"])


def prescreen_pov_words() -> tuple[list[str], float]:
    """POV words that the automatic prescreen counts as a sign of a writing manual.

    The prescreen (config/book_prescreen.yml) proposes `primary` when its 'instruction'
    signal is frequent enough, so these words would help decide the label. Empty since
    2026-11-12. Returns (words, signals per 1,000 words needed for `primary`).
    """
    with open(CONFIG / "book_prescreen.yml", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    words = []
    for pattern in config["signals"]["instruction"][LANG]:
        if re.search(r"point of view|viewpoint|bnarrators\?", pattern):
            words += [w.replace("\\b", "").replace("s?", "") for w in pattern.split("|")]
    return words, float(config["thresholds"]["instruction_primary"][LANG])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--selftest", action="store_true", help="only test the POV matcher")
    parser.add_argument("--spotcheck", type=int, metavar="N", help="print N random matched snippets (not saved)")
    parser.add_argument("--dimension", help="restrict --spotcheck to one dimension, e.g. general_pov")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    if args.selftest:
        problems = selftest()
        print("\n".join(problems) if problems else "POV matcher self-test passed.")
        raise SystemExit(1 if problems else 0)
    if args.spotcheck:
        spotcheck(args.spotcheck, args.seed, args.dimension)
        return
    build(args.workers)


if __name__ == "__main__":
    main()
