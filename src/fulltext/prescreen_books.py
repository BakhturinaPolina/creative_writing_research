"""Propose a corpus_role for every book from its title and a body-text sample.

Roles (see docs/scientific_readme.md section 2):
  primary     corpus types A-D (fiction-writing manuals, incl. German translations)
  comparison  types E-F and other related writing books (school, therapy, academic study,
              screenwriting, poetry); dataset C, and hard negatives for classifiers
  negative    fiction, unrelated non-fiction, language courses, blank notebooks,
              books outside EN/DE; easy negatives for classifiers
  unreviewed  not enough evidence (no text yet, weak signals, conflicts with the lists)

Seed titles (candidate_id `seed:`) are primary unless config/book_metadata_overrides.csv
marks the file as another book (`wrong_book`). Title rules use the corrected title.
Manual decisions live in config/book_role_overrides.csv and always win over the
automatic proposal. Writes prescreen_* and the final role columns back into
manifest.csv, a review table for the books that are not in the candidate lists, and
primary_check.csv: primary books whose label conflicts with the rules.

Usage:
  python -m src.fulltext.prescreen_books
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd
import yaml
from unidecode import unidecode

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, ROOT
from src.common.prescreen import load_screen_rules, prescreen
from src.fulltext.books import (
    CORPUS_LANGS,
    EXCERPTS_DIR,
    FULLTEXT_DIR,
    MANIFEST_PATH,
    METADATA_OVERRIDES,
    display_title_author,
    front_text,
    load_metadata_overrides,
    read_sample,
    word_count,
)
from src.fulltext.inventory import MANIFEST_COLS

CONFIG_PATH = CONFIG / "book_prescreen.yml"
OVERRIDES_PATH = CONFIG / "book_role_overrides.csv"
REVIEW_PATH = FULLTEXT_DIR / "review_unlisted.csv"
PRIMARY_CHECK_PATH = FULLTEXT_DIR / "primary_check.csv"
REVIEW_EXCERPTS_PATH = EXCERPTS_DIR / "review_excerpts.csv"

OVERRIDE_COLS = ["book_id", "corpus_role", "exclusion_reason", "corpus_type",
                 "translation_status", "original_language", "reviewer", "note"]
TITLE_EXCLUDE_REASON = {
    "likely_exclude.children_handwriting": "children_handwriting",
    "likely_exclude.technical_academic_business": "academic_writing",
    # School / therapy title cues used to mean comparison; under the Nov 2026 definition
    # they are narrative writing instruction → primary (not used to divert from primary).
    "likely_contextual.school_pedagogy": "school_pedagogy",
    "likely_contextual.therapy_coaching": "therapy",
}
SIGNALS = ["instruction", "fiction", "dialogue", "academic", "literary", "school", "therapy"]
# Comparison is a small contrast group: narratology and academic writing about writing,
# not narrative-craft teaching. Genre craft (screen / poetry / drama / CNF) and school or
# therapy creative writing stay in primary.
COMPARISON_TITLE_RULES = [
    ("narratology", "title: narratology / narrative theory", "narratology_theory"),
    ("composition", "title: composition / general writing handbook", "academic_writing"),
    ("discipline_writing", "title: writing for law, history or science", "academic_writing"),
    ("writing_studies", "title: research on writing", "research_discipline"),
]
# Title rules that send a book to negative (not narrative-craft teaching, not the contrast group).
NEGATIVE_TITLE_RULES = [
    ("journalism", "title: journalism", "journalism"),
    ("songwriting", "title: songwriting", "songwriting_only"),
]


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    signals = {
        name: {lang: [re.compile(p, re.IGNORECASE) for p in pats] for lang, pats in by_lang.items()}
        for name, by_lang in cfg["signals"].items()
    }
    titles = {name: re.compile("|".join(pats), re.IGNORECASE) for name, pats in cfg["title_rules"].items()}
    translation = [re.compile(p, re.IGNORECASE) for p in cfg["translation"]["de"]]
    english = re.compile("|".join(cfg["translation"]["english_markers"]), re.IGNORECASE)
    return signals, titles, translation, english, cfg["thresholds"]


def signal_rates(text: str, lang: str, signals) -> dict:
    words = max(word_count(text), 1)
    rates = {}
    for name in SIGNALS:
        patterns = signals[name].get(lang) or signals[name]["en"]
        hits = sum(len(p.findall(text)) for p in patterns)
        rates[f"rate_{name}"] = round(1000 * hits / words, 2)
    return rates


def comparison_title(title: str, titles) -> tuple[str, str, str] | None:
    """(role, reason, exclusion_reason) if the title marks narratology or academic writing (contrast group)."""
    for name, reason, exclusion in COMPARISON_TITLE_RULES:
        if titles.get(name) and titles[name].search(title):
            return "comparison", reason, exclusion
    return None


def negative_title(title: str, titles) -> tuple[str, str, str] | None:
    for name, reason, exclusion in NEGATIVE_TITLE_RULES:
        if titles.get(name) and titles[name].search(title):
            return "negative", reason, exclusion
    return None


def decide(row: dict, r: dict, titles, title_rules, t, title: str, wrong_book: bool) -> tuple[str, str, str]:
    """Return (role, reason, exclusion_reason). `title`: corrected title of the file.

    Primary = books whose subject is narrative / creative writing (incl. school pedagogy,
    exercises, screen / poetry / drama craft, creative nonfiction, therapy writing).
    Comparison = small contrast group: narratology and academic writing.
    """
    title_status, title_rule = prescreen(title, "", title_rules)
    lang = row["lang"] if row["lang"] in CORPUS_LANGS else "en"
    strong, weak = t["instruction_primary"][lang], t["instruction_weak"][lang]
    instr, fict, dial = r["rate_instruction"], r["rate_fiction"], r["rate_dialogue"]
    curated = row["download_priority"] in {"P0", "P1"} and not wrong_book

    if row["lang"] == "other":
        return "negative", "language not en/de", "out_of_language"
    if titles["language_course"].search(title):
        return "negative", "title: language course", "language_course"
    if titles["blank_notebook"].search(title):
        return "negative", "title: blank notebook", "not_instructional"
    if other := negative_title(title, titles):
        return other
    if str(row["candidate_id"]).startswith("seed:") and not wrong_book:
        return "primary", f"seed title {row['candidate_id'].removeprefix('seed:')}", ""
    if int(row["sample_words"] or 0) < t["min_sample_words"]:
        needs = "needs OCR" if row["format"] == "pdf" else "no text extracted"
        return "unreviewed", f"no text sample ({needs})", ""
    if title_status == "likely_exclude":
        # Academic-writing title cues → comparison; handwriting / children's books → negative.
        if title_rule.startswith("likely_exclude.technical_academic_business"):
            return "comparison", f"title rule {title_rule}", "academic_writing"
        return "negative", f"title rule {title_rule}", TITLE_EXCLUDE_REASON.get(title_rule, "other")
    if other := comparison_title(title, titles):
        return other
    if (fict >= t["fiction_tags_high"] or dial >= t["dialogue_high"]) and instr < weak:
        reason = f"fiction (dialogue tags {fict}/1k, quotes {dial}/1k, instruction {instr}/1k)"
        if curated:
            return "unreviewed", f"in curated list {row['download_priority']} but reads like {reason}", ""
        return "negative", reason, "not_instructional"

    if instr >= strong:
        if row["download_priority"] == "P0" or row["screen_status"] == "likely_primary":
            return "primary", f"instruction {instr}/1k + candidate list", ""
        return "primary", f"instruction {instr}/1k", ""

    if titles.get("literary_scholarship") and titles["literary_scholarship"].search(title):
        return "negative", "title: literary scholarship (not narratology)", "not_instructional"
    if r["rate_academic"] >= t["academic_high"]:
        if r["rate_literary"] >= t["literary_some"]:
            return "comparison", f"scholarship on literature/writing (citations {r['rate_academic']}/1k)", "academic_study"
        if curated:
            return "unreviewed", f"in curated list {row['download_priority']} but reads like unrelated scholarship", ""
        return "negative", f"unrelated scholarship (citations {r['rate_academic']}/1k)", "not_instructional"
    if instr >= weak:
        if curated:
            return "primary", f"weak instruction {instr}/1k, curated list {row['download_priority']}", ""
        return "unreviewed", f"weak instruction {instr}/1k", ""
    if curated:
        return "unreviewed", f"in curated list {row['download_priority']} but little instruction ({instr}/1k)", ""
    if r["rate_literary"] >= t["literary_some"]:
        return "unreviewed", f"about literature, not instructional ({r['rate_literary']}/1k)", ""
    return "negative", f"no writing or literary signal (instruction {instr}/1k)", "not_instructional"


def content_fingerprint(sample: str) -> str:
    """Hash of the first letters of the body sample; equal hashes = same book under two names."""
    letters = re.sub(r"[^a-zäöüß]", "", sample.lower())[:4000]
    return hashlib.md5(letters.encode("utf-8")).hexdigest() if len(letters) >= 2000 else ""


def title_match_share(title: str, text: str) -> float | str:
    """Share of the title's content words that occur in the book's own text."""
    # unidecode: filenames often lose umlauts ("Doblins", "Erzahlen")
    words = {w for w in re.findall(r"\w{4,}", unidecode(title).lower()) if not w.isdigit()}
    if not words or not text:
        return ""
    lowered = unidecode(text).lower()
    return round(sum(1 for w in words if w in lowered) / len(words), 2)


def translation_info(row: dict, text: str, translation, english) -> tuple[str, str]:
    if row["lang"] != "de" or not text:
        return "", ""
    for pattern in translation:
        match = pattern.search(text)
        if match:
            window = text[max(match.start() - 200, 0): match.end() + 300]
            return "translation", "en" if english.search(window) else ""
    return "original", "de"


def load_overrides() -> dict[str, dict]:
    if not OVERRIDES_PATH.is_file():
        return {}
    frame = pd.read_csv(OVERRIDES_PATH, dtype=str, encoding="utf-8").fillna("")
    return {row["book_id"]: row for row in frame.to_dict(orient="records")}


def load_wrong_books() -> set[str]:
    """book_ids whose file turned out to be another book than its catalogue record."""
    if not METADATA_OVERRIDES.is_file():
        return set()
    frame = pd.read_csv(METADATA_OVERRIDES, dtype=str, encoding="utf-8").fillna("")
    return set(frame.loc[frame["confidence"] == "wrong_book", "book_id"])


def role_source(reviewer: str) -> str:
    # sync_raw_layout records folder moves by hand with reviewer "manual".
    return "manual:folder" if reviewer == "manual" else f"review:{reviewer or 'unknown'}"


def primary_check(frame: pd.DataFrame, titles, title_rules, wrong_books: set[str]) -> pd.DataFrame:
    """Primary books whose label conflicts with the rules: one row per book, reasons joined."""
    rows = []
    for row in frame[frame["corpus_role"] == "primary"].to_dict(orient="records"):
        title = row["prescreen_title"]
        status, rule = prescreen(title, "", title_rules)
        reasons = []
        if row["prescreen_role"] != "primary":
            reasons.append(f"prescreen says {row['prescreen_role']}: {row['prescreen_reason']}")
        if status == "likely_exclude":
            title_reason = f"title rule {rule}"
        else:
            title_reason = (comparison_title(title, titles) or negative_title(title, titles) or (None, ""))[1]
        if title_reason and title_reason not in row["prescreen_reason"]:
            reasons.append(title_reason)
        if row["book_id"] in wrong_books and row["role_source"] != "prescreen":
            reasons.append("placed by hand under its old name; file is another book")
        if reasons:
            rows.append({"book_id": row["book_id"], "lang": row["lang"], "title": title,
                         "author": row["prescreen_author"], "role_source": row["role_source"],
                         "rate_instruction": row["rate_instruction"], "rate_school": row["rate_school"],
                         "rate_academic": row["rate_academic"], "check_reason": "; ".join(reasons)})
    return pd.DataFrame(rows, columns=["book_id", "lang", "title", "author", "role_source", "rate_instruction",
                                       "rate_school", "rate_academic", "check_reason"])


def main() -> None:
    log = setup_logger(__file__)
    signals, titles, translation, english, thresholds = load_config()
    title_rules = load_screen_rules()
    overrides = load_overrides()
    metadata = load_metadata_overrides()
    wrong_books = load_wrong_books()
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, encoding="utf-8").fillna("")
    roles_before = manifest.set_index("book_id")["corpus_role"]
    log.info(f"{len(manifest)} books; {len(overrides)} manual overrides; {len(wrong_books)} files are another book")

    rows, excerpts = [], []
    for i, row in enumerate(manifest.to_dict(orient="records"), 1):
        sample = read_sample(row["book_id"])
        rates = signal_rates(sample, row["lang"], signals)
        title, author = display_title_author(row, metadata)
        wrong_book = row["book_id"] in wrong_books
        role, reason, exclusion = decide(row, rates, titles, title_rules, thresholds, title, wrong_book)
        try:
            front = front_text(ROOT / row["raw_path"], row["format"])
        except Exception as exc:
            log.error(f"front matter unreadable: {row['book_id']}: {exc}")
            front = ""
        translation_status, original_language = translation_info(row, front, translation, english)
        share = title_match_share(row["file_title"], front + " " + sample[:20000])
        row["title_match"] = "" if share == "" else share >= thresholds["title_match_min"]
        if row["title_match"] is False and role != "unreviewed":
            reason += f"; title words not found in text ({share})"
        if translation_status == "translation" and role == "primary":
            reason += "; German translation"
        row.update(rates)
        row["content_fingerprint"] = content_fingerprint(sample)
        row["prescreen_role"], row["prescreen_reason"] = role, reason
        row["prescreen_title"], row["prescreen_author"] = title, author

        manual = overrides.get(row["book_id"])
        if manual:
            row["corpus_role"] = manual["corpus_role"]
            row["exclusion_reason"] = manual["exclusion_reason"]
            row["corpus_type"] = manual["corpus_type"] or row.get("corpus_type", "")
            row["role_source"] = role_source(manual["reviewer"])
        else:
            row["corpus_role"], row["exclusion_reason"] = role, exclusion
            row["role_source"] = "prescreen"
        row["translation_status"] = (manual or {}).get("translation_status") or translation_status
        row["original_language"] = (manual or {}).get("original_language") or original_language
        rows.append(row)
        mid = len(sample) // 2
        excerpts.append({"book_id": row["book_id"], "excerpt": sample[mid:mid + 300]})
        if i % 100 == 0:
            log.info(f"  {i}/{len(manifest)}")

    frame = pd.DataFrame(rows)
    # Same content under several filenames: point every copy at the first book_id.
    fp = frame["content_fingerprint"]
    first = frame[fp != ""].groupby("content_fingerprint")["book_id"].transform("min")
    frame["content_dup_of"] = ""
    frame.loc[first.index, "content_dup_of"] = first.where(first != frame.loc[first.index, "book_id"], "")
    frame.reindex(columns=MANIFEST_COLS).fillna("").to_csv(MANIFEST_PATH, index=False, encoding="utf-8")

    review_cols = ["book_id", "lang", "format", "file_title", "file_author", "sample_words", "title_match",
                   "content_dup_of",
                   "prescreen_role", "prescreen_reason", "corpus_role", "role_source", "exclusion_reason",
                   "translation_status", *[f"rate_{s}" for s in SIGNALS]]
    unlisted = frame[frame["in_candidate_lists"].astype(str) != "True"]
    unlisted = unlisted.sort_values(["lang", "prescreen_role", "file_title"])
    unlisted[review_cols].to_csv(REVIEW_PATH, index=False, encoding="utf-8")
    EXCERPTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(excerpts).to_csv(REVIEW_EXCERPTS_PATH, index=False, encoding="utf-8")
    check = primary_check(frame, titles, title_rules, wrong_books)
    check.sort_values(["lang", "role_source", "title"]).to_csv(PRIMARY_CHECK_PATH, index=False, encoding="utf-8")

    log.info(f"wrote {MANIFEST_PATH} and {REVIEW_PATH} ({len(unlisted)} unlisted books)")
    log.info(f"wrote {PRIMARY_CHECK_PATH}: {len(check)} primary books to check")
    before = frame["book_id"].map(roles_before).fillna("(new)")
    changed = frame[before != frame["corpus_role"]]
    log.info(f"corpus_role changed for {len(changed)} books (rows: before, columns: after):\n"
             + pd.crosstab([changed["lang"], before[changed.index]], changed["corpus_role"]).to_string())
    rate = frame["rate_instruction"].astype(float)
    for lang in sorted(CORPUS_LANGS):
        in_lang = frame["lang"] == lang
        curated = in_lang & frame["download_priority"].isin(["P0", "P1"])
        seeds = in_lang & frame["candidate_id"].str.startswith("seed:") & ~frame["book_id"].isin(wrong_books)
        log.info(f"instruction rate per 1k words ({lang}), median: curated {rate[curated].median():.1f}, "
                 f"seed titles {rate[seeds & (rate > 0)].median():.1f}; thresholds "
                 f"{thresholds['instruction_primary'][lang]} / {thresholds['instruction_weak'][lang]}")
    log.info("corpus_role by lang (all books):\n" + frame.groupby(["lang", "corpus_role"]).size().to_string())
    log.info("prescreen_role (unlisted books):\n" + unlisted["prescreen_role"].value_counts().to_string())
    log.info("translation_status (de):\n" + frame[frame["lang"] == "de"]["translation_status"].value_counts().to_string())
    log.info(f"title words not found in text: {(frame['title_match'].astype(str) == 'False').sum()}")
    log.info(f"content duplicates (same text, other filename): {(frame['content_dup_of'] != '').sum()}")
    log.close()


if __name__ == "__main__":
    main()
