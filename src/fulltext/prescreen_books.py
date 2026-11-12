"""Propose a corpus_role for every book from its title and a body-text sample.

Roles (see docs/scientific_readme.md section 2):
  primary     corpus types A-D (fiction-writing manuals, incl. German translations)
  comparison  types E-F and other related writing books (school, therapy, academic study,
              screenwriting, poetry); dataset C, and hard negatives for classifiers
  negative    fiction, unrelated non-fiction, language courses, blank notebooks,
              books outside EN/DE; easy negatives for classifiers
  unreviewed  not enough evidence (no text yet, weak signals, conflicts with the lists)

Manual decisions live in config/book_role_overrides.csv and always win over the
automatic proposal. Writes prescreen_* and the final role columns back into
manifest.csv and a review table for the books that are not in the candidate lists.

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
    front_text,
    read_sample,
    word_count,
)
from src.fulltext.inventory import MANIFEST_COLS

CONFIG_PATH = CONFIG / "book_prescreen.yml"
OVERRIDES_PATH = CONFIG / "book_role_overrides.csv"
REVIEW_PATH = FULLTEXT_DIR / "review_unlisted.csv"
REVIEW_EXCERPTS_PATH = EXCERPTS_DIR / "review_excerpts.csv"

OVERRIDE_COLS = ["book_id", "corpus_role", "exclusion_reason", "corpus_type",
                 "translation_status", "original_language", "reviewer", "note"]
TITLE_EXCLUDE_REASON = {
    "likely_exclude.children_handwriting": "children_handwriting",
    "likely_exclude.technical_academic_business": "academic_writing",
    "likely_contextual.school_pedagogy": "school_pedagogy",
    "likely_contextual.therapy_coaching": "therapy",
}
SIGNALS = ["instruction", "fiction", "dialogue", "academic", "literary", "school", "therapy"]


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


def decide(row: dict, r: dict, titles, title_rules, t) -> tuple[str, str, str]:
    """Return (role, reason, exclusion_reason)."""
    title = row["title"] or row["file_title"]
    title_status, title_rule = prescreen(title, "", title_rules)
    lang = row["lang"] if row["lang"] in CORPUS_LANGS else "en"
    strong, weak = t["instruction_primary"][lang], t["instruction_weak"][lang]
    instr, fict, dial = r["rate_instruction"], r["rate_fiction"], r["rate_dialogue"]
    curated = row["download_priority"] in {"P0", "P1"}

    if row["lang"] == "other":
        return "negative", "language not en/de", "out_of_language"
    if titles["language_course"].search(title):
        return "negative", "title: language course", "language_course"
    if titles["blank_notebook"].search(title):
        return "negative", "title: blank notebook", "not_instructional"
    if int(row["sample_words"] or 0) < t["min_sample_words"]:
        needs = "needs OCR" if row["format"] == "pdf" else "no text extracted"
        return "unreviewed", f"no text sample ({needs})", ""
    if title_status == "likely_exclude":
        return "negative", f"title rule {title_rule}", TITLE_EXCLUDE_REASON.get(title_rule, "other")
    if (fict >= t["fiction_tags_high"] or dial >= t["dialogue_high"]) and instr < weak:
        reason = f"fiction (dialogue tags {fict}/1k, quotes {dial}/1k, instruction {instr}/1k)"
        if curated:
            return "unreviewed", f"in curated list {row['download_priority']} but reads like {reason}", ""
        return "negative", reason, "not_instructional"

    if instr >= strong:
        school = r["rate_school"]
        if school >= t["school_high"] and school >= instr / 2:
            return "comparison", f"instructional but school-oriented ({school}/1k)", "school_pedagogy"
        if r["rate_therapy"] >= t["therapy_high"]:
            return "comparison", f"instructional but therapy-oriented ({r['rate_therapy']}/1k)", "therapy"
        if titles["screenwriting"].search(title):
            return "comparison", "title: screenwriting", "screenwriting_only"
        if titles["poetry"].search(title):
            return "comparison", "title: poetry", "poetry_only"
        if titles.get("playwriting") and titles["playwriting"].search(title):
            return "comparison", "title: playwriting", "playwriting_only"
        if titles.get("creative_nonfiction") and titles["creative_nonfiction"].search(title):
            return "comparison", "title: creative nonfiction / memoir craft", "creative_nonfiction"
        if title_status == "likely_contextual":
            return "comparison", f"title rule {title_rule}", TITLE_EXCLUDE_REASON.get(title_rule, "other")
        if row["download_priority"] == "P0" or row["screen_status"] == "likely_primary":
            return "primary", f"instruction {instr}/1k + candidate list", ""
        return "primary", f"instruction {instr}/1k", ""

    if titles.get("literary_scholarship") and titles["literary_scholarship"].search(title):
        return "negative", "title: literary scholarship (not narratology)", "not_instructional"
    if r["rate_academic"] >= t["academic_high"]:
        if r["rate_literary"] >= t["literary_some"]:
            # Narratology / writing theory stays comparison; other lit-scholarship is negative.
            # Title cues for narratology are weak here, so academic+literary still → comparison
            # unless the literary-scholarship title rule already fired above.
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


def main() -> None:
    log = setup_logger(__file__)
    signals, titles, translation, english, thresholds = load_config()
    title_rules = load_screen_rules()
    overrides = load_overrides()
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str, encoding="utf-8").fillna("")
    log.info(f"{len(manifest)} books; {len(overrides)} manual overrides")

    rows, excerpts = [], []
    for i, row in enumerate(manifest.to_dict(orient="records"), 1):
        sample = read_sample(row["book_id"])
        rates = signal_rates(sample, row["lang"], signals)
        role, reason, exclusion = decide(row, rates, titles, title_rules, thresholds)
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

        manual = overrides.get(row["book_id"])
        if manual:
            row["corpus_role"] = manual["corpus_role"]
            row["exclusion_reason"] = manual["exclusion_reason"]
            row["corpus_type"] = manual["corpus_type"] or row.get("corpus_type", "")
            row["role_source"] = f"review:{manual['reviewer'] or 'unknown'}"
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

    log.info(f"wrote {MANIFEST_PATH} and {REVIEW_PATH} ({len(unlisted)} unlisted books)")
    log.info("corpus_role by lang (all books):\n" + frame.groupby(["lang", "corpus_role"]).size().to_string())
    log.info("prescreen_role (unlisted books):\n" + unlisted["prescreen_role"].value_counts().to_string())
    log.info("translation_status (de):\n" + frame[frame["lang"] == "de"]["translation_status"].value_counts().to_string())
    log.info(f"title words not found in text: {(frame['title_match'].astype(str) == 'False').sum()}")
    log.info(f"content duplicates (same text, other filename): {(frame['content_dup_of'] != '').sum()}")
    log.close()


if __name__ == "__main__":
    main()
