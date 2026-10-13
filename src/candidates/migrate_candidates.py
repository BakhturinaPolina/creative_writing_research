"""Migrate colleague OpenLibrary lists and the seed corpus into the master schema.

Steps 1-7 of docs/candidate_lists.md. Raw files are read only.
Writes data/interim/catalogue/openlibrary_candidates.csv, seed_candidates.csv,
seed_recall.csv, prescreen_en.csv and prescreen_de.csv.
"""
import re

import pandas as pd

from src.common.logging_util import setup_logger
from src.common.normalize import normalize_author, normalize_isbn, normalize_title
from src.common.paths import DATA_INTERIM, DATA_RAW
from src.common.prescreen import load_screen_rules, prescreen
from src.common.schema import MASTER_COLUMNS

RAW_OL = DATA_RAW / "colleague_openlibrary"
SEED_PATH = DATA_RAW / "seed_corpus" / "seed_corpus_100__Corpus_100.csv"
CATALOGUE = DATA_INTERIM / "catalogue"
SCRIPT_BY_SOURCE = {
    "colleague_openlibrary_en": "scripts/openlibrary_scrape_en.py",
    "colleague_openlibrary_de": "scripts/openlibrary_scrape_de.py",
}

WORKING_COLUMNS = [
    "url",
    "year_raw",
    "languages_raw",
    "multi_language",
    "title_norm",
    "author_norm",
]
OUTPUT_COLUMNS = MASTER_COLUMNS + WORKING_COLUMNS

YEAR_MIN = 1450
YEAR_MAX = 2027
PRESCREEN_STATUSES = ("likely_primary", "uncertain")
OL_ID_RE = re.compile(r"(OL\d+)([WMwm])")
AUTHOR_SPLIT_RE = re.compile(r"\s*(?:;|/|\band\b|\bund\b)\s*", re.IGNORECASE)
ROLE_RE = re.compile(
    r"^(?:eds?|editors?|hrsg|herausgeber(?:in|innen)?|hg)\.?$",
    re.IGNORECASE,
)
LANG_MAP = {"eng": "en", "en": "en", "ger": "de", "deu": "de", "de": "de"}
SEED_LANG = {"english": "en", "german": "de"}
SEED_TRANSLATION = {"original": "original", "translation": "translation"}

RENAME = {
    "Название": "title",
    "Подзаголовок": "subtitle",
    "Автор": "author",
    "Год": "year",
    "Ссылка": "url",
    "ISBN": "isbn",
    "Издатель": "publisher",
    "Язык": "languages",
    "Языки": "languages",
}

FEED_COLUMNS = [
    "candidate_id",
    "title",
    "author_name",
    "publication_year",
    "isbn_13",
    "isbn_10",
    "edition_language",
    "md5",
    "search_status",
    "found_format",
    "search_date",
]

RECALL_COLUMNS = [
    "candidate_id",
    "edition_language",
    "author",
    "title",
    "match_status",
    "matched_candidate_id",
    "matched_source",
]

SOURCE_FOR_LANG = {
    "en": "colleague_openlibrary_en",
    "de": "colleague_openlibrary_de",
}


def _cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def parse_year(value):
    text = _cell(value)
    if re.fullmatch(r"\d{3,4}", text):
        year = int(text)
        if YEAR_MIN <= year <= YEAR_MAX:
            return str(year)
    return ""


def language_codes(value):
    codes = []
    for part in re.split(r"[,;/|]", _cell(value)):
        code = part.strip().lower()
        if not code:
            continue
        codes.append(LANG_MAP.get(code, code))
    return codes


def split_isbns(value):
    isbn_10, isbn_13 = [], []
    for part in re.split(r"[;,\s]+", _cell(value)):
        isbn = normalize_isbn(part)
        if not isbn:
            continue
        if len(isbn) == 13 and isbn not in isbn_13:
            isbn_13.append(isbn)
        elif len(isbn) == 10 and isbn not in isbn_10:
            isbn_10.append(isbn)
    return ";".join(isbn_10), ";".join(isbn_13)


def extract_ol(url):
    """Return ol_work_id, ol_edition_id, and the URL with /works/OL...M repaired."""
    url = _cell(url)
    match = OL_ID_RE.search(url)
    if not match:
        return "", "", url
    ol_id = f"{match.group(1)}{match.group(2).upper()}"
    if ol_id.endswith("W"):
        return ol_id, "", url
    repaired = url.replace("/works/", "/books/", 1) if "/works/" in url else url
    return "", ol_id, repaired


def first_author(author):
    text = _cell(author)
    if not text:
        return ""
    return AUTHOR_SPLIT_RE.split(text, maxsplit=1)[0].strip()


def author_surname(author):
    """Last token before the first comma, else the last token of the first author."""
    first = first_author(author)
    if not first:
        return ""
    head = first.split(",", 1)[0]
    tokens = normalize_author(head).split()
    return tokens[-1] if tokens else ""


def author_display(author):
    """First author in First-Last order.

    Colleague rows separate coauthors with commas and are already First Last.
    A single comma with a one-word left side is Last, First. An editor suffix
    such as ', ed.' is dropped.
    """
    first = first_author(author)
    if "," not in first:
        return first
    before, after = (part.strip() for part in first.split(",", 1))
    after_first = after.split(",")[0].strip()
    if ROLE_RE.match(after_first):
        return before
    if len(before.split()) == 1 and after_first and not ROLE_RE.match(after_first):
        return f"{after_first} {before}".strip()
    return before


def base_title_raw(title):
    return _cell(title).split(":", 1)[0].strip()


def base_title_norm(title):
    return normalize_title(base_title_raw(title))


def blank_row():
    return {column: "" for column in OUTPUT_COLUMNS}


def _finish_row(row, rules):
    row["title_norm"] = normalize_title(row["title"])
    row["author_norm"] = normalize_author(row["author"])
    status, rule_id = prescreen(row["title"], row["subtitle"], rules)
    row["eligibility_status"] = status
    row["screen_rule"] = rule_id
    return row


def _unique_ids(rows):
    seen = {}
    for row in rows:
        base = row["candidate_id"]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            row["candidate_id"] = f"{base}:{seen[base]}"
    return rows


def read_table(path, **kwargs):
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig", **kwargs)
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame.fillna("").rename(columns=RENAME)


def migrate_openlibrary(rules):
    specs = [
        (
            RAW_OL / "creative writing books scrabing_open library_english.csv",
            {"sep": ";", "skiprows": 1},
            "colleague_openlibrary_en",
            "subject:creative writing",
        ),
        (
            RAW_OL / "kreatives_schreiben_bücher_open library deutsch.csv",
            {"sep": ","},
            "colleague_openlibrary_de",
            "unknown_of_5",
        ),
    ]
    rows = []
    for path, reader_kwargs, source, query in specs:
        frame = read_table(path, **reader_kwargs)
        for number, record in enumerate(frame.to_dict(orient="records"), start=1):
            row = blank_row()
            row["candidate_source"] = source
            row["candidate_source_detail"] = path.name
            row["source_query"] = query
            row["source_script"] = SCRIPT_BY_SOURCE[source]
            row["postprocessing"] = "unknown"
            row["retrieval_date"] = "unknown"
            row["title"] = _cell(record.get("title"))
            row["subtitle"] = _cell(record.get("subtitle"))
            row["author"] = _cell(record.get("author"))
            row["publisher"] = _cell(record.get("publisher"))
            row["year_raw"] = _cell(record.get("year"))
            row["edition_year"] = parse_year(row["year_raw"])
            isbn_10, isbn_13 = split_isbns(record.get("isbn"))
            row["isbn_10"] = isbn_10
            row["isbn_13"] = isbn_13
            work_id, edition_id, url = extract_ol(record.get("url"))
            row["ol_work_id"] = work_id
            row["ol_edition_id"] = edition_id
            row["url"] = url
            ol_id = work_id or edition_id
            row["source_record_id"] = ol_id or f"row:{number}"
            row["candidate_id"] = f"{source}:{row['source_record_id']}"
            row["languages_raw"] = _cell(record.get("languages"))
            codes = language_codes(row["languages_raw"])
            row["edition_language"] = codes[0] if codes else ""
            row["multi_language"] = "yes" if len(codes) > 1 else "no"
            rows.append(_finish_row(row, rules))
    return pd.DataFrame(_unique_ids(rows))[OUTPUT_COLUMNS]


def _seed_lang(value):
    return SEED_LANG.get(_cell(value).lower(), "")


def migrate_seed(rules):
    frame = pd.read_csv(SEED_PATH, dtype=str, encoding="utf-8").fillna("")
    rows = []
    for record in frame.to_dict(orient="records"):
        row = blank_row()
        row["candidate_id"] = _cell(record.get("Corpus_ID"))
        row["source_record_id"] = row["candidate_id"]
        row["candidate_source"] = "seed_corpus"
        row["candidate_source_detail"] = "Corpus_100"
        row["source_script"] = "manual"
        row["title"] = _cell(record.get("Title"))
        row["author"] = _cell(record.get("Author_editor"))
        row["publisher"] = _cell(record.get("Publisher"))
        row["edition_statement"] = _cell(record.get("Edition_note"))
        row["year_raw"] = _cell(record.get("Selected_edition_year"))
        row["edition_year"] = parse_year(row["year_raw"])
        row["first_work_year"] = parse_year(record.get("First_year_corpus_language"))
        row["isbn_13"] = normalize_isbn(_cell(record.get("ISBN_13"))) or ""
        row["isbn_10"] = normalize_isbn(_cell(record.get("ISBN_10"))) or ""
        row["dnb_id"] = _cell(record.get("DNB_ID"))
        row["oclc"] = _cell(record.get("OCLC_ID"))
        row["edition_language"] = _seed_lang(record.get("Corpus_language"))
        row["languages_raw"] = _cell(record.get("Corpus_language"))
        row["multi_language"] = "no"
        row["original_title"] = _cell(record.get("Original_title"))
        row["original_language"] = _seed_lang(record.get("Original_language")) or _cell(
            record.get("Original_language")
        )
        row["translation_status"] = SEED_TRANSLATION.get(
            _cell(record.get("Translation_status")).lower(), ""
        )
        row["translator"] = _cell(record.get("Translator"))
        rows.append(_finish_row(row, rules))
    return pd.DataFrame(rows)[OUTPUT_COLUMNS]


def recall_against_seed(candidates, seed):
    index = {}
    for record in candidates.to_dict(orient="records"):
        key = (record["candidate_source"], base_title_norm(record["title"]), author_surname(record["author"]))
        if not key[1] or not key[2] or key in index:
            continue
        index[key] = record["candidate_id"]

    rows = []
    for record in seed.to_dict(orient="records"):
        source = SOURCE_FOR_LANG.get(record["edition_language"], "")
        surname = author_surname(record["author"])
        key = (source, base_title_norm(record["title"]), surname)
        matched = index.get(key, "") if surname else ""
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "edition_language": record["edition_language"],
                "author": record["author"],
                "title": record["title"],
                "match_status": "hit" if matched else "miss",
                "matched_candidate_id": matched,
                "matched_source": source if matched else "",
            }
        )
    return pd.DataFrame(rows)[RECALL_COLUMNS]


def export_prescreen_lists(candidates):
    """Screened OpenLibrary rows, one file per language.

    A row is included when `en` or `de` is among its language codes. Rows with
    no language code follow the list they came from. Downstream access checks
    apply one language to the whole file, so edition_language here is that language.
    """
    frames = {}
    records = candidates.to_dict(orient="records")
    for lang, source in SOURCE_FOR_LANG.items():
        selected = []
        for record in records:
            if record["eligibility_status"] not in PRESCREEN_STATUSES:
                continue
            codes = language_codes(record["languages_raw"])
            in_language = lang in codes or (not codes and record["candidate_source"] == source)
            if not in_language:
                continue
            selected.append(
                {
                    "candidate_id": record["candidate_id"],
                    "title": base_title_raw(record["title"]),
                    "author_name": author_display(record["author"]),
                    "publication_year": record["edition_year"],
                    "isbn_13": record["isbn_13"],
                    "isbn_10": record["isbn_10"],
                    "edition_language": lang,
                    "md5": "",
                    "search_status": "",
                    "found_format": "",
                    "search_date": "",
                }
            )
        frames[lang] = pd.DataFrame(selected, columns=FEED_COLUMNS)
    return frames["en"], frames["de"]


def _print_screen(frame, label):
    print(f"\n{label}: {len(frame)} rows")
    counts = (
        frame.groupby(["candidate_source", "eligibility_status"], dropna=False)
        .size()
        .rename("rows")
    )
    print(counts.to_string())


def _print_recall(frame):
    print("\nSeed recall against OpenLibrary (base title before colon + surname):")
    for lang in ("en", "de"):
        part = frame[frame["edition_language"] == lang]
        hits = int((part["match_status"] == "hit").sum())
        total = len(part)
        recall = hits / total if total else 0
        print(f"  {lang}: {hits}/{total} = {recall:.3f}")
    misses = frame[frame["match_status"] == "miss"]
    print(f"Misses ({len(misses)}):")
    for record in misses.to_dict(orient="records"):
        print(f"  {record['candidate_id']}  {record['author']} — {record['title']}")


def _check_openlibrary(frame):
    if len(frame) != 6442 + 824:
        raise SystemExit(f"expected 7266 OpenLibrary rows, got {len(frame)}")
    broken = frame["url"].str.contains(r"/works/OL\d+M", regex=True)
    if broken.any():
        raise SystemExit(f"{int(broken.sum())} /works/OL...M URLs were not repaired")
    if frame["candidate_id"].duplicated().any():
        raise SystemExit("candidate_id values are not unique")
    edition_ids = int((frame["ol_edition_id"] != "").sum())
    if edition_ids != 1206 + 5:
        raise SystemExit(f"expected 1211 edition ids, got {edition_ids}")


def main():
    log = setup_logger(__file__)
    try:
        log.info("Loading inclusion rules and migrating Open Library + seed lists")
        rules = load_screen_rules()
        candidates = migrate_openlibrary(rules)
        seed = migrate_seed(rules)
        recall = recall_against_seed(candidates, seed)
        prescreen_en, prescreen_de = export_prescreen_lists(candidates)

        _check_openlibrary(candidates)
        CATALOGUE.mkdir(parents=True, exist_ok=True)
        outputs = {
            CATALOGUE / "openlibrary_candidates.csv": candidates,
            CATALOGUE / "seed_candidates.csv": seed,
            CATALOGUE / "seed_recall.csv": recall,
            CATALOGUE / "prescreen_en.csv": prescreen_en,
            CATALOGUE / "prescreen_de.csv": prescreen_de,
        }
        for path, frame in outputs.items():
            frame.to_csv(path, index=False)
            log.info(f"saved {path} ({len(frame)} rows)")

        _print_screen(candidates, "OpenLibrary pre-screen")
        _print_screen(seed, "Seed pre-screen")
        _print_recall(recall)
        log.info(f"Prescreen lists: en {len(prescreen_en)}, de {len(prescreen_de)}")
        log.info("German recall near 3/50 is the expected sanity check, not a hard threshold.")
    finally:
        log.close()


if __name__ == "__main__":
    main()
