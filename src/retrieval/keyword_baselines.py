"""Baselines A (v0.1 keyword) and B (v0.2 PsyPer lexical) — streaming.

Writes complete qualifying literal-match tables (no top-k cap).
Full passage text only in gitignored sidecars when ``--with-passages``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_INTERIM, DATA_PROCESSED, ROOT
from src.retrieval.evidence import (
    collect_concept_evidence,
    collect_v01_evidence,
    candidate_status,
)
from src.retrieval.lexicon_v02 import load_lexicon
from src.retrieval.paragraphs import CorpusScanStats, iter_body_paragraphs
from src.retrieval.regex_map import load_regex_map
from src.retrieval.run_meta import base_run_meta, write_json

OUT_PROCESSED = DATA_PROCESSED
OUT_INTERIM = DATA_INTERIM / "retrieval"
V01_PATH = CONFIG / "pov_lexicon.yml"

POINTER_FIELDS = [
    "passage_id",
    "book_id",
    "language",
    "title",
    "author",
    "edition_year",
    "corpus_role",
    "primary_group",
    "chapter_idx",
    "para_idx",
    "concept_id",
    "psyper_tags",
    "retrieval_method",
    "bm25_score",
    "rank",
    "matched_terms",
    "matched_regex",
    "matched_spans",
    "evidence_tiers",
    "candidate_status",
    "selection_rule",
]


def _load_v01() -> dict:
    raw = yaml.safe_load(V01_PATH.read_text(encoding="utf-8"))
    return raw.get("dimensions") or {}


def run_baseline_a(lang: str, log, *, with_passages: bool = False) -> Path:
    from src.retrieval.textnorm import normalize_text

    dims = _load_v01()
    scan = CorpusScanStats()
    out = OUT_PROCESSED / f"pov_baseline_a_v01_{lang}.csv"
    side = OUT_INTERIM / f"pov_baseline_a_v01_{lang}_passages.csv"
    OUT_INTERIM.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    passage_ids: set[str] = set()

    import re as _re

    # Prefilter: high/medium multi-word (or long) needles + compiled regexes.
    # Skip short context singles (perspective, voice, …) — they match nearly everything.
    needles: list[str] = []
    compiled_rx: list[_re.Pattern] = []
    for block in dims.values():
        lang_block = (block or {}).get(lang) or {}
        for key in ("high", "medium"):
            for term in lang_block.get(key) or []:
                n = normalize_text(term).replace("-", " ")
                if not n:
                    continue
                if " " in n or "-" in term or len(n) >= 10:
                    needles.append(n)
        for pat in (block.get("regex") or {}).get(lang) or []:
            compiled_rx.append(_re.compile(pat, _re.IGNORECASE))

    with out.open("w", encoding="utf-8", newline="") as fh, (
        side.open("w", encoding="utf-8", newline="") if with_passages else open("/dev/null", "w")
    ) as sh:
        w = csv.DictWriter(fh, fieldnames=POINTER_FIELDS, extrasaction="ignore")
        w.writeheader()
        sw = None
        if with_passages:
            sw = csv.DictWriter(
                sh,
                fieldnames=POINTER_FIELDS + ["passage", "prev_passage", "next_passage"],
                extrasaction="ignore",
            )
            sw.writeheader()

        for i, para in enumerate(
            iter_body_paragraphs(lang, with_neighbors=with_passages, scan_stats=scan)
        ):
            hay = normalize_text(para.text).replace("-", " ")
            if not any(n in hay for n in needles) and not any(
                rx.search(para.text) for rx in compiled_rx
            ):
                continue
            by_dim = collect_v01_evidence(para.text, lang, dims)
            for dim, ev in by_dim.items():
                if not ev.qualifies():
                    continue
                row = {
                    "passage_id": para.passage_id,
                    "book_id": para.book_id,
                    "language": lang,
                    "title": para.title,
                    "author": para.author,
                    "edition_year": para.edition_year,
                    "corpus_role": para.corpus_role,
                    "primary_group": para.primary_group,
                    "chapter_idx": para.chapter_idx,
                    "para_idx": para.para_idx,
                    "concept_id": dim,
                    "psyper_tags": "",
                    "retrieval_method": "baseline_a_v01_keyword",
                    "bm25_score": "",
                    "rank": "",
                    "matched_terms": "; ".join(ev.high + ev.medium),
                    "matched_regex": "; ".join(ev.regex),
                    "matched_spans": "",
                    "evidence_tiers": json.dumps(
                        {
                            "high": len(ev.high),
                            "medium": len(ev.medium),
                            "context": len(ev.context),
                            "regex": len(ev.regex),
                        }
                    ),
                    "candidate_status": "literal_qualified",
                    "selection_rule": "v01_high_or_regex_or_2_supporting",
                }
                w.writerow(row)
                n_rows += 1
                passage_ids.add(para.passage_id)
                if sw is not None:
                    r = dict(row)
                    r["passage"] = para.text
                    r["prev_passage"] = para.prev_text
                    r["next_passage"] = para.next_text
                    sw.writerow(r)
            if i and i % 50000 == 0:
                log.info(f"  Baseline A {lang}: {i} paragraphs, {n_rows} rows so far")

    meta = base_run_meta(
        baseline="A",
        lang=lang,
        lexicon="v0.1",
        lexicon_path=str(V01_PATH.relative_to(ROOT)),
        n_candidate_rows=n_rows,
        n_unique_passages=len(passage_ids),
        expected_books=scan.expected_books,
        processed_books=scan.processed_books,
        skipped_missing=scan.skipped_missing,
        skipped_unreadable=scan.skipped_unreadable,
        n_paragraphs=scan.n_paragraphs,
        stores_book_text=False,
        passage_sidecar=with_passages,
    )
    write_json(OUT_PROCESSED / f"pov_baseline_a_v01_{lang}.meta.json", meta)
    log.info(
        f"Baseline A {lang}: {n_rows} rows, {len(passage_ids)} passages, "
        f"books {scan.processed_books}/{scan.expected_books}"
    )
    if scan.skipped_missing:
        log.warning(f"  missing JSON: {len(scan.skipped_missing)}")
    if scan.skipped_unreadable:
        log.warning(f"  unreadable JSON: {scan.skipped_unreadable}")
    return out


def run_baseline_b(lang: str, log, *, with_passages: bool = False) -> Path:
    from src.retrieval.textnorm import normalize_text

    lex = load_lexicon()
    rmap = load_regex_map()
    scan = CorpusScanStats()
    out = OUT_PROCESSED / f"pov_baseline_b_v02_lexical_{lang}.csv"
    side = OUT_INTERIM / f"pov_baseline_b_v02_lexical_{lang}_passages.csv"
    OUT_INTERIM.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    passage_ids: set[str] = set()
    concepts_hit: set[str] = set()

    # Prefilter needles → concepts (avoid full evidence on most paragraphs).
    needles: list[tuple[str, str]] = []
    for cid, ct in lex.concepts[lang].items():
        for term in list(ct.academic) + list(ct.craft) + list(ct.paraphrases):
            n = normalize_text(term).replace("-", " ")
            if len(n) >= 3:
                needles.append((n, cid))
    # Invert: regex name → concept ids that may use it
    regex_concepts: dict[str, list[str]] = {}
    for cid in lex.concepts[lang]:
        for name in rmap.names_for_concept(lang, cid):
            regex_concepts.setdefault(name, []).append(cid)

    with out.open("w", encoding="utf-8", newline="") as fh, (
        side.open("w", encoding="utf-8", newline="") if with_passages else open("/dev/null", "w")
    ) as sh:
        w = csv.DictWriter(fh, fieldnames=POINTER_FIELDS, extrasaction="ignore")
        w.writeheader()
        sw = None
        if with_passages:
            sw = csv.DictWriter(
                sh,
                fieldnames=POINTER_FIELDS + ["passage", "prev_passage", "next_passage"],
                extrasaction="ignore",
            )
            sw.writeheader()

        for i, para in enumerate(
            iter_body_paragraphs(lang, with_neighbors=with_passages, scan_stats=scan)
        ):
            hay = normalize_text(para.text).replace("-", " ")
            candidates: set[str] = set()
            for needle, cid in needles:
                if needle in hay:
                    candidates.add(cid)
            for name, pat in lex.example_regex.get(lang, {}).items():
                if name not in regex_concepts:
                    continue
                if pat.search(para.text):
                    candidates.update(regex_concepts[name])
            for cid in candidates:
                ct = lex.concepts[lang][cid]
                ev = collect_concept_evidence(para.text, cid, ct, lang, lex, rmap)
                ok, rule = candidate_status(ev, lang)
                if not ok:
                    continue
                row = {
                    "passage_id": para.passage_id,
                    "book_id": para.book_id,
                    "language": lang,
                    "title": para.title,
                    "author": para.author,
                    "edition_year": para.edition_year,
                    "corpus_role": para.corpus_role,
                    "primary_group": para.primary_group,
                    "chapter_idx": para.chapter_idx,
                    "para_idx": para.para_idx,
                    "concept_id": cid,
                    "psyper_tags": cid,
                    "retrieval_method": "baseline_b_v02_lexical",
                    "bm25_score": "",
                    "rank": "",
                    "matched_terms": "; ".join(ev.matched_terms),
                    "matched_regex": "; ".join(ev.matched_regex),
                    "matched_spans": json.dumps(ev.matched_spans, ensure_ascii=False),
                    "evidence_tiers": json.dumps(ev.evidence_tiers_summary()),
                    "candidate_status": "literal_qualified",
                    "selection_rule": rule,
                }
                w.writerow(row)
                n_rows += 1
                passage_ids.add(para.passage_id)
                concepts_hit.add(cid)
                if sw is not None:
                    r = dict(row)
                    r["passage"] = para.text
                    r["prev_passage"] = para.prev_text
                    r["next_passage"] = para.next_text
                    sw.writerow(r)
            if i and i % 50000 == 0:
                log.info(f"  Baseline B {lang}: {i} paragraphs, {n_rows} rows so far")

    meta = base_run_meta(
        baseline="B",
        lang=lang,
        lexicon_version=lex.version,
        lexicon_sha256=lex.sha256,
        regex_map_version=rmap.version,
        regex_map_sha256=__import__("src.retrieval.run_meta", fromlist=["file_sha256"]).file_sha256(
            rmap.path
        ),
        n_candidate_rows=n_rows,
        n_unique_passages=len(passage_ids),
        n_concepts_with_hits=len(concepts_hit),
        expected_books=scan.expected_books,
        processed_books=scan.processed_books,
        skipped_missing=scan.skipped_missing,
        skipped_unreadable=scan.skipped_unreadable,
        n_paragraphs=scan.n_paragraphs,
        stores_book_text=False,
        passage_sidecar=with_passages,
    )
    write_json(OUT_PROCESSED / f"pov_baseline_b_v02_lexical_{lang}.meta.json", meta)
    log.info(
        f"Baseline B {lang}: {n_rows} rows, {len(passage_ids)} passages, "
        f"concepts {len(concepts_hit)}, books {scan.processed_books}/{scan.expected_books}"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", choices=["a", "b", "all"], default="all")
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    parser.add_argument("--with-passages", action="store_true")
    args = parser.parse_args()
    log = setup_logger(__file__)
    langs = ["en", "de"] if args.lang == "all" else [args.lang]
    for lang in langs:
        if args.baseline in ("a", "all"):
            run_baseline_a(lang, log, with_passages=args.with_passages)
        if args.baseline in ("b", "all"):
            run_baseline_b(lang, log, with_passages=args.with_passages)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
