"""Build a book-held-out BM25-only development sample for lexicon expansion.

Writes pointer-only CSVs under data/interim/retrieval/. Passage text is only
written to a gitignored *_passages.csv sidecar when --with-passages is set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED
from src.retrieval.paragraphs import iter_body_paragraphs
from src.retrieval.run_meta import base_run_meta, write_json

OUT = DATA_INTERIM / "retrieval"
SEED = 20261122
PRIORITY_CONCEPTS = [
    "filter_words",
    "voice_participant",
    "voice_heterodiegetic",
    "voice_homodiegetic",
    "voice_eyewitness",
    "focalization_fixed",
    "pov_subject",
    "pov_object",
    "focalization_external",
    "reliability",
]


def build_dev_sample(
    log,
    *,
    seed: int = SEED,
    n_per_lang: int = 40,
    with_passages: bool = False,
) -> Path:
    rng = np.random.default_rng(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    held_books: set[str] = set()

    for lang in ("en", "de"):
        path = DATA_PROCESSED / f"pov_baseline_c_bm25_only_unvalidated_{lang}.csv"
        # Prefer v0.2 exploratory pool for discovery (available before v0.3 run)
        if not path.is_file():
            path = DATA_PROCESSED / f"pov_baseline_c_v03_bm25_only_unvalidated_{lang}.csv"
        if not path.is_file():
            log.warning(f"missing BM25-only file for {lang}")
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "passage_id" not in df.columns:
            df["passage_id"] = (
                df["book_id"]
                + ":c"
                + df["chapter_idx"].astype(str)
                + ":p"
                + df["para_idx"].astype(str)
            )
        # Prioritize low-coverage concepts
        pri = df[df["concept_id"].isin(PRIORITY_CONCEPTS)]
        rest = df[~df["concept_id"].isin(PRIORITY_CONCEPTS)]
        picked = []
        for part, n in ((pri, n_per_lang * 3 // 4), (rest, n_per_lang - n_per_lang * 3 // 4)):
            part = part.drop_duplicates("passage_id")
            if part.empty:
                continue
            n = min(n, len(part))
            idx = rng.choice(part.index.to_numpy(), size=n, replace=False)
            picked.append(part.loc[idx])
        if not picked:
            continue
        sample = pd.concat(picked, ignore_index=True).drop_duplicates("passage_id")
        if len(sample) > n_per_lang:
            sample = sample.sample(n=n_per_lang, random_state=seed)
        sample["dev_language"] = lang
        frames.append(sample)
        held_books.update(sample["book_id"].tolist())

    if not frames:
        raise SystemExit("No BM25-only baselines found.")

    out_df = pd.concat(frames, ignore_index=True)
    out_path = OUT / "lexicon_dev_sample_v03.csv"
    cols = [
        c
        for c in (
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
            "bm25_score",
            "rank",
            "retrieval_method",
        )
        if c in out_df.columns
    ]
    out_df[cols].to_csv(out_path, index=False)

    if with_passages:
        need = set(out_df["passage_id"])
        texts = {}
        for lang in ("en", "de"):
            for para in iter_body_paragraphs(lang, with_neighbors=True):
                if para.passage_id in need:
                    texts[para.passage_id] = (para.text, para.prev_text, para.next_text)
                if len(texts) >= len(need):
                    break
        side = out_df[cols].copy()
        side["passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[0])
        side["prev_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[1])
        side["next_passage"] = side["passage_id"].map(lambda x: texts.get(x, ("", "", ""))[2])
        side.to_csv(OUT / "lexicon_dev_sample_v03_passages.csv", index=False)

    held_path = OUT / "lexicon_dev_held_out_books_v03.txt"
    held_path.write_text("\n".join(sorted(held_books)) + "\n", encoding="utf-8")
    meta = base_run_meta(
        kind="lexicon_dev_sample_v03",
        seed=seed,
        n_passages=len(out_df),
        n_held_out_books=len(held_books),
        priority_concepts=PRIORITY_CONCEPTS,
        note="Development sample for provisional lexicon expansion; exclude from eval sample.",
    )
    write_json(OUT / "lexicon_dev_sample_v03.meta.json", meta)
    log.info(f"wrote {out_path} ({len(out_df)} passages); held-out books={len(held_books)}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-lang", type=int, default=40)
    parser.add_argument("--with-passages", action="store_true")
    args = parser.parse_args()
    log = setup_logger(__file__)
    build_dev_sample(
        log, seed=args.seed, n_per_lang=args.n_per_lang, with_passages=args.with_passages
    )
    log.info("Log finished.")


if __name__ == "__main__":
    main()
