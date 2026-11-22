"""Exhaustive-vs-prefilter audit on a reproducible bilingual passage sample.

Finite samples cannot prove perfect equivalence. Investigate every difference.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_INTERIM
from src.retrieval.evidence import collect_concept_evidence, qualifies
from src.retrieval.lexicon_v02 import LEXICON_PATH_V03, load_lexicon
from src.retrieval.match import (
    build_concept_prefilter_needles,
    haystack_for_prefilter,
)
from src.retrieval.paragraphs import iter_body_paragraphs
from src.retrieval.regex_map import load_regex_map
from src.retrieval.run_meta import base_run_meta, write_json
from src.retrieval.sampling import ReservoirSample, derive_stratum_seed

OUT = DATA_INTERIM / "retrieval"
SEED = 20261123
DEFAULT_N = 500  # per language (manageable on laptop)


def _compare(text: str, lang: str, lex, rmap) -> tuple[set[str], set[str]]:
    hay = haystack_for_prefilter(text)
    exhaustive: set[str] = set()
    prefiltered: set[str] = set()
    for cid, ct in lex.concepts[lang].items():
        ev = collect_concept_evidence(text, cid, ct, lang, lex, rmap)
        ok = qualifies(ev, lang)
        if ok:
            exhaustive.add(cid)
        needles = build_concept_prefilter_needles(
            ct.academic, ct.craft, ct.paraphrases, lang
        )
        regex_hit = False
        for name in rmap.names_for_concept(lang, cid):
            pat = lex.example_regex.get(lang, {}).get(name)
            if pat is not None and pat.search(text):
                regex_hit = True
                break
        if regex_hit or any(n in hay for n in needles):
            if ok:
                prefiltered.add(cid)
    return exhaustive, prefiltered


def run_audit(log, *, n_per_lang: int = DEFAULT_N, seed: int = SEED) -> Path:
    t0 = time.time()
    lex = load_lexicon(LEXICON_PATH_V03)
    rmap = load_regex_map(CONFIG / "regex_concept_map_v03.yml")
    OUT.mkdir(parents=True, exist_ok=True)
    diff_rows = []
    summary = {}

    for lang in ("en", "de"):
        res = ReservoirSample(k=n_per_lang, seed=derive_stratum_seed(seed, f"audit::{lang}"))
        for para in iter_body_paragraphs(lang):
            res.consider(para)
        n_agree = n_diff = 0
        for para in res.reservoir:
            exh, pref = _compare(para.text, lang, lex, rmap)
            if exh == pref:
                n_agree += 1
            else:
                n_diff += 1
                diff_rows.append(
                    {
                        "passage_id": para.passage_id,
                        "language": lang,
                        "book_id": para.book_id,
                        "only_exhaustive": ";".join(sorted(exh - pref)),
                        "only_prefilter": ";".join(sorted(pref - exh)),
                        "exhaustive": ";".join(sorted(exh)),
                        "prefilter": ";".join(sorted(pref)),
                        "char_count": len(para.text),
                    }
                )
        summary[lang] = {
            "n_sampled": len(res.reservoir),
            "N_stream_seen": res.population_size,
            "n_agree": n_agree,
            "n_differ": n_diff,
            "seed": res.seed,
        }
        log.info(f"{lang}: agree={n_agree} differ={n_diff} / {len(res.reservoir)}")

    diff_path = OUT / "matcher_prefilter_audit_v04_diffs.csv"
    with diff_path.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "passage_id",
            "language",
            "book_id",
            "only_exhaustive",
            "only_prefilter",
            "exhaustive",
            "prefilter",
            "char_count",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in diff_rows:
            w.writerow(r)

    md = [
        "# Matcher prefilter vs exhaustive audit (v0.4)",
        "",
        "Finite bilingual sample — **does not prove perfect equivalence**.",
        "",
        f"Seed={seed}; n_per_lang={n_per_lang}.",
        "",
    ]
    for lang, s in summary.items():
        md.append(
            f"- **{lang}**: sampled={s['n_sampled']} (stream N≈{s['N_stream_seen']}), "
            f"agree={s['n_agree']}, differ={s['n_differ']}"
        )
    md.append("")
    md.append(f"Differences written to `{diff_path.name}` ({len(diff_rows)} rows).")
    md.append("")
    md.append(
        "Investigate every difference before treating the optimized matcher as validated."
    )
    report = OUT / "matcher_prefilter_audit_v04.md"
    report.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(
        OUT / "matcher_prefilter_audit_v04.meta.json",
        base_run_meta(
            kind="matcher_prefilter_audit_v04",
            seed=seed,
            n_per_lang=n_per_lang,
            summary=summary,
            n_differences=len(diff_rows),
            duration_sec=round(time.time() - t0, 1),
            equivalence_claim="none — finite sample only",
        ),
    )
    log.info(f"wrote {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-per-lang", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    log = setup_logger(__file__)
    run_audit(log, n_per_lang=args.n_per_lang, seed=args.seed)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
