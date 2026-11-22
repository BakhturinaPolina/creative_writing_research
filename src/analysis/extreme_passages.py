"""Diagnose extreme-length passages and decide the resegmentation gate.

Gate rule (analysis v2): resegment only when a passage is classified as a
collapsed chapter with *recoverable* internal paragraph breaks (blank lines or
dense newlines that can be split safely). OCR-damaged and genuine long units
are recorded and left unchanged. Original passage IDs and texts are preserved
in gitignored JSON; this module writes metadata-only tables.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.narrative_perspective import (
    LANGUAGE_COLORS,
    LANG_LABEL,
    load_retrieval_results,
)
from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, DATA_PROCESSED, ROOT
from src.retrieval.paragraphs import JSON_ROOT, MIN_PARA_CHARS

REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
FIGURES = REPORTS / "figures" / "02_narrative_perspective"
EXTREME_CHARS = 100_000


def _load_passage_text(passage_id: str) -> tuple[str, int]:
    """Return (text, n_paragraphs_in_chapter) for a passage_id."""
    book_id, rest = passage_id.split(":", 1)
    lang = book_id[:2]
    parts = rest.split(":")
    c_idx = int(parts[0][1:])
    p_idx = int(parts[1][1:])
    path = JSON_ROOT / lang / f"{book_id}.json"
    if not path.is_file():
        return "", 0
    rec = json.loads(path.read_text(encoding="utf-8"))
    chapters = rec.get("chapters") or []
    if c_idx >= len(chapters):
        return "", 0
    paras = chapters[c_idx].get("paragraphs") or []
    text = paras[p_idx] if p_idx < len(paras) else ""
    return text or "", len(paras)


def classify_extreme_text(text: str, qc_flags: str, n_chapter_paras: int) -> tuple[str, bool]:
    """Return (class_label, recoverable_for_resegment).

    recoverable=True only when internal blank-line structure exists and can be
    split into multiple eligible paragraphs without inventing boundaries.
    """
    flags = (qc_flags or "").lower()
    n_nl = text.count("\n")
    n_blank = text.count("\n\n")
    # Sentence-ish length without newlines → collapsed extraction.
    if "possible_ocr_spacing" in flags and n_nl == 0:
        return "ocr_artifact_collapsed", False
    if n_blank >= 3:
        # Recoverable: blank-line paragraph structure still present.
        pieces = [p.strip() for p in text.split("\n\n") if p.strip()]
        eligible = [p for p in pieces if len(p) >= MIN_PARA_CHARS]
        if len(eligible) >= 2:
            return "collapsed_chapter_recoverable_breaks", True
        return "collapsed_chapter_few_eligible_breaks", False
    if n_nl >= 20:
        # Newlines without blank breaks — weak signal; do not auto-resegment.
        return "collapsed_chapter_newlines_no_blank_breaks", False
    if n_nl == 0 and len(text) >= EXTREME_CHARS:
        if n_chapter_paras == 1:
            return "concatenated_or_unsegmented_chapter", False
        return "oversized_paragraph_in_multi_para_chapter", False
    return "genuine_long_textual_unit", False


def diagnose_extreme_passages() -> tuple[pd.DataFrame, dict]:
    frames = []
    for lang in ("en", "de"):
        path = DATA_PROCESSED / f"corpus_quality_extreme_passages_{lang}_v1_candidate.csv"
        if path.is_file():
            frames.append(pd.read_csv(path, dtype=str).fillna(""))
    extreme = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # Lexical assignment share attributable to extreme passages.
    lexical = load_retrieval_results("lexical_v03")
    extreme_ids = set(extreme["passage_id"]) if len(extreme) else set()
    n_assign_total = len(lexical)
    n_assign_extreme = (
        int(lexical["passage_id"].isin(extreme_ids).sum()) if len(lexical) else 0
    )

    # Eligible word totals from passage manifests.
    eligible_words = {}
    extreme_words = {}
    length_samples = []
    for lang in ("en", "de"):
        pm = pd.read_csv(
            DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.csv",
            usecols=["passage_id", "word_count", "character_count"],
            dtype=str,
        )
        pm["word_count"] = pd.to_numeric(pm["word_count"], errors="coerce").fillna(0)
        pm["character_count"] = pd.to_numeric(pm["character_count"], errors="coerce").fillna(0)
        eligible_words[lang] = float(pm["word_count"].sum())
        eids = set(extreme.loc[extreme["language"] == lang, "passage_id"]) if len(extreme) else set()
        extreme_words[lang] = float(pm.loc[pm["passage_id"].isin(eids), "word_count"].sum())
        length_samples.append(pm.assign(language=lang))

    rows = []
    n_recoverable = 0
    for _, r in extreme.iterrows():
        pid = r["passage_id"]
        text, n_ch_paras = _load_passage_text(pid)
        cls, recoverable = classify_extreme_text(
            text, r.get("qc_flags", ""), n_ch_paras
        )
        if recoverable:
            n_recoverable += 1
        rows.append(
            {
                "passage_id": pid,
                "book_id": r.get("book_id", ""),
                "language": r.get("language", ""),
                "primary_group": r.get("primary_group", ""),
                "character_count": int(float(r.get("character_count") or 0)),
                "word_count": int(float(r.get("word_count") or 0)),
                "qc_flags": r.get("qc_flags", ""),
                "n_paragraphs_in_chapter": n_ch_paras,
                "n_newlines": text.count("\n"),
                "n_blank_breaks": text.count("\n\n"),
                "classification": cls,
                "recoverable_for_resegment": recoverable,
                "gate_action": "resegment" if recoverable else "retain_original",
            }
        )
    detail = pd.DataFrame(rows)

    gate_fires = n_recoverable > 0
    summary = {
        "n_extreme_passages": len(detail),
        "n_recoverable_collapsed": n_recoverable,
        "resegment_gate_fires": gate_fires,
        "decision": (
            "Create v2 resegment snapshot and rerun retrieval"
            if gate_fires
            else "Retain v0.3 retrieval execution; keep S3 sensitivity (drop extremes)"
        ),
        "lexical_assignments_total": n_assign_total,
        "lexical_assignments_in_extreme_passages": n_assign_extreme,
        "pct_lexical_assignments_in_extreme": (
            100.0 * n_assign_extreme / n_assign_total if n_assign_total else 0.0
        ),
        "eligible_words": eligible_words,
        "extreme_words": extreme_words,
        "pct_eligible_words_in_extreme": {
            lang: (
                100.0 * extreme_words[lang] / eligible_words[lang]
                if eligible_words[lang]
                else 0.0
            )
            for lang in ("en", "de")
        },
        "classification_counts": detail["classification"].value_counts().to_dict()
        if len(detail)
        else {},
    }

    # Length distribution table (percentiles of all eligible passages).
    all_pm = pd.concat(length_samples, ignore_index=True) if length_samples else pd.DataFrame()
    pct_rows = []
    for lang in ("en", "de"):
        sub = all_pm[all_pm["language"] == lang] if len(all_pm) else all_pm
        chars = sub["character_count"] if len(sub) else pd.Series(dtype=float)
        pct_rows.append(
            {
                "language": lang,
                "n_passages": len(sub),
                "char_p50": float(chars.quantile(0.50)) if len(chars) else 0,
                "char_p90": float(chars.quantile(0.90)) if len(chars) else 0,
                "char_p99": float(chars.quantile(0.99)) if len(chars) else 0,
                "char_p999": float(chars.quantile(0.999)) if len(chars) else 0,
                "char_max": float(chars.max()) if len(chars) else 0,
                "n_extreme_ge_100k": int((chars >= EXTREME_CHARS).sum()) if len(chars) else 0,
            }
        )
    percentiles = pd.DataFrame(pct_rows)
    return detail, summary, percentiles, all_pm


def plot_length_distribution(all_pm: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, lang in zip(axes, ("en", "de")):
        sub = all_pm[all_pm["language"] == lang]
        chars = np.clip(sub["character_count"].to_numpy(dtype=float), 1, None)
        ax.hist(np.log10(chars), bins=60, color=LANGUAGE_COLORS[LANG_LABEL[lang]], alpha=0.85)
        ax.axvline(np.log10(EXTREME_CHARS), color="red", linestyle="--", linewidth=1, label="100k chars")
        ax.set_title(LANG_LABEL[lang])
        ax.set_xlabel("log10(character count)")
        ax.set_ylabel("Passages")
        ax.legend(fontsize=8)
    fig.suptitle("Eligible passage length distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "02_passage_length_distribution.png", bbox_inches="tight")
    fig.savefig(out_dir / "02_passage_length_distribution.svg", bbox_inches="tight")
    plt.close(fig)


def build_wordcount_discrepancy() -> pd.DataFrame:
    """Per-book body vs eligible-passage word gaps for analysis-set books."""
    rows = []
    for lang in ("en", "de"):
        stats = pd.read_csv(DATA_PROCESSED / f"{lang}_book_stats.csv", dtype=str).fillna("")
        stats = stats[stats["in_analysis_set"].astype(str).str.lower().isin(("true", "1"))]
        pm = pd.read_csv(
            DATA_PROCESSED / f"passage_manifest_{lang}_v1_candidate.csv",
            usecols=["book_id", "word_count"],
            dtype=str,
        )
        pm["word_count"] = pd.to_numeric(pm["word_count"], errors="coerce").fillna(0)
        elig = pm.groupby("book_id")["word_count"].sum()
        for _, r in stats.iterrows():
            bid = r["book_id"]
            body = float(pd.to_numeric(r.get("body_words"), errors="coerce") or 0)
            raw = float(pd.to_numeric(r.get("n_words"), errors="coerce") or 0)
            eligible = float(elig.get(bid, 0))
            abs_gap = body - eligible
            rel_gap = (abs_gap / body) if body else 0.0
            rows.append(
                {
                    "book_id": bid,
                    "language": lang,
                    "raw_word_count": raw,
                    "cleaned_body_word_count": body,
                    "eligible_passage_word_count": eligible,
                    "excluded_word_count": max(0.0, abs_gap),
                    "absolute_discrepancy_body_minus_eligible": abs_gap,
                    "relative_discrepancy": rel_gap,
                    "note": (
                        "Eligible passages are body paragraphs with >=40 characters; "
                        "body_words include shorter paragraphs and front/back filtering differences."
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    log = setup_logger(__file__)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    detail, summary, percentiles, all_pm = diagnose_extreme_passages()
    detail.to_csv(TABLES / "02_extreme_passages.csv", index=False)
    percentiles.to_csv(TABLES / "02_passage_length_percentiles.csv", index=False)
    (DATA_PROCESSED / "extreme_passage_gate_v2.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_length_distribution(all_pm, FIGURES)

    disc = build_wordcount_discrepancy()
    disc.to_csv(TABLES / "02_wordcount_discrepancy.csv", index=False)
    log.info(
        f"Extreme passages={summary['n_extreme_passages']} "
        f"recoverable={summary['n_recoverable_collapsed']} "
        f"gate_fires={summary['resegment_gate_fires']}"
    )
    log.info(f"Decision: {summary['decision']}")
    n_mismatch = int((disc["absolute_discrepancy_body_minus_eligible"].abs() > 1).sum())
    log.info(f"Books with |body-eligible|>1: {n_mismatch} / {len(disc)}")


if __name__ == "__main__":
    main()
