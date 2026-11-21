"""Keyword research: DF/TF of PsyPer v0.2 academic/craft terms (streaming).

Usage:
  python -m src.retrieval.term_audit --lang de
  python -m src.retrieval.term_audit --lang en
  python -m src.retrieval.term_audit --lang all
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd
import yaml

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, ROOT
from src.retrieval.lexicon_v02 import classify_tier, iter_literal_terms, load_lexicon
from src.retrieval.paragraphs import iter_body_paragraphs
from src.retrieval.textnorm import normalize_text, phrase_in_text

OUT_DIR = DATA_INTERIM / "retrieval"
TIERS_PATH = OUT_DIR / "lexicon_term_tiers.yml"
SMOKE_PATH = OUT_DIR / "lexicon_pilot_smoke.md"

EXTRA_FP = [
    {"lang": "en", "text": "From an economic point of view, the contract is advantageous.", "label": "everyday_pov"},
    {"lang": "en", "text": "The marketing voice of the brand should stay consistent.", "label": "everyday_voice"},
    {"lang": "de", "text": "Aus finanzieller Sicht hat die Entscheidung Vorteile.", "label": "everyday_sicht"},
    {"lang": "de", "text": "Die Perspektive der Anleger ist entscheidend.", "label": "everyday_perspektive"},
]


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return ""


def _compile_term_matchers(terms: list[tuple[str, str, str]]):
    matchers = []
    for concept_id, tier, term in terms:
        p = normalize_text(term)
        if not p:
            continue
        if " " in p or "-" in p:
            matchers.append((concept_id, tier, term, "substr", p))
        else:
            matchers.append((concept_id, tier, term, "token", p))
    return matchers


def audit_lang(lang: str, log) -> pd.DataFrame:
    lex = load_lexicon()
    literal = iter_literal_terms(lex, lang)
    matchers = _compile_term_matchers(literal)
    substr = [(c, t, term, needle) for c, t, term, kind, needle in matchers if kind == "substr"]
    tokens_m = [(c, t, term, needle) for c, t, term, kind, needle in matchers if kind == "token"]
    stats = {
        (c, t, term): {"df": 0, "tf": 0, "df_primary": 0, "df_comparison": 0}
        for c, t, term, _, _ in matchers
    }
    regex_stats = {
        name: {"df": 0, "tf": 0, "df_primary": 0, "df_comparison": 0, "pat": pat}
        for name, pat in lex.example_regex.get(lang, {}).items()
    }

    log.info(f"{lang}: streaming scan over {len(matchers)} terms")
    n_docs = n_primary = n_comparison = 0
    for i, p in enumerate(iter_body_paragraphs(lang)):
        n_docs += 1
        role = p.corpus_role
        if role == "primary":
            n_primary += 1
        elif role == "comparison":
            n_comparison += 1
        norm = p.text_norm()
        toks = p.tokens()
        tokset = set(toks)
        for c, t, term, needle in substr:
            hits = norm.count(needle)
            if hits <= 0:
                continue
            s = stats[(c, t, term)]
            s["df"] += 1
            s["tf"] += hits
            if role == "primary":
                s["df_primary"] += 1
            elif role == "comparison":
                s["df_comparison"] += 1
        for c, t, term, needle in tokens_m:
            if needle not in tokset:
                continue
            hits = toks.count(needle)
            s = stats[(c, t, term)]
            s["df"] += 1
            s["tf"] += hits
            if role == "primary":
                s["df_primary"] += 1
            elif role == "comparison":
                s["df_comparison"] += 1
        for name, rs in regex_stats.items():
            found = rs["pat"].findall(p.text)
            if not found:
                continue
            rs["df"] += 1
            rs["tf"] += len(found)
            if role == "primary":
                rs["df_primary"] += 1
            elif role == "comparison":
                rs["df_comparison"] += 1
        del toks, tokset, norm
        if i and i % 50000 == 0:
            log.info(f"  {lang}: {i} paragraphs")

    log.info(f"{lang}: {n_docs} body paragraphs")
    rows = []
    for c, t, term, kind, needle in matchers:
        s = stats[(c, t, term)]
        df = s["df"]
        rows.append(
            {
                "lang": lang,
                "concept_id": c,
                "source_tier": t,
                "assigned_tier": classify_tier(lang, t, term),
                "term": term,
                "n_chars": len(term),
                "n_tokens": len(term.split()),
                "df": df,
                "tf": s["tf"],
                "df_share": round(df / n_docs, 6) if n_docs else 0.0,
                "df_primary": s["df_primary"],
                "df_comparison": s["df_comparison"],
                "df_share_primary": round(s["df_primary"] / n_primary, 6) if n_primary else 0.0,
                "df_share_comparison": round(s["df_comparison"] / n_comparison, 6) if n_comparison else 0.0,
                "zero_hit": df == 0,
            }
        )
    for name, rs in regex_stats.items():
        df = rs["df"]
        rows.append(
            {
                "lang": lang,
                "concept_id": f"regex:{name}",
                "source_tier": "regex",
                "assigned_tier": "high",
                "term": rs["pat"].pattern,
                "n_chars": len(rs["pat"].pattern),
                "n_tokens": 0,
                "df": df,
                "tf": rs["tf"],
                "df_share": round(df / n_docs, 6) if n_docs else 0.0,
                "df_primary": rs["df_primary"],
                "df_comparison": rs["df_comparison"],
                "df_share_primary": round(rs["df_primary"] / n_primary, 6) if n_primary else 0.0,
                "df_share_comparison": round(rs["df_comparison"] / n_comparison, 6) if n_comparison else 0.0,
                "zero_hit": df == 0,
            }
        )

    df_out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"lexicon_term_audit_{lang}.csv"
    df_out.to_csv(out_csv, index=False)
    meta = {
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "lang": lang,
        "lexicon_version": lex.version,
        "lexicon_path": str(lex.path.relative_to(ROOT)),
        "git_commit": _git_commit(),
        "n_paragraphs": n_docs,
        "n_primary_paras": n_primary,
        "n_comparison_paras": n_comparison,
        "n_terms": len(df_out),
        "zero_hit_terms": int(df_out["zero_hit"].sum()),
        "peak_strategy": "stream_one_paragraph",
    }
    (OUT_DIR / f"lexicon_term_audit_{lang}.meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    log.info(f"wrote {out_csv} ({len(df_out)} rows, {meta['zero_hit_terms']} zero-hit)")
    return df_out


def write_tiers(audits: dict[str, pd.DataFrame], log) -> dict:
    # Merge with existing tiers file so single-lang runs do not wipe the other language.
    existing = {}
    if TIERS_PATH.is_file():
        existing = yaml.safe_load(TIERS_PATH.read_text(encoding="utf-8")) or {}

    tiers = {
        "version": "0.2-draft-audit",
        "built": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "git_commit": _git_commit(),
        "rules": {
            "high": "academic terms and example_regex with df>0; zero-hit academic → unvalidated",
            "supporting": "multi-word craft with df>0",
            "demoted": "ambiguous/short craft; context never solo",
            "unvalidated": "zero corpus hits — keep for semantic retrieval later",
            "catalogue_candidates": "short multi-word high/supporting survivors for catalogue_queries.yml",
        },
        "languages": dict((existing.get("languages") or {})),
        "catalogue_candidates": dict((existing.get("catalogue_candidates") or {"en": [], "de": []})),
    }
    catalogue = tiers["catalogue_candidates"]
    for lang in ("en", "de"):
        catalogue.setdefault(lang, [])

    for lang, df in audits.items():
        if df is None or df.empty:
            continue
        high, supporting, demoted, unvalidated = [], [], [], []
        cat_lang: list[str] = []
        for _, r in df.iterrows():
            if str(r.concept_id).startswith("regex:"):
                entry = {"term": r.term, "concept_id": r.concept_id, "kind": "regex", "df": int(r.df)}
                (unvalidated if r.zero_hit else high).append(entry)
                continue
            entry = {
                "term": r.term,
                "concept_id": r.concept_id,
                "source_tier": r.source_tier,
                "df": int(r.df),
            }
            if r.zero_hit:
                unvalidated.append(entry)
            elif r.assigned_tier == "high":
                high.append(entry)
            elif r.assigned_tier == "supporting":
                supporting.append(entry)
            else:
                demoted.append(entry)
            ntok = int(r.n_tokens)
            if (
                not r.zero_hit
                and r.assigned_tier in ("high", "supporting")
                and 2 <= ntok <= 6
                and not str(r.concept_id).startswith("regex:")
            ):
                cat_lang.append(r.term)

        tiers["languages"][lang] = {
            "high": high,
            "supporting": supporting,
            "demoted": demoted,
            "unvalidated": unvalidated,
            "counts": {
                "high": len(high),
                "supporting": len(supporting),
                "demoted": len(demoted),
                "unvalidated": len(unvalidated),
            },
        }
        seen = set()
        uniq = []
        for t in cat_lang:
            key = t.casefold()
            if key not in seen:
                seen.add(key)
                uniq.append(t)
        catalogue[lang] = uniq[:40]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TIERS_PATH.write_text(yaml.safe_dump(tiers, allow_unicode=True, sort_keys=False), encoding="utf-8")
    log.info(f"wrote {TIERS_PATH}")
    return tiers


def pilot_smoke(log) -> None:
    lex = load_lexicon()
    lines = [
        "# Lexicon pilot smoke (v0.2-draft)",
        "",
        f"Built: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Lexicon: `{lex.path.relative_to(ROOT)}` ({lex.version})",
        "",
        "Rule: a passage is a *literal candidate* if any strong high term/regex hits,",
        "or weak POV surfaces plus an independent supporting cue, or ≥2 supporting.",
        "Ambiguous `point of view` / `regex:general_pov` alone never counts.",
        "",
    ]
    cases = list(lex.pilot_examples) + EXTRA_FP
    ok = fail = 0
    weak = {
        "point of view", "pov", "povs", "viewpoint", "viewpoints", "regex:general_pov"
    }
    for case in cases:
        lang = case["lang"]
        text = case["text"]
        norm = normalize_text(text)
        expected = set(case.get("expected") or [])
        reason = case.get("reason") or case.get("label") or ""
        hits_high, hits_supp = [], []
        for cid, ct in lex.concepts[lang].items():
            for term in ct.academic:
                if phrase_in_text(term, norm):
                    hits_high.append(f"{cid}:{term}")
            for term in ct.craft:
                if not phrase_in_text(term, norm):
                    continue
                tier = classify_tier(lang, "craft", term)
                if tier == "supporting":
                    hits_supp.append(f"{cid}:{term}")
                elif tier == "high":
                    hits_high.append(f"{cid}:{term}")
        for name, pat in lex.example_regex.get(lang, {}).items():
            if pat.search(text):
                hits_high.append(f"regex:{name}")
        strong_high = [
            h for h in hits_high
            if h.casefold().split(":", 1)[-1] not in weak
            and not h.casefold().startswith("regex:general_pov")
        ]
        weak_high = [h for h in hits_high if h not in strong_high]
        real_supp = [s for s in hits_supp if s.casefold().split(":", 1)[-1] not in weak]
        is_cand = bool(strong_high) or (bool(weak_high) and bool(real_supp)) or len(real_supp) >= 2
        want_empty = len(expected) == 0
        if want_empty and not is_cand:
            status = "PASS"
            ok += 1
        elif want_empty and is_cand:
            status = "FAIL"
            fail += 1
        elif expected and is_cand:
            status = "PASS"
            ok += 1
        else:
            status = "CHECK"
        lines.append(f"## [{status}] {lang}: {text[:90]}{'…' if len(text) > 90 else ''}")
        lines.append(f"- reason/expected: {reason or expected}")
        lines.append(f"- high hits: {hits_high or '—'}")
        lines.append(f"- supporting hits: {hits_supp or '—'}")
        lines.append(f"- literal_candidate: {is_cand}")
        lines.append("")
    lines.append(f"**Summary:** {ok} PASS, {fail} FAIL, {len(cases) - ok - fail} CHECK")
    SMOKE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info(f"wrote {SMOKE_PATH} ({ok} PASS, {fail} FAIL)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    args = parser.parse_args()
    log = setup_logger(__file__)
    langs = ["en", "de"] if args.lang == "all" else [args.lang]
    audits = {}
    for lang in langs:
        audits[lang] = audit_lang(lang, log)
    write_tiers(audits, log)
    pilot_smoke(log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
