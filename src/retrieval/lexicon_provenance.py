"""Lexicon term provenance v0.5 — controlled vocabulary, no invented sources."""
from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_INTERIM, DATA_PROCESSED
from src.retrieval.run_meta import base_run_meta, write_json

OUT = DATA_INTERIM / "retrieval"
LEX_V03 = CONFIG / "pov_lexicon_psyper_en_de_v0.3.yml"
LEX_V02 = CONFIG / "pov_lexicon_psyper_en_de_v0.2.yml"
LEX_V01 = CONFIG / "pov_lexicon.yml"

PROVENANCE_TYPES = (
    "psyper_schema",
    "academic_narratology",
    "creative_writing_terminology",
    "corpus_attested",
    "inherited_lexicon",
    "ai_suggested",
    "translation",
    "unverified_paraphrase",
    "unknown",
)

# Ambiguous German expressions requiring human conceptual review (no remaps).
AMBIGUOUS_DE = {
    "auktoriale erzählsituation": "stanzel_vs_genette_focalization",
    "auktoriale erzählweise": "stanzel_vs_genette_focalization",
    "auktoriale erzählung": "stanzel_vs_genette_focalization",
    "auktoriale perspektive": "stanzel_vs_genette_focalization",
    "auktoriale erzähler": "stanzel_vs_genette_focalization",
    "auktorial": "stanzel_vs_genette_focalization",
    "personale erzählsituation": "stanzel_vs_genette_focalization",
    "personale erzählweise": "stanzel_vs_genette_focalization",
    "personale erzählung": "stanzel_vs_genette_focalization",
    "personale perspektive": "stanzel_vs_genette_focalization",
    "allwissenheit": "omniscience_scope_vs_zero_focalization",
    "allwissender erzähler": "omniscience_scope_vs_zero_focalization",
    "allwissend": "omniscience_scope_vs_zero_focalization",
    "wissensvorsprung": "reader_knowledge_vs_focalization_vs_information_control",
    "wissensvorsprung des erzählers": "reader_knowledge_vs_focalization_vs_information_control",
}

# High-gain / priority DE forms from v0.3 audit (enrich notes only).
PRIORITY_DE_SUBSTRINGS = (
    "allwissen",
    "auktorial",
    "personal",
    "wissensvorsprung",
    "nullfokalis",
)


def _term_id(concept_id: str, language: str, term: str) -> str:
    raw = f"{concept_id}\0{language}\0{term}".encode("utf-8")
    return hashlib.blake2b(raw, digest_size=8).hexdigest()


def _norm(term: str) -> str:
    return re.sub(r"\s+", " ", str(term).casefold().strip())


def _collect_terms(raw_lex: dict) -> dict[tuple[str, str, str], set[str]]:
    out: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for block in raw_lex.get("concepts") or []:
        cid = block.get("id")
        for lang in ("en", "de"):
            node = block.get(lang) or {}
            for term_type, key in (
                ("academic", "academic"),
                ("craft", "craft"),
                ("paraphrase", "paraphrases"),
                ("context", "context"),
            ):
                for term in node.get(key) or []:
                    out[(cid, lang, term_type)].add(_norm(term))
            for term in (node.get("evidence_overrides") or {}):
                out[(cid, lang, "override")].add(_norm(term))
    return out


def _attestation_counts() -> dict[tuple[str, str, str], int]:
    """Count concept×lang×normalized matched_term occurrences from lexical v0.3."""
    counts: Counter = Counter()
    for lang in ("en", "de"):
        path = DATA_PROCESSED / f"pov_baseline_b_v03_lexical_{lang}.csv"
        if not path.is_file():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for _, row in df.iterrows():
            cid = row.get("retrieval_concept_id") or row.get("concept_id") or ""
            terms = row.get("matched_terms") or ""
            for part in re.split(r"[|;]+", terms):
                t = _norm(part)
                if t:
                    counts[(cid, lang, t)] += 1
    return dict(counts)


def build_provenance_v05(log) -> pd.DataFrame:
    raw03 = yaml.safe_load(LEX_V03.read_text(encoding="utf-8")) or {}
    raw02 = yaml.safe_load(LEX_V02.read_text(encoding="utf-8")) or {}
    v02 = _collect_terms(raw02)
    attest = _attestation_counts()

    # PsyPer-related: concepts that carry non-empty psyper tag lists.
    psyper_concepts = {
        block.get("id")
        for block in (raw03.get("concepts") or [])
        if block.get("psyper")
    }

    rows = []
    for block in raw03.get("concepts") or []:
        cid = block.get("id")
        for lang in ("en", "de"):
            node = block.get(lang) or {}
            overrides = {
                _norm(k): str(v) for k, v in (node.get("evidence_overrides") or {}).items()
            }
            for term_type, key in (
                ("academic", "academic"),
                ("craft", "craft"),
                ("paraphrase", "paraphrases"),
                ("context", "context"),
            ):
                for term in node.get(key) or []:
                    original = str(term)
                    normalized = _norm(term)
                    in_v02 = normalized in v02.get((cid, lang, term_type), set())
                    n_att = int(attest.get((cid, lang, normalized), 0))
                    corpus_attested = n_att > 0

                    # Controlled provenance — only with evidence; never invent citations.
                    if in_v02:
                        provenance_type = "inherited_lexicon"
                        source_reference = "pov_lexicon_psyper_en_de_v0.2.yml"
                        lexicon_version_introduced = "0.2"
                        human_review_status = "inherited"
                        notes = (
                            "Inherited from earlier lexicon version; original theoretical "
                            "or empirical source not established in project records."
                        )
                    elif term_type == "paraphrase":
                        provenance_type = "unverified_paraphrase"
                        source_reference = "pov_lexicon_psyper_en_de_v0.3.yml"
                        lexicon_version_introduced = "0.3"
                        human_review_status = "needs_human_review"
                        notes = (
                            "v0.3 paraphrase expansion; treat as AI-suggested / brief-derived "
                            "until verified. Separate from corpus_attested flag."
                        )
                    else:
                        # New non-paraphrase v0.3 terms: unknown origin unless priority notes apply.
                        provenance_type = "unknown"
                        source_reference = "pov_lexicon_psyper_en_de_v0.3.yml"
                        lexicon_version_introduced = "0.3"
                        human_review_status = "needs_human_review"
                        notes = (
                            "Added in v0.3; original source not established. "
                            "Do not assume corpus derivation solely from attestation."
                        )

                    # Academic tier alone does not prove academic_narratology without a citation.
                    # Craft tier alone does not prove creative_writing_terminology without evidence.
                    if term_type == "academic" and provenance_type == "inherited_lexicon":
                        notes += " Listed under academic tier; source citation unknown."
                    if term_type == "craft" and provenance_type == "inherited_lexicon":
                        notes += " Listed under craft tier; source citation unknown."

                    # Priority enrichment flags (documentation only).
                    if lang == "de" and any(s in normalized for s in PRIORITY_DE_SUBSTRINGS):
                        human_review_status = "priority_human_review"
                        notes += " Priority DE expression (gain / ambiguity / low-coverage cluster)."

                    # Ambiguity documentation — no conceptual remaps.
                    conceptual_ambiguity = AMBIGUOUS_DE.get(normalized, "")
                    if not conceptual_ambiguity:
                        for key_amb, label in AMBIGUOUS_DE.items():
                            if key_amb in normalized or normalized in key_amb:
                                conceptual_ambiguity = label
                                break

                    # PsyPer schema link is about concept mapping, not term origin.
                    if cid in psyper_concepts and term_type == "academic" and in_v02:
                        # Soft hint only when term is academic and concept has psyper tags —
                        # still inherited unless we have an actual scheme citation.
                        notes += (
                            f" Concept carries psyper-related tags {block.get('psyper')}; "
                            "this does not make the term an official PsyPer category label."
                        )

                    # AI-suggested for unverified paraphrases (explicit category).
                    if provenance_type == "unverified_paraphrase":
                        # Also record companion type in notes; keep single primary type.
                        notes += " Primary type unverified_paraphrase; may also be ai_suggested."

                    rows.append(
                        {
                            "term_id": _term_id(cid, lang, original),
                            "concept_id": cid,
                            "language": lang,
                            "original_term": original,
                            "normalized_term": normalized,
                            "term_type": term_type,
                            "evidence_tier": overrides.get(normalized, ""),
                            "lexicon_version_introduced": lexicon_version_introduced,
                            "provenance_type": provenance_type,
                            "source_reference": source_reference,
                            "source_location": f"concepts.{cid}.{lang}.{key}",
                            "development_passage_id": "",
                            "corpus_attested": int(corpus_attested),
                            "corpus_attestation_count": n_att,
                            "human_review_status": human_review_status,
                            "conceptual_ambiguity": conceptual_ambiguity,
                            "notes": notes.strip(),
                        }
                    )

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "lexicon_term_provenance_v05.csv"
    df.to_csv(path, index=False)

    n_unique_expr = df["normalized_term"].nunique()
    summary_lines = [
        "# Lexicon provenance completeness (v0.5)",
        "",
        f"- Lexicon **entries** (concept × language × term): **{len(df)}**",
        f"- Unique normalized **expressions** (may map to several concepts): **{n_unique_expr}**",
        "",
        "## By provenance_type",
        "",
    ]
    for k, v in df["provenance_type"].value_counts().items():
        summary_lines.append(f"- `{k}`: {v}")
    summary_lines += [
        "",
        "## By term_type (evidence role)",
        "",
    ]
    for k, v in df["term_type"].value_counts().items():
        summary_lines.append(f"- `{k}`: {v}")
    summary_lines += [
        "",
        "## Traceability",
        "",
        f"- Traceable source beyond lexicon file version: "
        f"**{(df['provenance_type'] == 'psyper_schema').sum() + (df['provenance_type'] == 'academic_narratology').sum()}** "
        "(none assigned without supporting citation evidence)",
        f"- Entries with empirical corpus attestation (`corpus_attested=1`): "
        f"**{int(df['corpus_attested'].sum())}**",
        f"- Awaiting human verification (`needs_human_review` / `priority_human_review`): "
        f"**{df['human_review_status'].isin(['needs_human_review','priority_human_review']).sum()}**",
        f"- Unknown provenance explicitly recorded: "
        f"**{(df['provenance_type'] == 'unknown').sum()}**",
        f"- Inherited lexicon (original source not established): "
        f"**{(df['provenance_type'] == 'inherited_lexicon').sum()}**",
        "",
        "## Conceptual ambiguity flags",
        "",
        f"- Rows with non-empty `conceptual_ambiguity`: "
        f"**{(df['conceptual_ambiguity'].astype(str).str.len() > 0).sum()}**",
        "",
        "Unknown provenance is recorded as `unknown`. Corpus attestation is independent of "
        "whether a term was originally derived from the corpus.",
        "",
        "Do **not** equate Stanzel narrative situations with Genette focalization categories "
        "without human audit decisions. No conceptual remaps applied in v0.5.",
    ]
    (OUT / "lexicon_provenance_completeness_v05.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    write_json(
        OUT / "lexicon_term_provenance_v05.meta.json",
        base_run_meta(
            kind="lexicon_term_provenance_v05",
            n_entries=len(df),
            n_unique_expressions=int(n_unique_expr),
            provenance_types=df["provenance_type"].value_counts().to_dict(),
            controlled_vocabulary=list(PROVENANCE_TYPES),
        ),
    )
    log.info(f"wrote {path} ({len(df)} entries; {n_unique_expr} unique expressions)")
    return df


def write_psyper_distinction_doc(log) -> Path:
    """Document PsyPer-related tags vs project retrieval concepts."""
    raw03 = yaml.safe_load(LEX_V03.read_text(encoding="utf-8")) or {}
    concepts = raw03.get("concepts") or []
    lines = [
        "# PsyPer-related tags versus project retrieval concepts (v0.5)",
        "",
        "This project uses a **36-concept retrieval lexicon** informed by PsyPer-related",
        "annotation interests. **Not all 36 retrieval concepts are official PsyPer categories.**",
        "",
        "PsyPer mappings in the YAML (`psyper:` lists) indicate *potential relevance* to",
        "annotation codes — not logical equivalence, and not an official PsyPer scheme dump.",
        "",
        "Original PsyPer coding manuals are **not** present as standalone documents in this",
        "repository. Available in-repo materials:",
        "",
        "- `config/pov_lexicon_psyper_en_de_v0.3.yml` — retrieval lexicon with `psyper` hints",
        "- `docs/scientific_readme.md` §9.1 — project POV taxonomy dimensions",
        "- `docs/scientific_readme.md` §10 — project annotation relevance/context/function labels",
        "",
        "## Project retrieval concepts (36)",
        "",
        "| concept_id | psyper_related_tags | note |",
        "|---|---|---|",
    ]
    for block in concepts:
        cid = block.get("id", "")
        tags = block.get("psyper") or []
        note = (block.get("note") or "").replace("|", "/")
        tag_s = ", ".join(str(t) for t in tags) if tags else "*(none / project-specific)*"
        lines.append(f"| `{cid}` | {tag_s} | {note} |")

    lines += [
        "",
        "## Distinctions preserved for human audit",
        "",
        "Do not automatically equate:",
        "",
        "- Grammatical person vs narrator participation",
        "- Narratorial stance vs focalization",
        "- Information access vs reader knowledge",
        "- Stanzel *auktoriale / personale Erzählsituation* vs Genette zero / internal focalization",
        "- *Wissensvorsprung* / dramatic irony vs focalization (may relate to `information_control` / `reader_effect`)",
        "",
        "No conceptual remaps were applied in v0.5 pending human audit decisions.",
        "",
    ]
    path = Path("docs") / "psyper_vs_retrieval_concepts_v05.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"wrote {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-doc", action="store_true")
    args = parser.parse_args()
    log = setup_logger(__file__)
    build_provenance_v05(log)
    if not args.skip_doc:
        write_psyper_distinction_doc(log)
    log.info("Log finished.")


if __name__ == "__main__":
    main()
