"""BM25 query construction: transparent dedupe and optional weights."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.retrieval.lexicon_v02 import LexiconV02
from src.retrieval.textnorm import tokenize

# Function words that must not dominate expanded queries (EN + DE).
STOPWORDS = {
    "en": {
        "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "with",
        "from", "by", "as", "is", "are", "be", "that", "this", "it", "at",
    },
    "de": {
        "der", "die", "das", "den", "dem", "des", "ein", "eine", "einer", "eines",
        "einem", "einen", "und", "oder", "mit", "von", "zu", "zur", "zum", "im",
        "in", "am", "auf", "für", "als", "ist", "sind", "wird",
    },
}


@dataclass
class QueryConfig:
    """Optional term weights and construction knobs."""
    academic_weight: float = 1.0
    craft_weight: float = 1.0
    paraphrase_weight: float = 1.0
    drop_stopwords: bool = True
    dedupe_tokens: bool = True
    # If False, repeated tokens across phrases are kept once (default).
    # Set True only when intentional repeated-token weighting is desired.
    allow_repeated_token_weighting: bool = False
    skip_demoted_singletons: bool = True


@dataclass
class ConceptQuery:
    concept_id: str
    expressions: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    token_weights: dict[str, float] = field(default_factory=dict)


def build_concept_query(
    concept_id: str,
    academic: list[str],
    craft: list[str],
    paraphrases: list[str],
    lang: str,
    demoted_terms: set[str] | None = None,
    cfg: QueryConfig | None = None,
) -> ConceptQuery:
    cfg = cfg or QueryConfig()
    demoted_terms = demoted_terms or set()
    stops = STOPWORDS.get(lang, set()) if cfg.drop_stopwords else set()

    # Deduplicate identical expressions (casefold).
    seen_expr: dict[str, str] = {}
    weighted: list[tuple[str, float]] = []
    for phrases, w in (
        (academic, cfg.academic_weight),
        (craft, cfg.craft_weight),
        (paraphrases, cfg.paraphrase_weight),
    ):
        for phrase in phrases:
            key = phrase.casefold().strip()
            if not key or key in seen_expr:
                continue
            if (
                cfg.skip_demoted_singletons
                and key in demoted_terms
                and " " not in phrase
                and "-" not in phrase
            ):
                continue
            seen_expr[key] = phrase
            weighted.append((phrase, w))

    q = ConceptQuery(concept_id=concept_id, expressions=list(seen_expr.values()))
    token_weight: dict[str, float] = {}
    ordered_tokens: list[str] = []
    seen_tok: set[str] = set()
    for phrase, w in weighted:
        for tok in tokenize(phrase):
            if tok in stops:
                continue
            if cfg.dedupe_tokens and not cfg.allow_repeated_token_weighting:
                if tok in seen_tok:
                    # Keep max weight if same token appears in another expression.
                    token_weight[tok] = max(token_weight.get(tok, 0.0), w)
                    continue
                seen_tok.add(tok)
                ordered_tokens.append(tok)
                token_weight[tok] = w
            else:
                ordered_tokens.append(tok)
                token_weight[tok] = token_weight.get(tok, 0.0) + w
    q.tokens = ordered_tokens
    q.token_weights = token_weight
    return q


def build_all_queries(
    lex: LexiconV02,
    lang: str,
    demoted_terms: set[str] | None = None,
    cfg: QueryConfig | None = None,
) -> dict[str, ConceptQuery]:
    out: dict[str, ConceptQuery] = {}
    for concept_id, ct in lex.concepts[lang].items():
        q = build_concept_query(
            concept_id,
            ct.academic,
            ct.craft,
            ct.paraphrases,
            lang,
            demoted_terms=demoted_terms,
            cfg=cfg,
        )
        if q.tokens:
            out[concept_id] = q
    return out
