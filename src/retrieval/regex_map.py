"""Load explicit regex → concept mapping."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.common.paths import CONFIG

MAP_PATH = CONFIG / "regex_concept_map_v01.yml"

ROLE_CONCEPT_SPECIFIC = "concept_specific"
ROLE_CONTEXT_ONLY = "context_only"
ROLE_SUPPORTING = "supporting_insufficient"
ROLE_AMBIGUOUS = "ambiguous_unrelated"


@dataclass
class RegexMapping:
    name: str
    role: str
    concepts: list[str] = field(default_factory=list)
    concept_filters: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""

    def applies_to(self, concept_id: str, matched_text: str = "") -> bool:
        if self.concept_filters:
            needles = self.concept_filters.get(concept_id)
            if needles is None:
                return False
            mt = matched_text.casefold()
            return any(n.casefold() in mt for n in needles)
        return concept_id in self.concepts


@dataclass
class RegexConceptMap:
    version: str
    path: Path
    languages: dict[str, dict[str, RegexMapping]]

    def get(self, lang: str, name: str) -> RegexMapping | None:
        return self.languages.get(lang, {}).get(name)

    def names_for_concept(self, lang: str, concept_id: str) -> list[str]:
        out = []
        for name, m in self.languages.get(lang, {}).items():
            if m.concept_filters:
                if concept_id in m.concept_filters:
                    out.append(name)
            elif concept_id in m.concepts:
                out.append(name)
        return out


def load_regex_map(path: Path | None = None) -> RegexConceptMap:
    path = path or MAP_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    languages: dict[str, dict[str, RegexMapping]] = {}
    for lang, block in (raw.get("languages") or {}).items():
        languages[lang] = {}
        for name, spec in (block or {}).items():
            languages[lang][name] = RegexMapping(
                name=name,
                role=str(spec.get("role") or ROLE_AMBIGUOUS),
                concepts=list(spec.get("concepts") or []),
                concept_filters=dict(spec.get("concept_filters") or {}),
                notes=str(spec.get("notes") or ""),
            )
    return RegexConceptMap(
        version=str(raw.get("version", "")),
        path=path,
        languages=languages,
    )
