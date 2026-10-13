"""Rule-based title/subtitle pre-screen from config/inclusion_rules.yml.

Nothing is deleted: each record gets likely_primary / contextual / exclude / uncertain.
Shared by the candidate migrator and the Open Library harvester.
"""
import re
from pathlib import Path

import yaml

from src.common.paths import CONFIG

RULES_PATH = CONFIG / "inclusion_rules.yml"
SCREEN_ORDER = ("likely_exclude", "likely_contextual", "likely_primary")


def load_screen_rules(path: Path | None = None):
    """Load regex patterns from YAML and compile them once."""
    path = path or RULES_PATH
    with open(path, encoding="utf-8") as f:
        patterns = yaml.safe_load(f)["prescreen_patterns"]
    compiled = []
    for status in SCREEN_ORDER:
        for group, group_patterns in (patterns.get(status) or {}).items():
            for pattern in group_patterns:
                compiled.append((status, f"{status}.{group}", re.compile(pattern, re.IGNORECASE)))
    return compiled


def prescreen(title, subtitle, rules):
    """Return (eligibility_status, screen_rule_id) for one title+subtitle."""
    text = f"{title or ''} {subtitle or ''}".strip()
    for status, rule_id, pattern in rules:
        if pattern.search(text):
            return status, rule_id
    return "uncertain", ""
