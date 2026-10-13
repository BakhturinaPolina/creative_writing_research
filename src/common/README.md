# Shared helpers (used by every phase)

Small utilities that other folders import so we do not copy the same code.

| File | What it does |
|---|---|
| `paths.py` | Repo root and `data/` / `config/` / `logs/` paths |
| `logging_util.py` | Timestamped prints to the terminal **and** `logs/*.log` |
| `schema.py` | Column lists for catalogue hits and the master bibliography |
| `normalize.py` | Title / author / ISBN cleaning for matching |
| `prescreen.py` | Rule-based title screen from `config/inclusion_rules.yml` |

You normally do not run anything here directly. Other scripts import these modules.
