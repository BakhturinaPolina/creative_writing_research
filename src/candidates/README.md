# Candidates (research plan weeks 3–4)

Turn frozen colleague Open Library lists and the seed bibliography into clean working tables.

| Script | What it does |
|---|---|
| `migrate_candidates.py` | Repair headers/IDs, pre-screen, measure seed recall, write `data/interim/catalogue/` |
| `build_final_lists.py` | Merge filtered OL + seed into `data/interim/final_lists/` |

## Example commands

```bash
python -m src.candidates.migrate_candidates
python -m src.candidates.build_final_lists
```

Raw inputs stay under `data/raw/`. Nothing here downloads full texts.
