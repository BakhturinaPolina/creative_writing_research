# Creative-writing corpus history

Two outputs:

1. An evidence-based argument for the corpus temporal range.
2. A master bibliographic table from which the corpus is selected.

Google Ngrams show when terminology becomes culturally visible. Library catalogues show when relevant books actually appear. Both are needed.

This is the corpus-history part of a larger project on how narrative perspective (POV) is taught in English and German writing manuals.

## Documentation

- [docs/scientific_readme.md](docs/scientific_readme.md): research questions, corpus definition, methods
- [docs/research_plan.md](docs/research_plan.md): 24-week plan
- [docs/research_log.md](docs/research_log.md): progress log and decision register
- [docs/candidate_lists.md](docs/candidate_lists.md): existing candidate lists (colleague OpenLibrary data, seed corpus) and their cleaning plan

## Setup

```bash
python3 -m venv .venv          # if ensurepip is missing: python3 -m venv --without-pip .venv, then bootstrap pip with get-pip.py
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # only needed for WorldCat
```

`requirements.lock.txt` records the exact versions last used.

## Layout

- `config/`: Ngram terms, catalogue queries, inclusion rules, bilingual POV lexicon (`pov_lexicon.yml`).
- `src/`: code grouped by research-plan phase (each folder has a short README):
  - `common/` — paths, logging, schema, normalize, prescreen
  - `candidates/` — migrate colleague lists + build final candidate lists (weeks 3–4)
  - `ngrams/` — Google Books Ngram figures (week 5)
  - `catalogue/` — DNB, HeBIS, Open Library harvest (weeks 5–7)
  - `merge_enrich/`, `publication/`, `access/`, `fulltext/`, `retrieval/`, `annotation/`, `analysis/` — later phases (stubs / README only for now)
- `logs/`: timestamped run logs from every CLI (files are gitignored; folder is kept).
- `data/raw/`: untouched inputs and API responses. API responses are not tracked by git; `colleague_openlibrary/` (including `scripts/`) and `seed_corpus/` are tracked because they cannot be re-fetched.
- `docs/`: protocol, plan, log.
- `data/interim/`: working tables — see [`data/interim/README.md`](data/interim/README.md).
- `data/processed/`: final analysis tables.
- `notebooks/`, `figures/`: exploration and plots.

## Run order

```bash
python -m src.ngrams.ngrams --group english_corpus_terms --corpus en
python -m src.ngrams.ngrams --group german_corpus_terms --corpus de
python -m src.catalogue.dnb --query 'tit="kreatives schreiben"' --max-pages 2
python -m src.candidates.migrate_candidates          # → data/interim/catalogue/
python -m src.candidates.build_final_lists           # → data/interim/final_lists/ (union + review)
python -m src.catalogue.openlibrary --lang de --max-pages 1
python -m src.catalogue.hebis --isbn 9780226616551 --max-pages 1
```

Colleagues reviewing candidates: open `data/interim/final_lists/review_{en,de}.csv` (see [`data/interim/README.md`](data/interim/README.md)).

Status: `ngrams`, `dnb`, `hebis`, `openlibrary`, `migrate_candidates`, `build_final_lists` and `backfill_isbn` work. `worldcat`, `deduplicate` and `publication_stats` are stubs.
