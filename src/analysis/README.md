# Classifier and analysis (research plan weeks 19–24)

Train a POV passage classifier, run it on the full accessible corpus, and compute book- and decade-level measures for the research questions.

So far this folder holds descriptive book-level tables for English and German. The classifier and the measures in `docs/scientific_readme.md` sections 9–11 are still to come.

```bash
python -m src.analysis.corpus_stats                # English book-level table
python -m src.analysis.corpus_stats --lang de      # German book-level table
python -m src.analysis.corpus_stats --selftest     # check the POV matcher on made-up sentences
python -m src.analysis.corpus_stats --spotcheck 40 # print 40 random matches in context (terminal only, not saved)
```

- `corpus_stats.py`: one row per `primary` / `comparison` file with quality flags, `in_analysis_set`, `primary_group` (fiction_manual / pedagogy_genre_craft / contrast), and exploratory POV term counts.
- `plots.py`: shared figure style and one function per figure (English notebook).
- Notebook: [`notebooks/04_corpus_descriptive_stats_en.ipynb`](../../notebooks/04_corpus_descriptive_stats_en.ipynb). It reads the English table and the manifest.
- Bilingual retrieval exploration (v1, preserved): [`notebooks/05_narrative_perspective_exploration.ipynb`](../../notebooks/05_narrative_perspective_exploration.ipynb) + `reports/01_*`.
- Full analysis v2: [`notebooks/06_narrative_perspective_full_analysis.ipynb`](../../notebooks/06_narrative_perspective_full_analysis.ipynb), interactive HTML [`reports/02_narrative_perspective_interactive.html`](../../reports/02_narrative_perspective_interactive.html), tables/figures under `reports/tables/02_*` and `reports/figures/02_narrative_perspective/`.
- Regenerate v2:

```bash
python -m src.analysis.inventory
python -m src.analysis.extreme_passages
python -m src.analysis.run_np_exploration --edition v2
python -m src.analysis.build_interactive_report
```

Helpers: `narrative_perspective.py`, `analysis_v2.py`, `inventory.py`, `extreme_passages.py`. Methodology: [`docs/analysis_methodology_v2.md`](../../docs/analysis_methodology_v2.md). Terminology review: [`docs/terminology_review_guide.md`](../../docs/terminology_review_guide.md).

## Outputs

None of these contain book text, so all of them may be committed.

| File | Content |
|---|---|
| `data/processed/en_book_stats.csv` | English primary/comparison: flags, word counts, POV terms, `primary_group` |
| `data/processed/en_book_stats.meta.json` | build time, commit, checksums, analysis-set and group counts |
| `data/processed/de_book_stats.csv` | German counterpart |
| `data/processed/de_book_stats.meta.json` | German build metadata |
| `figures/corpus_stats/fig*.{png,pdf}` | the English notebook's figures, for slides and print |
| `notebooks/exports/04_corpus_descriptive_stats_en.html` | the notebook without code, for readers who do not run Python |

## Analysis set

A file enters the figures unless one of these applies, checked in this order:

1. `needs_ocr`: scanned PDF without text.
2. `short_text`: fewer than 10,000 words.
3. `role_conflict`: the same work (same title and author surname, or `content_dup_of`) sits in both `primary` and `comparison`.
4. `surplus_duplicate`: another file of the same work is longer.

Nothing is deleted. `exclusion_step` names the reason, `in_analysis_set` is the result.

`identity_check` does not exclude a book, but its catalogue year is not used:

- `renamed`: `config/book_metadata_overrides.csv` marks the title as changed (`identity_changed`).
- `unresolved`: an open `wrong_book` suggestion in `title_author_review.csv`, or `title_match=False`.

## POV term counts (exploratory)

- Terms: language-specific `high` terms and `regex` patterns of each dimension in `config/pov_lexicon.yml` (`--lang en|de`).
- Body chapters only. Spaces and hyphens are interchangeable, case is ignored; English allows an optional plural -s.
- `pov_<dimension>_n` counts per dimension. `pov_n` merges overlapping matches across dimensions. `pov_per_10k` = `pov_n` per 10,000 body words.
- Nothing is counted per paragraph: paragraphs are about ten times longer in PDFs than in e-books.

`primary_group` splits primary for robustness (not a second corpus): `fiction_manual`,
`pedagogy_genre_craft`, or `contrast` (the comparison books). See `robustness_primary_split()`
and `plots.plot_pov_density_robustness`. German school/therapy/genre title cues are included
alongside the English ones.

Known limits: the counts are not validated against annotation; "point of view" also matches
the everyday sense ("opinion"). Since 2026-11-12 the instruction signal no longer counts POV
words. Primary is the broad narrative-writing set (see `src/fulltext/README.md`); comparison
is narratology + academic writing about writing only.
