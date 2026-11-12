# Classifier and analysis (research plan weeks 19–24)

Train a POV passage classifier, run it on the full accessible corpus, and compute book- and decade-level measures for the research questions.

So far this folder holds a first descriptive pass over the English books. The classifier and the measures in `docs/scientific_readme.md` sections 9–11 are still to come.

```bash
python -m src.analysis.corpus_stats                # book-level table (needs the full texts, about 2 minutes)
python -m src.analysis.corpus_stats --selftest     # check the POV matcher on made-up sentences
python -m src.analysis.corpus_stats --spotcheck 40 # print 40 random matches in context (terminal only, not saved)
```

- `corpus_stats.py`: one row per English `primary` / `comparison` file with quality flags, the analysis-set flag and exploratory POV term counts. Also the small tables the notebook shows (flow, key figures, sensitivity).
- `plots.py`: shared figure style and one function per figure.
- Notebook: [`notebooks/04_corpus_descriptive_stats_en.ipynb`](../../notebooks/04_corpus_descriptive_stats_en.ipynb). It reads only the table below and the manifest, so it runs without the book files.

## Outputs

None of these contain book text, so all of them may be committed.

| File | Content |
|---|---|
| `data/processed/en_book_stats.csv` | one row per file: catalogue data, flags, word counts, POV term counts |
| `data/processed/en_book_stats.meta.json` | build time, commit, manifest and lexicon checksums, row counts |
| `figures/corpus_stats/fig*.{png,pdf}` | the notebook's figures, for slides and print |
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

- Terms: English `high` terms and `regex` patterns of each dimension in `config/pov_lexicon.yml`.
- Body chapters only. Spaces and hyphens are interchangeable, case is ignored, a plural -s is allowed.
- `pov_<dimension>_n` counts per dimension. `pov_n` merges overlapping matches across dimensions. `pov_per_10k` = `pov_n` per 10,000 body words.
- Nothing is counted per paragraph: paragraphs are about ten times longer in PDFs than in e-books.

`primary_group` splits primary for robustness (not a second corpus): `fiction_manual`,
`pedagogy_genre_craft`, or `contrast` (the comparison books). See `robustness_primary_split()`
and `plots.plot_pov_density_robustness`.

Known limits: the counts are not validated against annotation; "point of view" also matches
the everyday sense ("opinion"). Since 2026-11-12 the instruction signal no longer counts POV
words. Primary is the broad narrative-writing set (see `src/fulltext/README.md`); comparison
is narratology + academic writing only.
