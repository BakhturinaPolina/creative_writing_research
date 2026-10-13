# Ngrams (research plan week 5)

Google Books Ngram figures for English and German corpus / POV terms.

They help decide the historical period (DEC-002). They do **not** decide which books belong in the corpus.

| Script | What it does |
|---|---|
| `ngrams.py` | Fetch series, cache raw JSON, write processed CSVs and figures |

## Example command

```bash
python -m src.ngrams.ngrams --group english_corpus_terms --corpus en
```

Terms live in `config/ngram_terms.yml`.
