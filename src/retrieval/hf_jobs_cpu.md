# CPU Hugging Face Jobs pathway (fallback)

Use when local RAM cannot run the streaming lexical pipeline safely.

## Requirements

- Private storage / Space for restricted `data/interim/fulltext/json/{en,de}/`
- CPU-only Job (no GPU)
- Repo checkout with `config/`, `src/retrieval/`, `data/processed/{en,de}_book_stats.csv`

## Suggested command

```bash
pip install -r requirements.txt pytest
python -m pytest tests/retrieval -q
python -m src.retrieval.run_all --lang all --with-passages --top-k 500
```

Upload only pointer CSVs and aggregate reports publicly. Keep `*_passages.csv` private.
