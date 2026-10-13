# Catalogue harvesting (research plan weeks 5–7)

Search library catalogues and Open Library. Each hit keeps `source`, `source_query`, and `retrieval_date`.

| Script | What it does |
|---|---|
| `dnb.py` | German National Library (DNB) SRU search → CSV + raw XML |
| `dnb_seed_gaps.py` | Look up seed titles that Open Library missed, via DNB |
| `hebis.py` | Hessian union catalogue (HeBIS) SRU — Frankfurt holdings |
| `hebis_seed_holdings.py` | Check seed ISBNs / titles in HeBIS |
| `openlibrary.py` | Improved Open Library search (ISBN, subjects, provenance) |
| `compare_ol_to_colleague.py` | Compare new OL harvest to colleague lists + seed recall |
| `worldcat.py` | Stub only (needs institutional OCLC credentials) |

## Example commands

```bash
python -m src.catalogue.dnb --query 'tit="kreatives schreiben"' --max-pages 1
python -m src.catalogue.dnb_seed_gaps --lang de --max-pages 1 --limit 5
python -m src.catalogue.dnb_seed_gaps --lang de --loose --ids DE-013,DE-015 --merge-existing
python -m src.catalogue.hebis --isbn 9780226616551 --max-pages 1
python -m src.catalogue.openlibrary --lang de --max-pages 1
```

Query phrases: `config/catalogue_queries.yml`. Run logs: `logs/`.
