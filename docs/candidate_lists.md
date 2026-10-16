# Existing candidate lists

These files are a **discovery layer**, not a corpus. The raw files are frozen and must never be edited. All cleaning writes to `data/interim/`.

## Files

### `data/raw/colleague_openlibrary/creative writing books scrabing_open library_english.csv`

| Property | Value |
|---|---|
| Origin | Colleague; OpenLibrary Search API, `subject="creative writing"` |
| Rows | 6,442 records |
| Format | `;`-separated, UTF-8 with BOM. Line 1 is a spurious `Column1;...;Column7`; line 2 holds the real headers |
| Columns (RU → EN) | `Название`=title, `Автор`=author, `Год`=year, `Ссылка`=url, `ISBN`, `Издатель`=publisher, `Язык`=languages |
| Years | 1840-2026; 30 missing; 498 before 1980 (208 before 1970) |
| Language tag `eng` | 5,224 of 6,442 |

Known defects:

- ISBN and publisher are **empty in every row**. The values were never requested from the Search API.
- 1,206 URLs have the form `/works/OL...M`, i.e. an edition ID placed in a work URL.
- No language filter, no relevance filter, no query provenance.
- Deduplicated on (title, author string) only. As a result, "Linda Anderson" and "Anderson, Linda" are counted separately.
- Probably re-exported through a spreadsheet after the script ran, so the script does not fully reproduce the file. Record `postprocessing = unknown` until the colleague confirms.

### `data/raw/colleague_openlibrary/kreatives_schreiben_bücher_open library deutsch.csv`

| Property | Value |
|---|---|
| Origin | Colleague; OpenLibrary free-text searches `kreatives schreiben`, `kreativ schreiben`, `Schreibwerkstatt`, `Schreiben lernen`, `Creative Writing`; filtered to `ger` + broad title markers (`schreib*`, `werkstatt`, `ratgeber`, `erzählen`, ...) |
| Rows | 824 records |
| Format | `,`-separated, UTF-8 with BOM |
| Columns (RU → EN) | `Название`=title, `Подзаголовок`=subtitle (almost always empty), `Автор`=author, `Год`=year, `Языки`=languages, `Ссылка`=url, `ISBN`, `Издатель`=publisher |
| Years | 1601-2026; 1 missing; 18 before 1980 |

Known defects:

- ISBN and publisher are empty.
- 5 URLs have the `/works/OL...M` mismatch.
- The query `Schreiben lernen` combined with the `schreib*` marker pulls in **children's handwriting books**. A colleague diagnostic found 557 such records out of 824, and a quick title-pattern check found 494 or more. Including didactic, therapeutic and technical titles, about 598/824 (73%) are not primary-corpus material.
- It is unknown which of the five queries found each record.
- Recall against the 50 German seed titles is only about 3/50.

### `data/raw/seed_corpus/creative_writing_seed_corpus_100.xlsx`

Manually curated seed bibliography (benchmark and Tier-1 candidates). Each sheet is exported unchanged to CSV (UTF-8) in the same folder:

```bash
cd data/raw/seed_corpus
python -c "import pandas as pd
for n, df in pd.read_excel('creative_writing_seed_corpus_100.xlsx', sheet_name=None, dtype=str).items():
    df.to_csv(f'seed_corpus_100__{n}.csv', index=False, encoding='utf-8')"
```

| Sheet → CSV | Content |
|---|---|
| `Corpus_100` → `seed_corpus_100__Corpus_100.csv` | 100 titles (50 EN, 50 DE), 29 columns |
| `Dashboard` → `seed_corpus_100__Dashboard.csv` | Summary counts (the first row is a title line, not a header) |
| `Codebook` → `seed_corpus_100__Codebook.csv` | Field definitions and verification rules (the first row is a title line) |

Main columns:

- identity: `Corpus_ID`, `Corpus_language`, `Author_editor`, `Title`, `Original_title`
- years and edition: `First_year_corpus_language`, `Selected_edition_year`, `Edition_note`
- identifiers: `Publisher`, `ISBN_13`, `ISBN_10`, `DNB_ID`, `OCLC_ID`
- translation: `Original_language`, `Translation_status`, `Translator`
- type and relevance: `Book_type`, `Priority` (Tier 1-3), `POV_relevance` (High/Medium/Low), `POV_notes`
- access: `Full_text_status`, `Frankfurt_DNB_status`, `Rights_TDM_preliminary`, source and access URLs
- workflow: `Verification_status`, `Notes`, `Next_action`

Profile and gaps:

- 20 of the titles are translations.
- Verification: 69 verified, 31 partial.
- Missing identifiers and links: `DNB_ID` for 99/100, `OCLC_ID` for 89/100, `Access_source_URL` for 97/100. `First_year_corpus_language` is missing for 6.
- `Book_type` has 39 free-text values, which need mapping to `corpus_type` A-F.
- Skewed to recent titles: 1925-2022, with only 3 before 1970. The historical seeds (Besant 1884, James 1884, Lubbock 1921, Forster 1927, Brande 1934) are missing.
- Full-text status is unchecked for 99 titles.
- `POV_relevance` was assigned by the curators. The seed therefore belongs to the benchmark and to dataset B, **not** to prevalence estimates.

### `data/raw/colleague_openlibrary/scripts/`

Provenance scrapers (MiniMax), kept next to the frozen CSVs they produced:

- `openlibrary_scrape_en.py` — subject `creative writing`
- `openlibrary_scrape_de.py` — five German queries + `ger` / title-hint filters

The on-disk CSVs may still differ slightly from a fresh re-run (spreadsheet re-export; `postprocessing = unknown`).

**For new Open Library harvests**, use [`src/catalogue/openlibrary.py`](../src/catalogue/openlibrary.py) instead (rich `fields=`, `source_query`, craft queries from `config/catalogue_queries.yml`, inclusion-rule post-screen, optional editions enrich). Compare a run to the colleague lists with `python -m src.catalogue.compare_ol_to_colleague`.

### `data/interim/filtered_openlibrary/`

Manually cleaned subsets of the colleague scrapes (UTF-8, `;`-separated):

| File | Rows | Notes |
|---|---:|---|
| `filtered_english_corp.csv` | 1,086 | All OLIDs occur in the raw English scrape |
| `filtered_german_corp.csv` | 80 | All OLIDs occur in the raw German scrape |

Columns: `title`, `author`, `language`/`year` (order differs EN vs DE), `link`. No ISBN. These feed `python -m src.candidates.build_final_lists` together with the seed list.

### Final lists

See [`data/interim/README.md`](../data/interim/README.md). Colleague review: `data/interim/final_lists/review_{en,de}.csv`. Search feeds: `final_list_{en,de}.csv` (P0–P2 true-union). Full audit: `union_all.csv`. Rebuild with `python -m src.candidates.build_final_lists`. Broader auto-prescreen lists still live under `data/interim/catalogue/prescreen_*.csv` as provenance.

### Not yet in the repository

- `creative writing cue words (1).xlsx` (44 full / 14 reduced cues). It feeds `config/pov_lexicon.yml` v2.
- The colleague's curated 27-item German bibliography (`Literaturliste_Kreatives_Schreiben`) and its video list.
- The ~30-book English POV set from the earlier project.

When they arrive, put them under `data/raw/<name>/`, and add a section here.

---

## Cleaning plan

Output: `data/interim/catalogue/openlibrary_candidates.csv` and `data/interim/catalogue/seed_candidates.csv`, both in the master schema (`src/common/schema.py: MASTER_COLUMNS`).

1. **Read correctly.**
   - English: `sep=";"`, `skiprows=1`, `encoding="utf-8-sig"`.
   - German: `sep=","`, `encoding="utf-8-sig"`.
   - Read everything as `str`.
2. **Rename columns** from Russian to English (mapping as in the tables above).
3. **OpenLibrary IDs.**
   - Extract `OL\d+[WM]` from the URL. IDs ending in `W` become `ol_work_id`, IDs ending in `M` become `ol_edition_id`.
   - Repair `/works/OL...M` URLs to `/books/OL...M`.
   - Later, resolve each edition to its parent work.
4. **Normalize fields.**
   - Year: integer, out-of-range values set to NA; the original value is kept.
   - Language: split the list, add the primary language and a `multi_language` flag.
   - Add normalized title and author using `src/common/normalize.py`.
5. **Provenance.**
   - Columns: `candidate_source = colleague_openlibrary_en|de`, `source_query` (EN: `subject:creative writing`; DE: `unknown_of_5`), `source_script = scripts/openlibrary_scrape_{en,de}.py`, `postprocessing = unknown`, `retrieval_date = unknown`, `candidate_id`.
6. **Rule-based pre-screen** (patterns in `config/inclusion_rules.yml`). **Nothing is deleted.**
   - `likely_exclude`: handwriting, alphabet or number books; technical, academic, business or copywriting titles.
   - `likely_contextual`: Schreibdidaktik, Aufsatz, therapy, coaching.
   - `likely_primary`: fiction, novel, story or craft markers.
   - `uncertain`: everything else.
   - Record the matching rule in `screen_rule`.
7. **Benchmark recall.**
   - Match the seed corpus against the candidates on normalized base title (before the colon) and author surname.
   - Report hits per language.
   - List the seed titles each source misses. They become targets for DNB, HeBIS and KVK.
8. **Weak deduplication only within each source**, flagged rather than collapsed. Final work/edition deduplication happens after all catalogues are merged.
9. **Manual check** of a random sample of 100 records per pre-screen class, to estimate the error rate of each rule.

## Enrichment needed before a book can be downloaded

Run this only on the screened subset (`likely_primary` + `uncertain`), not on all 7,266 rows. OpenLibrary asks for low-volume API use; for large sets use the monthly data dumps.

| Step | Source | Fields obtained |
|---|---|---|
| a | OpenLibrary `/works/{id}.json`, `/works/{id}/editions.json`, `/books/{id}.json` | edition list, ISBN-10/13, publisher, publish date, pages, subjects, description, `ocaid` (Internet Archive ID), languages, translation info |
| b | OpenLibrary Search with `fields=key,isbn,publisher,ia,ebook_access,subject,language` | `ebook_access` (`public` / `borrowable` / `printdisabled` / `no_ebook`), IA identifiers |
| c | DNB SRU `isb=` / `tit=` + `per=` | `DNB_ID`, URN, subjects (GND), DDC, original title, translator, free digital object |
| d | HeBIS SRU (ISBN) | Hessian/Frankfurt holdings, local call numbers |
| e | WorldCat (if credentials) / KVK (manual) | `OCLC`, international holdings |
| f | Rights check | `rights_TDM_status`: `public_domain` / `open_license` / `TDM_60d_library` / `licensed_ebook` / `print_only` |

A record counts as **ready for acquisition** when it has:

- `edition_id` (our ID) with the selected edition: year, publisher, ISBN (or IA/URN for old books)
- `corpus_type` and `eligibility_status = include`
- `fulltext_source` (IA/OL, Gutenberg, HathiTrust, DNB digital object, Goethe UB licence, physical copy) and `format`
- `rights_TDM_status`
- `physical_holding` / `frankfurt_holding` if no digital copy exists

## Next action

Sample the pre-screen classes (step 9). Run full DNB seed-gap pass for all DE misses. Retry HeBIS from a network that can reach `sru.hebis.de`. Expand the improved Open Library harvest beyond the smoke pages. Run access checks on `final_list_*.csv` when ready.
