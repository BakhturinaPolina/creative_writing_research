# Narrative perspective in English- and German-language writing manuals

Corpus construction, historical validation and semantic retrieval.

Status: protocol draft v0.1 (2026-10-08). The decisions marked *provisional* are listed in the decision register in [research_log.md](research_log.md).

---

## 1. Research question

> How has narrative perspective (point of view) been conceptualized, taught and operationalized in English- and German-language writing manuals over time?

| RQ | Question | Main measure |
|---|---|---|
| RQ1 | What proportion of writing manuals discuss narrative perspective substantially? | book-level prevalence (`pov_any`, `pov_dedicated_chapter`) |
| RQ2 | How much textual space is devoted to it? | `pov_share_of_book`, POV passages per 10,000 words |
| RQ3 | Which concepts make up POV discourse (person, narrator knowledge, focalization, distance, voice, switching, ...)? | dimension labels per passage |
| RQ4 | How does that conceptual structure change over time? | dimension distributions by decade |
| RQ5 | How do English and German manuals differ? | the same measures by language, controlling for year, type and length |
| RQ6 | To what extent is German craft discourse mediated through translations of Anglophone manuals? | German originals vs translations, and when concepts first appear in each |

The project is **not** about collecting as many manuals as possible. It is about a reproducibly defined population, with known coverage, from which prevalence claims can be made.

---

## 2. Corpus definition

**Primary corpus:** books whose principal purpose is to teach, advise on, or systematically discuss the production of literary narrative prose.

Not in the primary corpus (they can serve as comparison corpora): academic writing, copywriting, journalism, grammar/style guides, screenwriting, therapeutic writing, school composition, and theoretical narratology.

German needs particular care. *Kreatives Schreiben* covers school pedagogy, self-expression, therapy and literary authorship. Catalogue searches therefore retrieve broadly, and classification happens afterwards.

### 2.1 Corpus type (`corpus_type`)

| Code | Type | Use |
|---|---|---|
| A | General creative-writing / fiction-writing manual | main |
| B | Novel-writing handbook | main |
| C | Specialized narrative-craft book (e.g. POV, scene, dialogue) | main |
| D | Writer memoir with substantial craft instruction | main, flagged |
| E | Pedagogical creative-writing handbook | robustness / comparison |
| F | Narratology / literary theory | comparison only |

### 2.2 Three nested datasets

The population corpus has **no POV requirement**. Selecting books because they discuss POV and then asking how often books discuss POV would inflate the result. The earlier project's criterion ("chapters on narrative perspective, voice, or mode") is therefore dropped for dataset A.

| Dataset | Purpose | POV requirement |
|---|---|---|
| A. Population corpus | Main quantitative estimates (RQ1-RQ6) | none |
| B. POV development corpus | Vocabulary, embeddings, annotation taxonomy, classifier training | known POV material deliberately overrepresented (e.g. the ~30-book English set from the earlier project: 1980-2016, ~6,008 pages, ~391 POV pages, 29 POV labels) |
| C. Contextual / comparison corpus | Didactics, therapy, academic commentary, articles, videos | not used in prevalence statistics |

Online videos (Sanderson, Reedsy, Abbie Emmons, Shaelin Writes, ...) go into a separate `corpus_audio_video` table. They are used for discovering terminology (`deep POV`, `head hopping`, `close third`).

### 2.3 Works and editions

- **Edition** is a specific publication (ISBN, year, publisher). It matters for acquisition and for counting market activity.
- **Work** is the conceptual book across editions. It matters for cross-sectional statistics and intellectual history.

For example, Burroway's *Writing Fiction* (1982, 10th ed. 2019) is one work with many editions. Main statistics count works. Edition change is a separate research object.

### 2.4 Translations

The fields recorded are `original_language`, `original_title`, `original_year`, `translation_year`, `translator` and `edition`. German translations of Anglophone manuals are treated as data, not noise. They are the basis for RQ6.

---

## 3. Periodization (provisional)

| Period | Scope | Rationale |
|---|---|---|
| **1970-2025** | main, directly comparable EN-DE corpus | creative writing is recognizable as a category in both traditions |
| **1880-1969** | historical extension | English handbook period (1880s-1930s), Lubbock (1921), the Iowa workshop (1936); German pre-history under other labels |

The facts behind these choices:

- **English:**
  - Literary-advice research describes a handbook period from the 1880s to the 1930s, then the postwar "program era", then an expansion of advice authors in the 1980s-90s ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-53614-5_1)).
  - Iowa has taught creative writing since 1897 and the Writers' Workshop dates from 1936 ([Univ. of Iowa](https://writing.uiowa.edu/our-history)).
  - POV is systematized in Lubbock's *The Craft of Fiction* (1921) ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1002/9781444337839.wbelctv1p004)).
- **German:**
  - *Kreatives Schreiben* arrives as a term and movement in the 1970s ([Springer](https://link.springer.com/article/10.1007/s11620-016-0324-2)).
  - *Kreatives Schreiben an Hochschulen* appears in 1988 ([De Gruyter](https://www.degruyterbrill.com/document/doi/10.1515/9783110935837/html?lang=en)).
  - Older German searches use `Schriftstellerei`, `Erzählkunst`, `Dichtkunst`, `Technik des Romans`, `Technik des Erzählens`, `Erzähltechnik` and `literarisches Schreiben`.

The cutoff (1970, 1980 or another year) is decided **after** the Ngram and catalogue publication-frequency analysis. The old 1980 criterion is not inherited automatically.

---

## 4. Evidence for the temporal range

> Google Ngrams show when terminology becomes culturally visible. Bibliographic catalogues show when relevant books actually appear.

| Evidence | What it measures | Use |
|---|---|---|
| Google Ngram | frequency of terms across digitized books | linguistic / cultural emergence |
| New works per year | first appearance of distinct manuals | growth of the field |
| Editions per year | (re)publication activity | market / institutionalization |
| Translations per year | transfer of Anglophone craft discourse | EN to DE influence |
| Books with POV chapters | importance of perspective within the genre | substantive question |

Notes on Ngrams:

- The Ngram data covers up to 2022 (dataset update July 2024). Ngrams that occur in fewer than 40 books are excluded, and charts can change between releases ([Google Books Ngram info](https://books.google.com/ngrams/info)).
- Requests always use `smoothing=0`. Smoothing is applied locally. Every response is cached with its provenance.
- The 2019 corpus is used as a robustness check.
- Ngrams validate the periodization. They do **not** determine corpus membership.

The target statement:

> Lexical evidence from Google Books indicates the historical emergence of the terminology, while bibliographic records show when dedicated instructional publications became numerous enough for systematic corpus analysis.

---

## 5. Discovery channels

1. **DNB SRU** (German). Its metadata is CC0, it has an SRU interface, and its SRU Query Tool handles up to 100,000 records per query ([DNBLab blog](https://blog.dnb.de/neu-im-dnblab-tool-zur-sru-schnittstellenabfrage-und-mehr/)). Its legal collection remit covers works published in Germany since 1913 and German-language works published abroad.
2. **HeBIS SRU** (Hessian union catalogue). It is free, needs no authentication, returns MARC-XML or PICA+, and gives up to 500 records per request ([HeBIS SRU](https://www.hebis.de/uploads/2020/10/SRU.pdf)). It also shows Frankfurt holdings.
3. **WorldCat Search API v2**, optional. It uses OAuth 2.0 and needs institutional credentials ([OCLC](https://www.oclc.org/developer/api/oclc-apis/worldcat-search-api.en.html)). **KVK** is used only for manual gap-finding and validation, because it is a meta-search with no database of its own ([KIT](https://www.bibliothek.kit.edu/english/kvk-help.php)).
4. **Library of Congress / British Library** for English coverage.
5. **Google Books, Internet Archive, HathiTrust, Project Gutenberg and repositories** for discovery, tables of contents and legally accessible full texts.
6. **Citation chaining:** bibliographies of major manuals and of histories of literary advice. This finds material that terminology-based searches miss.

Existing inputs: the colleague's OpenLibrary scrape (EN 6,442 / DE 824 records) and the 100-title seed bibliography. See [candidate_lists.md](candidate_lists.md).

Every catalogue hit keeps `source`, `source_query` and `retrieval_date`. The query-hit table is kept separately from the deduplicated book table, so that the productivity of each query can be measured.

---

## 6. Gold benchmark

The benchmark has about 50 EN and 50 DE unquestionably relevant titles (the seed corpus), plus the useful items from the colleague's curated German list and the earlier English POV set. Every discovery source is scored by **recall against the benchmark**.

For example, the OpenLibrary scrape finds about 29/50 of the EN seeds but only about 3/50 of the DE seeds. This is why DNB and HeBIS are needed.

---

## 7. Corpus size targets

| Stage | English | German |
|---|---:|---:|
| Pilot | 75-100 | 75-100 |
| Core | 400-600 | 250-400 |
| Strong final | 800-1,200 | 500-800 |
| Stretch | all accessible qualifying works (1,500-2,500+ total) | |

The two languages are **not** forced to the same size. Comparisons use proportions, density per 10,000 words, the share of each book, and probabilities, plus regression with language, year, type and length as predictors.

---

## 8. Full-text access

- §60d UrhG permits reproductions for scientific text and data mining (TDM) by research organizations, libraries and non-commercial researchers, and permits secure retention of the copies ([§60d UrhG](https://www.gesetze-im-internet.de/urhg/__60d.html)).
- **Goethe-Universität UB, Digital Humanities** (`dighum@ub.uni-frankfurt.de`) supports DH resources and text analysis ([UB Frankfurt](https://www.ub.uni-frankfurt.de/digitalhumanities/)).
- **DNB / DNBLab and DH-Call** provide access to copyright-protected material for research projects ([ABI Technik](https://www.degruyterbrill.com/document/doi/10.1515/abitech-2025-0047/html?lang=en)).
- Request framing: research access to a defined corpus for non-commercial TDM under §60d UrhG, preferably as PDF/OCR/plain text or inside a controlled computational environment, with no redistribution of the texts. The request includes the research question, affiliation, title count, formats and security arrangements.

---

## 9. Passage-level analysis pipeline

```text
PDF / EPUB / OCR
  -> text normalization (headers, page numbers, OCR artifacts)
  -> segmentation (book > chapter > section > paragraph / 2-3 sentence window)
  -> high-recall lexical retrieval (config/pov_lexicon.yml)
  -> semantic retrieval (multilingual sentence embeddings vs. prototypes)
  -> union of candidates
  -> human-annotated gold sample
  -> supervised classifier
  -> validation (precision / recall / F1; target recall >= .90 on substantive POV passages)
  -> full-corpus inference
  -> book- and decade-level measures
  -> qualitative close reading
```

**Retrieval rule:** a passage becomes a candidate if it has one high-precision cue, OR two or more supporting cues in the same window, OR high semantic similarity to a POV prototype. A contextual cue alone (`voice`, `mode`, `distance`, `perspective`) never counts.

Recall is prioritized over precision. Systematically missing historical terminology would bias the diachronic results.

Methodological principle (carried over from the earlier romance-novel project): categories are defined by **narrative function**, not by vocabulary alone.

### 9.1 POV taxonomy

POV is treated as **combinations of dimensions**, not as 29 flat classes:

- general POV
- person
- narrator role/status
- knowledge/focalization
- consciousness access
- distance/depth
- objective/dramatic mode
- involvement/voice
- multiplicity/switching
- reliability
- thought representation
- information control

For example, "first-person peripheral" is person = first combined with role = witness. "Deep third" is person = third combined with distance = close and access = internal. The full bilingual lexicon is in `config/pov_lexicon.yml`.

---

## 10. Annotation scheme

**Relevance:**

- 0 = unrelated
- 1 = mentions perspective but does not teach or discuss it
- 2 = substantive discussion
- 3 = perspective is the central subject

**Context (multi-label):**

- definition/terminology
- choice of narrator
- first vs third person
- omniscience/limited knowledge
- focalization
- distance/interiority
- voice
- consistency and POV switching
- multiple perspectives
- unreliable narration
- reader identification/empathy
- showing/telling
- scene construction
- exercise/example
- prescriptive rule
- exception/criticism of a rule

**Communicative function:**

- definition
- recommendation
- warning
- example
- exercise
- comparison
- prohibition

Target: 800-1,500 annotated passages, with a double-annotated subset for inter-annotator agreement.

---

## 11. Book-level measures

```text
pov_any, pov_passages_n, pov_words, pov_share_of_book, pov_dedicated_chapter,
pov_first_person, pov_third_person, pov_omniscience, pov_focalization,
pov_distance, pov_switching, pov_reader_effect, pov_prescriptive_language
```

These are aggregated by year, decade, language, country, original vs translation, book type, publisher, edition and genre specialization. Raw counts are never the main outcome: a 600-page textbook naturally has more hits than a 120-page handbook.

---

## 12. Known limitations

- Catalogue coverage is uneven (English depends on OpenLibrary, LoC, BL and WorldCat access).
- Ngram corpora change between releases, and their OCR and language detection are imperfect.
- Access to full texts will not be random: recent copyrighted books may be under-represented. `fulltext_status` and `rights_TDM_status` are recorded so this bias can be reported.
- The German category boundary (pedagogy and therapy vs literary craft) requires judgement. Exclusion reasons are recorded explicitly.
- Some inputs have unknown provenance. The colleague's English CSV was likely re-exported from a spreadsheet after the script ran.

---

## 13. Repository map

- `config/`: query vocabularies, inclusion rules, POV lexicon
- `src/`: code by research-plan phase (`common`, `candidates`, `ngrams`, `catalogue`, …); each folder has a short README
- `logs/`: timestamped CLI run logs
- `data/raw/`: untouched inputs and API responses
- `data/interim/`: working tables (see `data/interim/README.md`); `data/processed/`: final tables
- `docs/`: this protocol, [research_plan.md](research_plan.md), [research_log.md](research_log.md), [candidate_lists.md](candidate_lists.md)
