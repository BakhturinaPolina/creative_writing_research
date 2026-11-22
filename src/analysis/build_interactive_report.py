"""Build standalone offline HTML research report with bilingual lexicon explorer.

No CDN dependencies. Plotly JS is inlined from the local package. Suggestions
are stored in browser localStorage until the reviewer exports JSON/CSV.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly
import yaml

from src.analysis.narrative_perspective import LANG_LABEL, load_concept_families, load_concept_ids
from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_INTERIM, DATA_PROCESSED, ROOT

REPORTS = ROOT / "reports"
TABLES = REPORTS / "tables"
OUT = REPORTS / "02_narrative_perspective_interactive.html"
COLORS = {"en": "#3976C6", "de": "#CC8C42", "English": "#3976C6", "German": "#CC8C42"}


def _read_csv(name: str) -> pd.DataFrame:
    path = TABLES / name
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _df_records(df: pd.DataFrame) -> list[dict]:
    if df is None or not len(df):
        return []
    out = df.copy()
    # Replace inf/-inf for JSON
    out = out.replace([float("inf"), float("-inf")], [None, None])
    return json.loads(out.to_json(orient="records", date_format="iso"))


def load_lexicon_payload() -> dict:
    raw = yaml.safe_load((CONFIG / "pov_lexicon_psyper_en_de_v0.3.yml").read_text(encoding="utf-8"))
    fam = load_concept_families()
    fam_map = {}
    fam_labels = {}
    fam_notes = {}
    for block in fam.get("families") or []:
        for cid in block.get("concepts") or []:
            fam_map[cid] = block["id"]
            fam_labels[cid] = block["label"]
            fam_notes[cid] = block.get("ambiguity_notes", "")
    prov_path = DATA_INTERIM / "retrieval" / "lexicon_term_provenance_v05.csv"
    prov = pd.read_csv(prov_path, dtype=str).fillna("") if prov_path.is_file() else pd.DataFrame()

    # PsyPer notes from docs table (operational)
    psyper_doc = {}
    doc = ROOT / "docs" / "psyper_vs_retrieval_concepts_v05.md"
    if doc.is_file():
        for line in doc.read_text(encoding="utf-8").splitlines():
            if line.startswith("| `") and "psyper" not in line.lower():
                parts = [p.strip() for p in line.strip("|").split("|")]
                if len(parts) >= 3 and parts[0].startswith("`"):
                    cid = parts[0].strip("`")
                    psyper_doc[cid] = {"psyper_related_tags": parts[1], "doc_note": parts[2]}

    concepts = []
    vocab = []
    for c in raw.get("concepts") or []:
        cid = c["id"]
        note = (c.get("note") or "").strip()
        definition_status = "from_lexicon_note" if note else "operational_definition_needs_review"
        if not note:
            note = f"Operational retrieval concept `{cid}` (definition pending review)."
        pd_info = psyper_doc.get(cid, {})
        concepts.append(
            {
                "concept_id": cid,
                "concept_label": cid.replace("_", " "),
                "concept_family": fam_map.get(cid, "unassigned"),
                "concept_family_label": fam_labels.get(cid, "Unassigned"),
                "definition": note,
                "definition_status": definition_status,
                "psyper_mapping": c.get("psyper") or [],
                "psyper_related_tags_doc": pd_info.get("psyper_related_tags", ""),
                "family_ambiguity_notes": fam_notes.get(cid, ""),
                "theoretical_context": pd_info.get("doc_note", ""),
            }
        )
        for lang in ("en", "de"):
            block = c.get(lang) or {}
            for term_type, terms in block.items():
                for term in terms or []:
                    # Provenance join
                    prow = prov[
                        (prov["concept_id"] == cid)
                        & (prov["language"] == lang)
                        & (prov["original_term"].astype(str) == str(term))
                    ] if len(prov) else prov
                    pr = prow.iloc[0].to_dict() if len(prow) else {}
                    vocab.append(
                        {
                            "concept_id": cid,
                            "language": lang,
                            "term": term,
                            "term_type": term_type,
                            "evidence_tier": pr.get("evidence_tier", ""),
                            "provenance_type": pr.get("provenance_type", ""),
                            "lexicon_version_introduced": pr.get("lexicon_version_introduced", ""),
                            "corpus_attested": pr.get("corpus_attested", ""),
                            "corpus_attestation_count": pr.get("corpus_attestation_count", ""),
                            "conceptual_ambiguity": pr.get("conceptual_ambiguity", ""),
                            "human_review_status": pr.get("human_review_status", ""),
                            "matching_type": "dictionary_term",
                        }
                    )
    # Regex map (advanced view)
    regex_path = CONFIG / "regex_concept_map_v03.yml"
    regex_entries = []
    if regex_path.is_file():
        rraw = yaml.safe_load(regex_path.read_text(encoding="utf-8"))
        for item in rraw.get("patterns") or rraw.get("regexes") or []:
            if isinstance(item, dict):
                regex_entries.append(item)
        # Alternate structure: concept → patterns
        if not regex_entries and isinstance(rraw, dict):
            for k, v in rraw.items():
                if k in {"version", "lexicon_version", "built", "note"}:
                    continue
                if isinstance(v, list):
                    for pat in v:
                        regex_entries.append({"concept_id": k, "pattern": pat})
                elif isinstance(v, dict):
                    regex_entries.append({"concept_id": k, **v})

    return {
        "lexicon_version": raw.get("version", "0.3-draft"),
        "n_concepts": len(concepts),
        "concepts": concepts,
        "vocabulary": vocab,
        "regex_entries": regex_entries[:500],
        "psyper_disclaimer": (
            "The 36 retrieval concepts are informed by PsyPer-related interests. "
            "They are not identical to the original PsyPer annotation scheme."
        ),
    }


def load_report_data() -> dict:
    meta_path = REPORTS / "02_narrative_perspective_run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    gate_path = DATA_PROCESSED / "extreme_passage_gate_v2.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}

    freq = _read_csv("02_concept_frequencies.csv")
    zero_de = []
    if len(freq):
        zero_de = (
            freq[(freq.language == "de") & (freq.n_assignments == 0)]["concept_id"].tolist()
        )

    return {
        "meta": meta,
        "gate": gate,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "colors": COLORS,
        "tables": {
            "corpus_overview": _df_records(_read_csv("02_corpus_overview.csv")),
            "pipeline_funnel": _df_records(_read_csv("02_pipeline_funnel.csv")),
            "concept_frequencies": _df_records(freq),
            "concept_family_frequencies": _df_records(_read_csv("02_concept_family_frequencies.csv")),
            "concept_dispersion": _df_records(_read_csv("02_concept_dispersion.csv")),
            "retrieval_coverage": _df_records(_read_csv("02_retrieval_coverage.csv")),
            "composition": _df_records(_read_csv("02_corpus_composition_by_group.csv")),
            "standardization_weights": _df_records(_read_csv("02_standardization_weights.csv")),
            "standardized_frequencies": _df_records(_read_csv("02_standardized_frequencies.csv")),
            "complementarity": _df_records(_read_csv("02_retrieval_complementarity.csv")),
            "sensitivity": _df_records(_read_csv("02_sensitivity_analysis.csv")),
            "robustness": _df_records(_read_csv("02_robustness_labels.csv")),
            "bootstrap": _df_records(_read_csv("02_bootstrap_intervals.csv")),
            "historical": _df_records(_read_csv("02_historical_overview.csv")),
            "extreme_passages": _df_records(_read_csv("02_extreme_passages.csv")),
            "cooccurrence_pairs_en": _df_records(_read_csv("02_concept_cooccurrence_pairs_en.csv")),
            "cooccurrence_pairs_de": _df_records(_read_csv("02_concept_cooccurrence_pairs_de.csv")),
        },
        "callouts": {
            "zero_match_concepts_de": zero_de,
            "corpus_size_note": "English analysis set is much larger than German (books and words).",
            "composition_note": "German contrast materials hold a larger word share than English contrast materials.",
            "stance_note": "German narratorial/figural stance frequencies are unusually high relative to English.",
            "focalization_zero_note": "German zero-focalization counts depend strongly on allwissen*/omniscient_de rules (see S4).",
            "bm25_note": "German BM25-new expansion is large relative to lexical hits; expansion is not validated recall.",
            "missing_years_note": "Many books lack publication year; missing years are not treated as zero observations.",
            "eval_note": "Manual relevance labels for the v0.5 evaluation sample are not yet available; precision/recall remain pending.",
        },
        "lexicon": load_lexicon_payload(),
    }


def plotly_js() -> str:
    path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return path.read_text(encoding="utf-8")


def css() -> str:
    return """
:root {
  --en: #3976C6; --de: #CC8C42; --bg: #f7f8fb; --fg: #1a1d26; --card: #ffffff;
  --muted: #5c6578; --border: #d7dbe7; --accent: #2f5aa8; --danger: #a33b3b;
  --ok: #2f6b4f; --callout: #eef3fb;
}
[data-theme="dark"] {
  --bg: #12141a; --fg: #e8eaf0; --card: #1c2030; --muted: #a7b0c3;
  --border: #30364a; --accent: #7aa2e3; --callout: #232841;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0; font-family: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.5;
}
a { color: var(--accent); }
header.app-header {
  position: sticky; top: 0; z-index: 40; background: var(--card);
  border-bottom: 1px solid var(--border); padding: 0.75rem 1rem;
}
.header-top { display: flex; flex-wrap: wrap; gap: 0.75rem; justify-content: space-between; align-items: center; }
.meta-badges span {
  display: inline-block; background: var(--callout); border: 1px solid var(--border);
  border-radius: 999px; padding: 0.15rem 0.6rem; margin: 0.15rem; font-size: 0.8rem;
}
nav.toc { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
nav.toc a {
  text-decoration: none; font-size: 0.82rem; padding: 0.25rem 0.55rem;
  border: 1px solid var(--border); border-radius: 6px; background: var(--bg); color: var(--fg);
}
main { max-width: 1100px; margin: 0 auto; padding: 1rem 1rem 4rem; }
section {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  padding: 1rem 1.1rem; margin: 1rem 0;
}
h1,h2,h3 { line-height: 1.25; }
h1 { font-size: 1.6rem; margin: 0.2rem 0; }
h2 { font-size: 1.25rem; margin-top: 0; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; }
.muted { color: var(--muted); }
.callout {
  border-left: 4px solid var(--accent); background: var(--callout);
  padding: 0.7rem 0.9rem; margin: 0.8rem 0; border-radius: 0 8px 8px 0;
}
.callout.warn { border-left-color: var(--de); }
.callout.risk { border-left-color: var(--danger); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
@media (max-width: 800px) { .grid2 { grid-template-columns: 1fr; } }
.controls { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.6rem 0; align-items: center; }
.controls label { font-size: 0.85rem; color: var(--muted); }
.controls select, .controls input, .controls button, textarea {
  font: inherit; padding: 0.35rem 0.5rem; border-radius: 6px;
  border: 1px solid var(--border); background: var(--bg); color: var(--fg);
}
button.primary { background: var(--accent); color: white; border-color: var(--accent); cursor: pointer; }
button.ghost { background: transparent; cursor: pointer; }
table.data {
  width: 100%; border-collapse: collapse; font-size: 0.86rem; overflow-x: auto; display: block;
}
table.data th, table.data td {
  border-bottom: 1px solid var(--border); padding: 0.35rem 0.45rem; text-align: left; white-space: nowrap;
}
table.data th { position: sticky; top: 0; background: var(--card); }
.concept-link { color: var(--accent); cursor: pointer; text-decoration: underline; background: none; border: none; font: inherit; padding: 0; }
.concept-card {
  border: 1px solid var(--border); border-radius: 10px; padding: 0.8rem; margin: 0.6rem 0; background: var(--bg);
}
.concept-card h3 { margin: 0 0 0.4rem; }
.tag { display: inline-block; font-size: 0.75rem; padding: 0.1rem 0.45rem; border-radius: 999px; margin: 0.1rem; border: 1px solid var(--border); }
.tag.en { background: color-mix(in srgb, var(--en) 18%, transparent); }
.tag.de { background: color-mix(in srgb, var(--de) 18%, transparent); }
.vocab-row { display: grid; grid-template-columns: 2fr 1fr 1fr 1fr; gap: 0.4rem; padding: 0.35rem 0; border-bottom: 1px solid var(--border); font-size: 0.86rem; }
@media (max-width: 800px) { .vocab-row { grid-template-columns: 1fr; } }
.banner {
  background: #fff4e5; color: #5c3b00; border: 1px solid #f0d2a0; padding: 0.6rem 0.8rem;
  border-radius: 8px; margin: 0.6rem 0;
}
[data-theme="dark"] .banner { background: #3a2c14; color: #f3d9a4; border-color: #6a5224; }
.hidden { display: none !important; }
details.method { margin: 0.5rem 0; }
.chart { width: 100%; min-height: 360px; }
.footer-note { font-size: 0.85rem; color: var(--muted); margin-top: 2rem; }
"""


def js_app() -> str:
    return r"""
const DATA = window.REPORT_DATA;
const LS_KEY = 'np_lexicon_suggestions_v1';

function uid(prefix='id') {
  return prefix + '-' + Math.random().toString(16).slice(2) + '-' + Date.now().toString(16);
}
function ensureReviewer() {
  let id = localStorage.getItem('np_reviewer_id');
  if (!id) { id = uid('reviewer'); localStorage.setItem('np_reviewer_id', id); }
  return id;
}
function loadSuggestions() {
  try { return JSON.parse(localStorage.getItem(LS_KEY) || '[]'); } catch { return []; }
}
function saveSuggestions(arr) {
  localStorage.setItem(LS_KEY, JSON.stringify(arr));
  updateSuggestionCount();
}
function updateSuggestionCount() {
  const el = document.getElementById('suggestion-count');
  if (el) el.textContent = String(loadSuggestions().length);
}

function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('np_theme', theme);
}
function initTheme() {
  const t = localStorage.getItem('np_theme') || 'light';
  setTheme(t);
}

function num(x) {
  if (x === null || x === undefined || x === '') return 0;
  const n = Number(x);
  return Number.isFinite(n) ? n : 0;
}

function openConcept(cid) {
  const card = document.getElementById('card-' + cid);
  if (!card) return;
  card.open = true;
  card.scrollIntoView({behavior:'smooth', block:'start'});
  document.querySelectorAll('.concept-card').forEach(c => c.classList.remove('highlight'));
  card.classList.add('highlight');
}

function metricValue(row, metric) {
  if (metric === 'absolute') return num(row.n_assignments);
  if (metric === 'passages') return num(row.n_unique_passages);
  if (metric === 'prevalence') return num(row.percentage_of_books);
  if (metric === 'body') return num(row.frequency_per_10000_body_words);
  return num(row.frequency_per_10000_eligible_words);
}

function renderFreqChart() {
  const metric = document.getElementById('metric-select').value;
  const family = document.getElementById('family-filter').value;
  let rows = DATA.tables.concept_frequencies || [];
  if (family !== 'all') rows = rows.filter(r => r.concept_family === family);
  const concepts = [...new Set(rows.map(r => r.concept_id))];
  const en = concepts.map(c => {
    const r = rows.find(x => x.concept_id===c && x.language==='en');
    return r ? metricValue(r, metric) : 0;
  });
  const de = concepts.map(c => {
    const r = rows.find(x => x.concept_id===c && x.language==='de');
    return r ? metricValue(r, metric) : 0;
  });
  const ylab = {
    absolute:'Assignments', passages:'Unique passages', prevalence:'Book prevalence (%)',
    eligible:'Per 10k eligible words', body:'Per 10k body words'
  }[metric] || 'Value';
  Plotly.newPlot('chart-freq', [
    {type:'bar', orientation:'h', y:concepts, x:en, name:'English', marker:{color:DATA.colors.en}},
    {type:'bar', orientation:'h', y:concepts, x:de, name:'German', marker:{color:DATA.colors.de}}
  ], {
    barmode:'group', margin:{l:160,r:20,t:30,b:40}, height: Math.max(420, concepts.length*18),
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color: getComputedStyle(document.body).getPropertyValue('--fg')},
    title:'Concept frequencies', xaxis:{title:ylab}
  }, {responsive:true, displayModeBar:false});
  document.getElementById('chart-freq').on('plotly_click', ev => {
    const cid = ev.points[0].y; openConcept(cid);
  });
}

function renderFamilyChart() {
  const rows = DATA.tables.concept_family_frequencies || [];
  const labels = [...new Set(rows.map(r => r.concept_family_label))];
  Plotly.newPlot('chart-family', [
    {type:'bar', orientation:'h', y:labels,
      x: labels.map(l => num((rows.find(r=>r.concept_family_label===l && r.language==='en')||{}).frequency_per_10000_eligible_words)),
      name:'English', marker:{color:DATA.colors.en}},
    {type:'bar', orientation:'h', y:labels,
      x: labels.map(l => num((rows.find(r=>r.concept_family_label===l && r.language==='de')||{}).frequency_per_10000_eligible_words)),
      name:'German', marker:{color:DATA.colors.de}}
  ], {
    barmode:'group', margin:{l:220,r:20,t:30,b:40}, height: 420,
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color: getComputedStyle(document.body).getPropertyValue('--fg')},
    title:'Family frequencies (deduplicated passages)', xaxis:{title:'Per 10k eligible words'}
  }, {responsive:true, displayModeBar:false});
}

function renderDispersion() {
  const rows = (DATA.tables.concept_frequencies||[]).map(f => {
    const d = (DATA.tables.concept_dispersion||[]).find(x => x.concept_id===f.concept_id && x.language===f.language) || {};
    return {...f, ...d};
  });
  Plotly.newPlot('chart-dispersion', [
    {
      type:'scatter', mode:'markers+text',
      x: rows.filter(r=>r.language==='en').map(r=>num(r.pct_books_with_concept)),
      y: rows.filter(r=>r.language==='en').map(r=>num(r.frequency_per_10000_eligible_words)),
      text: rows.filter(r=>r.language==='en').map(r=>r.concept_id),
      textposition:'top center', textfont:{size:8},
      name:'English', marker:{color:DATA.colors.en, size:9}
    },
    {
      type:'scatter', mode:'markers+text',
      x: rows.filter(r=>r.language==='de').map(r=>num(r.pct_books_with_concept)),
      y: rows.filter(r=>r.language==='de').map(r=>num(r.frequency_per_10000_eligible_words)),
      text: rows.filter(r=>r.language==='de').map(r=>r.concept_id),
      textposition:'top center', textfont:{size:8},
      name:'German', marker:{color:DATA.colors.de, size:9}
    }
  ], {
    height: 480, margin:{l:50,r:20,t:40,b:50},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color: getComputedStyle(document.body).getPropertyValue('--fg')},
    title:'Frequency vs book prevalence (exploratory zones)',
    xaxis:{title:'Book prevalence (%)'}, yaxis:{title:'Per 10k eligible words'}
  }, {responsive:true, displayModeBar:false});
}

function renderStdChart() {
  const mode = document.getElementById('std-mode').value;
  const rows = DATA.tables.standardized_frequencies || [];
  const cids = rows.map(r => r.concept_id);
  const enKey = mode === 'std' ? 'freq_en_standardized' : 'freq_en_original';
  const deKey = mode === 'std' ? 'freq_de_standardized' : 'freq_de_original';
  Plotly.newPlot('chart-std', [
    {type:'bar', x:cids, y:rows.map(r=>num(r[enKey])), name:'English', marker:{color:DATA.colors.en}},
    {type:'bar', x:cids, y:rows.map(r=>num(r[deKey])), name:'German', marker:{color:DATA.colors.de}}
  ], {
    barmode:'group', height: 420, margin:{l:40,r:20,t:40,b:120},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color: getComputedStyle(document.body).getPropertyValue('--fg')},
    title: mode==='std' ? 'Standardized frequencies' : 'Original frequencies',
    xaxis:{tickangle:-60}
  }, {responsive:true, displayModeBar:false});
}

function renderComplementarity() {
  const rows = DATA.tables.complementarity || [];
  Plotly.newPlot('chart-comp', [
    {type:'bar', x:rows.map(r=>r.language_label), y:rows.map(r=>num(r.n_L_passages)), name:'Lexical passages', marker:{color:DATA.colors.en}},
    {type:'bar', x:rows.map(r=>r.language_label), y:rows.map(r=>num(r.n_Bnew_passages)), name:'Bnew passages', marker:{color:DATA.colors.de}},
    {type:'bar', x:rows.map(r=>r.language_label), y:rows.map(r=>num(r.n_combined_unique_passages_U)), name:'Combined U', marker:{color:'#6b7280'}}
  ], {
    barmode:'group', height: 360, margin:{l:50,r:20,t:40,b:40},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color: getComputedStyle(document.body).getPropertyValue('--fg')},
    title:'Lexical and BM25 complementarity (passages)'
  }, {responsive:true, displayModeBar:false});
}

function renderSensitivity() {
  const cid = document.getElementById('sens-concept').value;
  const rows = (DATA.tables.sensitivity||[]).filter(r => r.concept_id === cid);
  Plotly.newPlot('chart-sens', [{
    type:'scatter', mode:'lines+markers',
    x: rows.map(r=>r.scenario),
    y: rows.map(r => { const v=Number(r.frequency_ratio_en_over_de); return Number.isFinite(v)?v:null; }),
    name: cid, marker:{size:8}
  }], {
    height: 360, margin:{l:50,r:20,t:40,b:40},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    font:{color: getComputedStyle(document.body).getPropertyValue('--fg')},
    title:'EN/DE frequency ratio across scenarios',
    yaxis:{title:'Ratio EN/DE'}, shapes:[{type:'line', x0:0, x1:1, xref:'paper', y0:1, y1:1, line:{dash:'dot', color:'#888'}}]
  }, {responsive:true, displayModeBar:false});
}

function renderOverviewTable() {
  const rows = DATA.tables.corpus_overview || [];
  const el = document.getElementById('overview-table');
  el.innerHTML = '<table class="data"><thead><tr><th>Language</th><th>Listed</th><th>Analysis</th><th>Excluded</th><th>Eligible words</th><th>Paragraphs</th></tr></thead><tbody>' +
    rows.map(r => `<tr><td>${r.language_label}</td><td>${r.listed_books}</td><td>${r.analysis_set_books}</td><td>${r.excluded_books}</td><td>${Math.round(num(r.eligible_words)).toLocaleString()}</td><td>${r.eligible_paragraphs}</td></tr>`).join('') +
    '</tbody></table>';
}

function renderConceptCards() {
  const host = document.getElementById('concept-cards');
  const freq = DATA.tables.concept_frequencies || [];
  const robust = Object.fromEntries((DATA.tables.robustness||[]).map(r => [r.concept_id, r.robustness]));
  host.innerHTML = DATA.lexicon.concepts.map(c => {
    const en = freq.find(f => f.concept_id===c.concept_id && f.language==='en') || {};
    const de = freq.find(f => f.concept_id===c.concept_id && f.language==='de') || {};
    const enVoc = DATA.lexicon.vocabulary.filter(v => v.concept_id===c.concept_id && v.language==='en');
    const deVoc = DATA.lexicon.vocabulary.filter(v => v.concept_id===c.concept_id && v.language==='de');
    const asym = Math.abs(enVoc.length - deVoc.length) >= 8 || (num(en.n_assignments)>0 && num(de.n_assignments)===0) || (num(de.n_assignments)>0 && num(en.n_assignments)===0);
    return `<details class="concept-card" id="card-${c.concept_id}">
      <summary><strong>${c.concept_label}</strong> <span class="tag">${c.concept_id}</span> <span class="tag">${c.concept_family_label}</span>${asym?' <span class="tag">asymmetric coverage</span>':''}</summary>
      <p><em>Definition (${c.definition_status}):</em> ${c.definition}</p>
      <p class="muted">PsyPer-related tags: ${(c.psyper_mapping||[]).join(', ') || 'none / project-specific'}. ${DATA.lexicon.psyper_disclaimer}</p>
      <p>${c.theoretical_context || ''}</p>
      <p>${c.family_ambiguity_notes || ''}</p>
      <div class="grid2">
        <div><strong>English vocabulary (${enVoc.length})</strong><ul>${enVoc.slice(0,40).map(v=>`<li>${v.term} <span class="tag en">${v.term_type}</span></li>`).join('')}${enVoc.length>40?'<li>…</li>':''}</ul></div>
        <div><strong>German vocabulary (${deVoc.length})</strong><ul>${deVoc.slice(0,40).map(v=>`<li>${v.term} <span class="tag de">${v.term_type}</span></li>`).join('')}${deVoc.length>40?'<li>…</li>':''}</ul></div>
      </div>
      <p><strong>Retrieval stats:</strong> EN assignments=${num(en.n_assignments)}, rate=${num(en.frequency_per_10000_eligible_words).toFixed(3)} /10k;
         DE assignments=${num(de.n_assignments)}, rate=${num(de.frequency_per_10000_eligible_words).toFixed(3)} /10k.
         Robustness: ${robust[c.concept_id] || 'n/a'}.</p>
      <details><summary>Advanced matching / provenance</summary>
        <p class="muted">Technical regex patterns are listed in the lexicon explorer advanced view. Terms above are readable dictionary forms.</p>
      </details>
      <div class="suggest-box">
        <h4>Suggest a terminology change</h4>
        <p class="muted">Stored locally in this browser until you export JSON.</p>
        <div class="controls">
          <label>Expression <input data-role="expr" placeholder="Proposed term"></label>
          <label>Language <select data-role="lang"><option value="en">English</option><option value="de">German</option></select></label>
          <label>Action <select data-role="action"><option value="add_term">Add term</option><option value="add_paraphrase">Add paraphrase</option><option value="add_variant">Add morphological variant</option><option value="remove_term">Remove term</option><option value="change_definition">Change definition</option><option value="change_psyper_mapping">Change PsyPer mapping</option><option value="change_matching">Change matching conditions</option></select></label>
          <label>Type <select data-role="type"><option value="academic">Theoretical term</option><option value="craft">Practical writing expression</option><option value="paraphrase">Paraphrase</option><option value="variant">Morphological variant</option></select></label>
        </div>
        <label>How should this be interpreted?<br><textarea data-role="rationale" rows="2" style="width:100%"></textarea></label>
        <label>Could it need contextual disambiguation?<br><textarea data-role="ambiguity" rows="2" style="width:100%"></textarea></label>
        <div class="controls">
          <label>Confidence <select data-role="confidence"><option>medium</option><option>high</option><option>low</option></select></label>
          <label>Source <input data-role="source" placeholder="optional"></label>
          <label>Example (constructed) <input data-role="example" placeholder="optional constructed example"></label>
          <button class="primary" data-role="save">Save suggestion</button>
        </div>
      </div>
    </details>`;
  }).join('');

  host.querySelectorAll('.suggest-box').forEach(box => {
    const card = box.closest('.concept-card');
    const cid = card.id.replace('card-','');
    box.querySelector('[data-role=save]').addEventListener('click', () => {
      const proposed = box.querySelector('[data-role=expr]').value.trim();
      if (!proposed) { alert('Please enter an expression or description.'); return; }
      const s = {
        suggestion_id: uid('sug'),
        concept_id: cid,
        language: box.querySelector('[data-role=lang]').value,
        action: box.querySelector('[data-role=action]').value,
        proposed_expression: proposed,
        expression_type: box.querySelector('[data-role=type]').value,
        rationale: box.querySelector('[data-role=rationale]').value,
        alternative_concept_id: null,
        ambiguity_notes: box.querySelector('[data-role=ambiguity]').value,
        source_reference: box.querySelector('[data-role=source]').value,
        example_sentence: box.querySelector('[data-role=example]').value,
        confidence: box.querySelector('[data-role=confidence]').value,
        status: 'pending_review'
      };
      const all = loadSuggestions();
      // duplicate detection
      const dup = all.find(x => x.concept_id===s.concept_id && x.language===s.language && x.action===s.action && (x.proposed_expression||'').toLowerCase()===s.proposed_expression.toLowerCase());
      if (dup) { alert('A similar pending suggestion already exists locally.'); return; }
      all.push(s); saveSuggestions(all);
      alert('Suggestion saved locally. Export when ready to share.');
      box.querySelector('[data-role=expr]').value='';
      box.querySelector('[data-role=rationale]').value='';
    });
  });
}

function renderVocabExplorer() {
  const q = (document.getElementById('vocab-search').value || '').toLowerCase();
  const lang = document.getElementById('vocab-lang').value;
  const family = document.getElementById('vocab-family').value;
  const prov = document.getElementById('vocab-prov').value;
  const ttype = document.getElementById('vocab-type').value;
  const famMap = Object.fromEntries(DATA.lexicon.concepts.map(c => [c.concept_id, c.concept_family]));
  let rows = DATA.lexicon.vocabulary.slice();
  if (lang !== 'all') rows = rows.filter(r => r.language === lang);
  if (family !== 'all') rows = rows.filter(r => famMap[r.concept_id] === family);
  if (prov !== 'all') rows = rows.filter(r => r.provenance_type === prov);
  if (ttype !== 'all') rows = rows.filter(r => r.term_type === ttype);
  if (q) rows = rows.filter(r => (r.term||'').toLowerCase().includes(q) || (r.concept_id||'').includes(q));
  const host = document.getElementById('vocab-list');
  host.innerHTML = `<div class="muted">${rows.length} terms shown</div>` + rows.slice(0, 300).map(r =>
    `<div class="vocab-row"><div><button class="concept-link" onclick="openConcept('${r.concept_id}')">${r.term}</button></div>
     <div>${r.language}</div><div>${r.term_type}</div><div>${r.provenance_type || '—'} · ${r.concept_id}</div></div>`
  ).join('') + (rows.length>300 ? '<p class="muted">Showing first 300 matches. Refine filters to narrow.</p>' : '');
}

function exportSuggestions(fmt) {
  const payload = {
    schema_version: '1.0',
    lexicon_version: DATA.lexicon.lexicon_version,
    corpus_snapshot_id: (DATA.meta.corpus_snapshot_id || ''),
    reviewer_id: ensureReviewer(),
    suggestions: loadSuggestions()
  };
  let blob, name;
  if (fmt === 'csv') {
    const cols = ['suggestion_id','concept_id','language','action','proposed_expression','expression_type','rationale','alternative_concept_id','ambiguity_notes','source_reference','example_sentence','confidence','status'];
    const lines = [cols.join(',')].concat(payload.suggestions.map(s => cols.map(c => JSON.stringify(s[c] ?? '')).join(',')));
    blob = new Blob([lines.join('\n')], {type:'text/csv'});
    name = 'terminology_suggestions.csv';
  } else {
    blob = new Blob([JSON.stringify(payload, null, 2)], {type:'application/json'});
    name = 'terminology_suggestions.json';
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
}

function importSuggestions(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const data = JSON.parse(reader.result);
      const incoming = data.suggestions || [];
      const all = loadSuggestions();
      let added = 0;
      incoming.forEach(s => {
        if (!s.suggestion_id) s.suggestion_id = uid('sug');
        const dup = all.find(x => x.suggestion_id === s.suggestion_id ||
          (x.concept_id===s.concept_id && x.language===s.language && x.action===s.action && (x.proposed_expression||'').toLowerCase()===(s.proposed_expression||'').toLowerCase()));
        if (!dup) { all.push(s); added++; }
      });
      saveSuggestions(all);
      alert(`Imported ${added} new suggestions (${incoming.length} in file).`);
      renderSuggestionList();
    } catch (e) { alert('Invalid JSON: ' + e.message); }
  };
  reader.readAsText(file);
}

function renderSuggestionList() {
  const host = document.getElementById('suggestion-list');
  const all = loadSuggestions();
  host.innerHTML = all.length ? all.map((s,i) =>
    `<div class="vocab-row"><div>${s.proposed_expression} <span class="tag">${s.action}</span></div>
     <div>${s.language} / ${s.concept_id}</div><div>${s.status}</div>
     <div><button class="ghost" data-i="${i}">Delete</button></div></div>`
  ).join('') : '<p class="muted">No local suggestions yet.</p>';
  host.querySelectorAll('button[data-i]').forEach(btn => {
    btn.addEventListener('click', () => {
      const arr = loadSuggestions(); arr.splice(Number(btn.dataset.i),1); saveSuggestions(arr); renderSuggestionList();
    });
  });
}

function populateFilters() {
  const families = [...new Set(DATA.lexicon.concepts.map(c => c.concept_family))];
  ['family-filter','vocab-family'].forEach(id => {
    const el = document.getElementById(id);
    families.forEach(f => {
      const o = document.createElement('option'); o.value=f; o.textContent=f; el.appendChild(o);
    });
  });
  const sens = document.getElementById('sens-concept');
  DATA.lexicon.concepts.forEach(c => {
    const o=document.createElement('option'); o.value=c.concept_id; o.textContent=c.concept_id; sens.appendChild(o);
  });
  sens.value = 'focalization_zero';
  const provs = [...new Set(DATA.lexicon.vocabulary.map(v => v.provenance_type).filter(Boolean))];
  const vp = document.getElementById('vocab-prov');
  provs.forEach(p => { const o=document.createElement('option'); o.value=p; o.textContent=p; vp.appendChild(o); });
}

function boot() {
  initTheme();
  ensureReviewer();
  updateSuggestionCount();
  document.getElementById('reviewer-id').textContent = ensureReviewer();
  const m = DATA.meta || {};
  document.getElementById('badge-snapshot').textContent = m.corpus_snapshot_id || 'n/a';
  document.getElementById('badge-lexicon').textContent = m.lexicon_version || DATA.lexicon.lexicon_version;
  document.getElementById('badge-retrieval').textContent = m.retrieval_execution_id || 'v03';
  document.getElementById('badge-analysis').textContent = m.analysis_version || 'v2';
  document.getElementById('badge-generated').textContent = DATA.generated_at;

  populateFilters();
  renderOverviewTable();
  renderConceptCards();
  renderVocabExplorer();
  renderSuggestionList();
  renderFreqChart();
  renderFamilyChart();
  renderDispersion();
  renderStdChart();
  renderComplementarity();
  renderSensitivity();

  document.getElementById('metric-select').addEventListener('change', renderFreqChart);
  document.getElementById('family-filter').addEventListener('change', renderFreqChart);
  document.getElementById('std-mode').addEventListener('change', renderStdChart);
  document.getElementById('sens-concept').addEventListener('change', renderSensitivity);
  ['vocab-search','vocab-lang','vocab-family','vocab-prov','vocab-type'].forEach(id => {
    document.getElementById(id).addEventListener('input', renderVocabExplorer);
    document.getElementById(id).addEventListener('change', renderVocabExplorer);
  });
  document.getElementById('theme-toggle').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    setTheme(cur); renderFreqChart(); renderFamilyChart(); renderDispersion(); renderStdChart(); renderComplementarity(); renderSensitivity();
  });
  document.getElementById('export-json').addEventListener('click', () => exportSuggestions('json'));
  document.getElementById('export-csv').addEventListener('click', () => exportSuggestions('csv'));
  document.getElementById('import-json').addEventListener('change', (e) => {
    if (e.target.files[0]) importSuggestions(e.target.files[0]);
  });
}
document.addEventListener('DOMContentLoaded', boot);
"""


def html_body(data: dict) -> str:
    callouts = data["callouts"]
    zero = ", ".join(f"`{c}`" for c in callouts.get("zero_match_concepts_de") or []) or "none"
    return f"""
<header class="app-header">
  <div class="header-top">
    <div>
      <h1>Narrative perspective in creative-writing manuals</h1>
      <p class="muted">Interactive bilingual research report (analysis v2). English interface; terms kept in original languages.</p>
      <div class="meta-badges">
        <span>Snapshot: <strong id="badge-snapshot"></strong></span>
        <span>Lexicon: <strong id="badge-lexicon"></strong></span>
        <span>Retrieval: <strong id="badge-retrieval"></strong></span>
        <span>Analysis: <strong id="badge-analysis"></strong></span>
        <span>Generated: <strong id="badge-generated"></strong></span>
      </div>
    </div>
    <div class="controls">
      <button class="ghost" id="theme-toggle" type="button">Toggle light/dark</button>
      <span class="muted">Reviewer <code id="reviewer-id"></code> · suggestions <strong id="suggestion-count">0</strong></span>
    </div>
  </div>
  <nav class="toc" aria-label="Section navigation">
    <a href="#sec-background">Background</a>
    <a href="#sec-corpus">Corpus</a>
    <a href="#sec-methods">Methods</a>
    <a href="#sec-coverage">Coverage</a>
    <a href="#sec-concepts">Concepts</a>
    <a href="#sec-families">Families</a>
    <a href="#sec-dispersion">Dispersion</a>
    <a href="#sec-categories">Categories</a>
    <a href="#sec-cooccurrence">Co-occurrence</a>
    <a href="#sec-bm25">Lexical–BM25</a>
    <a href="#sec-sensitivity">Sensitivity</a>
    <a href="#sec-german">German audit</a>
    <a href="#sec-lexicon">Lexicon</a>
    <a href="#sec-interpret">Interpretations</a>
    <a href="#sec-limitations">Limitations</a>
  </nav>
</header>
<main>
<section id="sec-background">
  <h2>1. Research background and questions</h2>
  <p>This project investigates how narrative perspective is conceptualized, discussed and taught in English and German creative-writing manuals, using a bilingual retrieval lexicon informed by the PsyPer annotation scheme.</p>
  <p><strong>Observed retrieval patterns are not validated linguistic findings.</strong> Frequency counts reflect dictionary matches in searchable paragraphs. Precision and recall await human annotation of the evaluation sample.</p>
  <div class="callout">{callouts['corpus_size_note']}</div>
</section>

<section id="sec-corpus">
  <h2>2. Corpus composition and data quality</h2>
  <div id="overview-table"></div>
  <div class="callout warn">{callouts['composition_note']}</div>
  <div class="callout">{callouts['missing_years_note']}</div>
  <details class="method"><summary>Word-count denominators</summary>
    <p>Primary rates use <strong>eligible searchable words</strong> (body paragraphs with at least 40 characters). Cleaned body words are a supplementary denominator. Differences between these fields are expected: short paragraphs contribute to body words but not to eligible passage words.</p>
  </details>
  <p>Extreme-length passages (≥100,000 characters): 56 total. Resegment gate: <strong>{"fired" if data.get("gate",{}).get("resegment_gate_fires") else "did not fire"}</strong> (no recoverable blank-line structure). Original passage IDs retained; scenario S3 drops them for sensitivity.</p>
</section>

<section id="sec-methods">
  <h2>3. Retrieval methodology</h2>
  <p>Lexical retrieval (v0.3) matches dictionary terms and concept-scoped regexes. BM25 (v0.3) provides qualified hits (Bq) and an exploratory BM25-only file (Bnew). Historical raw BM25 heap (Braw) is <strong>not</strong> on disk and is not reconstructed.</p>
  <p>Pools: <code>L</code> lexical · <code>Bq</code> qualified BM25 · <code>Bnew</code> BM25-only after subtracting L · <code>U</code> = L ∪ Bnew.</p>
</section>

<section id="sec-coverage">
  <h2>4. Overall retrieval coverage</h2>
  <p>A <em>passage</em> is a unique paragraph. An <em>assignment</em> is a passage–concept pair. One passage may carry several concept assignments.</p>
  <p class="muted">See table <code>02_retrieval_coverage.csv</code> for exact counts.</p>
</section>

<section id="sec-concepts">
  <h2>5. Concept frequencies</h2>
  <div class="controls">
    <label>Metric
      <select id="metric-select">
        <option value="eligible">Normalized (eligible words)</option>
        <option value="absolute">Absolute assignments</option>
        <option value="passages">Unique passages</option>
        <option value="prevalence">Book prevalence</option>
        <option value="body">Normalized (body words)</option>
      </select>
    </label>
    <label>Family
      <select id="family-filter"><option value="all">All families</option></select>
    </label>
  </div>
  <div id="chart-freq" class="chart"></div>
  <div class="callout warn">German zero-match concepts: {zero}</div>
  <p>Click a concept name in the chart to open its card below.</p>
</section>

<section id="sec-families">
  <h2>6. Concept families</h2>
  <p>Family frequencies count unique passages within each family. One passage may belong to several families; do not sum family rates as mutually exclusive totals.</p>
  <div id="chart-family" class="chart"></div>
</section>

<section id="sec-dispersion">
  <h2>7. Book-level dispersion</h2>
  <p>Zones are exploratory visual aids, not formal statistical classes.</p>
  <div id="chart-dispersion" class="chart"></div>
</section>

<section id="sec-categories">
  <h2>8. Corpus-category comparisons and standardization</h2>
  <p>Shared reference weights are the mean of English and German category eligible-word shares. Standardization adjusts for composition differences; it does not remove dictionary bias.</p>
  <div class="controls">
    <label>Show
      <select id="std-mode">
        <option value="original">Original frequencies</option>
        <option value="std">Standardized frequencies</option>
      </select>
    </label>
  </div>
  <div id="chart-std" class="chart"></div>
</section>

<section id="sec-cooccurrence">
  <h2>9. Concept co-occurrence</h2>
  <p>Co-occurrence uses unique passages. High overlap may reflect shared dictionary expressions, overlapping regexes, long passages, or genuine multi-concept discussion. It is not evidence of causality or conceptual equivalence.</p>
  <p class="muted">Pair tables: <code>02_concept_cooccurrence_pairs_en.csv</code> / <code>_de.csv</code> (and <code>_no_extreme</code> variants).</p>
</section>

<section id="sec-bm25">
  <h2>10. Lexical and BM25 complementarity</h2>
  <div class="callout warn">{callouts['bm25_note']}</div>
  <div id="chart-comp" class="chart"></div>
</section>

<section id="sec-sensitivity">
  <h2>11. Sensitivity analysis</h2>
  <div class="callout risk">{callouts['focalization_zero_note']}</div>
  <div class="callout warn">{callouts['stance_note']}</div>
  <div class="controls">
    <label>Concept <select id="sens-concept"></select></label>
  </div>
  <div id="chart-sens" class="chart"></div>
  <p>Scenarios S1–S11 include composition, extreme passages, allwissen* conservatism (S4), ambiguous German stance triggers (S7), influential books, and alternative denominators.</p>
</section>

<section id="sec-german">
  <h2>12. German terminology audit</h2>
  <p>The German audit file is <strong>heuristic development data</strong>, not an independent precision estimate. Human review fields are empty. The 71/78 figure must not be read as validated precision.</p>
</section>

<section id="sec-lexicon">
  <h2>13. Interactive bilingual lexicon</h2>
  <div class="banner">Suggestions are stored only in this browser until you export a JSON or CSV file and share it. The active retrieval dictionary is never modified automatically.</div>
  <div class="controls">
    <input id="vocab-search" placeholder="Search vocabulary">
    <select id="vocab-lang"><option value="all">All languages</option><option value="en">English</option><option value="de">German</option></select>
    <select id="vocab-family"><option value="all">All families</option></select>
    <select id="vocab-prov"><option value="all">All provenance</option></select>
    <select id="vocab-type"><option value="all">All term types</option><option value="academic">academic</option><option value="craft">craft</option><option value="paraphrases">paraphrases</option><option value="context">context</option></select>
    <button class="primary" id="export-json" type="button">Export JSON</button>
    <button class="ghost" id="export-csv" type="button">Export CSV</button>
    <label class="ghost">Import JSON <input id="import-json" type="file" accept="application/json"></label>
  </div>
  <div id="vocab-list"></div>
  <h3>Concept cards</h3>
  <div id="concept-cards"></div>
  <h3>Local suggestions</h3>
  <div id="suggestion-list"></div>
</section>

<section id="sec-interpret">
  <h2>14. Preliminary interpretations</h2>
  <p>Interpretations below are provisional readings of retrieval patterns:</p>
  <ul>
    <li>English and German corpora differ in size and category mix; standardized rates help separate composition from residual differences.</li>
    <li>German stance and zero-focalization signals are sensitive to particular lexical rules (allwissen* vs Stanzel vocabulary).</li>
    <li>BM25 adds many German candidates; without annotation this is complementarity, not recall gain.</li>
  </ul>
</section>

<section id="sec-limitations">
  <h2>15. Limitations and future work</h2>
  <div class="callout risk">{callouts['eval_note']}</div>
  <ul>
    <li>Dictionary bias is not captured by bootstrap intervals.</li>
    <li>work_id is missing for most books; confirmed work-level deduplication is limited to surplus_duplicate rules.</li>
    <li>Extreme passages remain unsegmented (no recoverable breaks).</li>
    <li>Future methods (dense retrieval, hybrid, RRF) can append to <code>02_assignments_long.csv</code> without rewriting the analysis framework.</li>
  </ul>
  <p class="footer-note">This HTML contains aggregated statistics, definitions and dictionary terms only. It does not include copyrighted full texts or corpus excerpts.</p>
</section>
</main>
"""


def build_html(data: dict) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    # Escape </script> in JSON
    data_json = data_json.replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Narrative perspective — bilingual interactive report (v2)</title>
<style>{css()}</style>
<script>{plotly_js()}</script>
</head>
<body>
{html_body(data)}
<script>window.REPORT_DATA = {data_json};</script>
<script>{js_app()}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    log = setup_logger(__file__)
    log.info("Loading report data")
    data = load_report_data()
    log.info("Building HTML (inlining Plotly)")
    html = build_html(data)
    OUT.write_text(html, encoding="utf-8")
    # Sanity: no external http script/link
    if re.search(r'<script[^>]+src=["\']https?://', html) or re.search(
        r'<link[^>]+href=["\']https?://', html
    ):
        raise RuntimeError("HTML contains external network dependencies")
    log.info(f"Wrote {OUT} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
