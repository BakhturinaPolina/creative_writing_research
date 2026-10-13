"""Google Books Ngram retrieval (undocumented JSON endpoint) with local caching.

Every response is stored in data/raw/ngrams/ together with its query parameters.
Smoothing is always requested as 0; any smoothing is applied locally.
"""
import argparse
import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

from src.common.logging_util import setup_logger
from src.common.paths import CONFIG, DATA_PROCESSED, DATA_RAW, ROOT

BASE_URL = "https://books.google.com/ngrams/json"
RAW_DIR = DATA_RAW / "ngrams"
PROCESSED_DIR = DATA_PROCESSED
FIGURES_DIR = ROOT / "figures" / "ngrams"


def load_terms(group, path=None):
    path = path or (CONFIG / "ngram_terms.yml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)[group]


def fetch_ngram(terms, corpus, year_start=1800, year_end=2022, smoothing=0, case_insensitive=True):
    params = {
        "content": ",".join(terms),
        "year_start": year_start,
        "year_end": year_end,
        "corpus": corpus,
        "smoothing": smoothing,
    }
    if case_insensitive:
        params["case_insensitive"] = "true"
    response = requests.get(
        BASE_URL,
        params=params,
        timeout=60,
        headers={"User-Agent": "academic-research-ngram-project/1.0"},
    )
    response.raise_for_status()
    return response.json(), response.url


def save_raw(data, url, path, corpus, year_start, year_end, smoothing, terms):
    payload = {
        "query_url": url,
        "retrieved": datetime.now(timezone.utc).isoformat(),
        "corpus": corpus,
        "year_start": year_start,
        "year_end": year_end,
        "smoothing": smoothing,
        "terms": list(terms),
        "data": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ngram_json_to_long(data, year_start):
    """With case_insensitive=true Google returns case variants plus a combined '<term> (All)' series; keep the latter."""
    combined = [s for s in data if s["ngram"].endswith(" (All)")]
    data = combined or data
    rows = [
        {"term": s["ngram"].removesuffix(" (All)"), "year": year_start + i, "frequency": freq}
        for s in data
        for i, freq in enumerate(s["timeseries"])
    ]
    return pd.DataFrame(rows)


def get_ngram_frame(terms, corpus, name, year_start=1800, year_end=2022, refresh=False):
    """Return long DataFrame; use cached raw JSON unless refresh is True."""
    path = RAW_DIR / f"{corpus}_{name}.json"
    if path.exists() and not refresh:
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        data, url = fetch_ngram(terms, corpus, year_start, year_end)
        save_raw(data, url, path, corpus, year_start, year_end, 0, terms)
        payload = json.loads(path.read_text(encoding="utf-8"))
    return ngram_json_to_long(payload["data"], payload["year_start"])


def add_smoothing(df, window=5):
    df = df.sort_values(["term", "year"]).copy()
    grouped = df.groupby("term")["frequency"]
    df["frequency_5yr"] = grouped.transform(lambda x: x.rolling(window, center=True).mean())
    df["relative_to_peak"] = grouped.transform(lambda x: x / x.max() if x.max() > 0 else x)
    return df


def sustained_emergence(group, proportion=0.10, window=5):
    """First year of a `window`-year run with frequency >= proportion * term maximum."""
    group = group.sort_values("year").reset_index(drop=True)
    threshold = group["frequency"].max() * proportion
    if threshold <= 0:
        return None
    sustained = (group["frequency"] >= threshold).rolling(window).sum() >= window
    if not sustained.any():
        return None
    return int(group.loc[sustained.idxmax(), "year"] - window + 1)


def milestones(df):
    rows = []
    for term, g in df.groupby("term"):
        nonzero = g.loc[g["frequency"] > 0, "year"]
        rows.append(
            {
                "term": term,
                "first_nonzero_year": int(nonzero.min()) if len(nonzero) else None,
                "peak_year": int(g.loc[g["frequency"].idxmax(), "year"]),
                "peak_frequency": g["frequency"].max(),
                "sustained_emergence_year": sustained_emergence(g),
            }
        )
    return pd.DataFrame(rows)


def plot_terms(df, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for term, g in df.groupby("term"):
        ax.plot(g["year"], g["frequency_5yr"], label=term)
    ax.set_xlabel("Year")
    ax.set_ylabel("Relative frequency (5-yr mean)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", required=True, help="Key in config/ngram_terms.yml")
    p.add_argument("--corpus", required=True, help="e.g. en, de, eng_2019")
    p.add_argument("--year-start", type=int, default=1800)
    p.add_argument("--year-end", type=int, default=2022)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    log = setup_logger(__file__)
    try:
        log.info(f"Fetching ngrams group={args.group!r} corpus={args.corpus!r}")
        terms = load_terms(args.group)
        df = get_ngram_frame(
            terms, args.corpus, args.group, args.year_start, args.year_end, args.refresh
        )
        df = add_smoothing(df)

        stem = f"ngram_{args.corpus}_{args.group}"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = PROCESSED_DIR / f"{stem}.csv"
        out_miles = PROCESSED_DIR / f"{stem}_milestones.csv"
        out_fig = FIGURES_DIR / f"{stem}.png"
        df.to_csv(out_csv, index=False)
        miles = milestones(df)
        miles.to_csv(out_miles, index=False)
        plot_terms(df, f"{args.group} ({args.corpus})", out_fig)
        log.info(f"saved {out_csv}, {out_miles}, {out_fig}")
        log.info("\n" + miles.to_string(index=False))
    finally:
        log.close()


if __name__ == "__main__":
    main()
