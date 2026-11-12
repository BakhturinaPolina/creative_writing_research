"""Figures for the corpus statistics notebook (one function per figure).

House rules, applied to every figure:
- one fixed colour and marker per corpus (Okabe-Ito blue / vermillion, readable
  with colour-vision deficiency and in greyscale),
- distributions are shown as single books plus median and middle half, never as bars of means,
- proportions carry 95% Wilson intervals,
- the number of books (n) is written into the figure,
- count and percentage axes start at zero; log axes are labelled as such.

Each function takes the book-level table from `corpus_stats.load_book_stats()`
(already filtered to the books that should be shown) and returns the figure.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from src.analysis.corpus_stats import (
    DIMENSION_LABELS,
    EXCLUSION_REASON_LABELS,
    PRIMARY_GROUP_LABELS,
    ROLE_LABELS,
    ROLE_SOURCE_LABELS,
    ROLES,
    dimension_columns,
    wilson_interval,
)
from src.common.paths import ROOT

FIGURES_DIR = ROOT / "figures" / "corpus_stats"
COLORS = {"primary": "#0072B2", "comparison": "#D55E00"}
MARKERS = {"primary": "o", "comparison": "s"}
# Within-primary robustness cut (Okabe-Ito bluish green for pedagogy/genre craft).
GROUP_COLORS = {
    "fiction_manual": "#0072B2",
    "pedagogy_genre_craft": "#009E73",
    "contrast": "#D55E00",
}
GROUP_MARKERS = {
    "fiction_manual": "o",
    "pedagogy_genre_craft": "D",
    "contrast": "s",
}
GROUP_ORDER = ["fiction_manual", "pedagogy_genre_craft", "contrast"]
INK = "#222222"
GREY = "#666666"
DENSITY_LABEL = "POV terms per 10,000 words"
DENSITY_TICKS = [0, 0.1, 1, 10, 100]


def set_style() -> None:
    """Shared look for all figures. Call once at the top of the notebook."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": GREY,
        "axes.axisbelow": True,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # text stays editable in the PDF
    })


def save_figure(fig, name: str) -> None:
    """Save as PNG (300 dpi, for slides) and PDF (vector, for print)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", metadata={"CreationDate": None})  # no date: identical re-runs give identical files


def _corpus_label(role: str, n: int) -> str:
    return f"{ROLE_LABELS[role]} (n = {n})"


def _strip(ax, position: float, values, role: str, rng, vertical: bool = False, spread: float = 0.27) -> None:
    """One dot per book with jitter, a thick bar for the middle half and a tick for the median."""
    values = np.asarray(values, dtype=float)
    jitter = position + rng.uniform(-spread, spread, len(values))
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    dots = dict(s=13, color=COLORS[role], alpha=0.35, linewidths=0, marker=MARKERS[role], zorder=2)
    if vertical:
        ax.scatter(jitter, values, **dots)
        ax.plot([position, position], [q1, q3], color=INK, lw=3.5, solid_capstyle="butt", zorder=3)
        ax.plot([position], [median], marker="_", ms=20, mew=2.5, color=INK, zorder=4)
    else:
        ax.scatter(values, jitter, **dots)
        ax.plot([q1, q3], [position, position], color=INK, lw=3.5, solid_capstyle="butt", zorder=3)
        ax.plot([median], [position], marker="|", ms=20, mew=2.5, color=INK, zorder=4)


def _strip_legend(ax, vertical: bool = False, **kwargs) -> None:
    handles = [
        Line2D([], [], marker="o", color=GREY, alpha=0.5, lw=0, ms=4, label="one book"),
        Line2D([], [], color=INK, lw=3.5, label="middle half of the books"),
        Line2D([], [], marker="_" if vertical else "|", color=INK, lw=0, ms=12, mew=2.5, label="median"),
    ]
    ax.legend(handles=handles, **kwargs)


def _density_axis(ax, axis: str = "x") -> None:
    """Log-type axis for densities that keeps a place for zero (books without any POV term)."""
    scale = dict(linthresh=0.1, linscale=0.35)
    if axis == "x":
        ax.set_xscale("symlog", **scale)
        ax.set_xlim(-0.03, 400)
        ax.set_xticks(DENSITY_TICKS, [str(t) for t in DENSITY_TICKS])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlabel(f"{DENSITY_LABEL} (log scale; 0 = no term found)")
    else:
        ax.set_yscale("symlog", **scale)
        ax.set_ylim(-0.03, 400)
        ax.set_yticks(DENSITY_TICKS, [str(t) for t in DENSITY_TICKS])
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylabel(f"{DENSITY_LABEL}\n(log scale; 0 = no term found)")


# --------------------------------------------------------------------------- figures

def plot_label_sources(files: pd.DataFrame):
    """Who or what decided the folder label, and why books went to `comparison`."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.75})

    sources = list(ROLE_SOURCE_LABELS)
    height = 0.38
    for offset, role in zip((-height / 2, height / 2), ROLES):
        subset = files[files["corpus_role"] == role]
        counts = subset["role_source"].value_counts().reindex(sources, fill_value=0)
        shares = counts / len(subset) * 100
        positions = np.arange(len(sources)) + offset
        left.barh(positions, shares, height=height, color=COLORS[role], label=_corpus_label(role, len(subset)))
        for y, share, count in zip(positions, shares, counts):
            left.text(share + 1.5, y, f"{share:.0f}% ({count})", va="center", fontsize=9, color=INK)
    left.set_yticks(np.arange(len(sources)), [ROLE_SOURCE_LABELS[s] for s in sources])
    left.invert_yaxis()
    left.set_xlim(0, 118)
    left.set_xticks([0, 25, 50, 75, 100])
    left.set_xlabel("Share of the corpus's files (%)")
    left.grid(axis="x")
    left.legend(loc="lower right", fontsize=9)
    left.set_title("Where does the folder label come from?")

    comparison = files[files["corpus_role"] == "comparison"]
    reasons = comparison["exclusion_reason"].replace("", "none").value_counts()
    labels = [EXCLUSION_REASON_LABELS.get(r, "No reason recorded" if r == "none" else r) for r in reasons.index]
    bar_colors = [GREY if r == "none" else COLORS["comparison"] for r in reasons.index]
    right.barh(np.arange(len(reasons)), reasons.values, color=bar_colors)
    for y, count in enumerate(reasons.values):
        right.text(count + 1, y, str(count), va="center", fontsize=9, color=INK)
    right.set_yticks(np.arange(len(reasons)), labels)
    right.invert_yaxis()
    right.set_xlim(0, reasons.max() * 1.12)
    right.set_xlabel(f"Number of files (n = {len(comparison)})")
    right.grid(axis="x")
    right.set_title("Why is a book in comparison?")
    return fig


YEAR_COLUMN_LABELS = {
    "edition_year_clean": "edition we hold",
    "work_year_clean": "first publication",
}


def plot_year_decades(books: pd.DataFrame):
    """Books per decade of the edition and of first publication, side by side, with an undated bar.

    Expects the table from `corpus_stats.load_time_books()`.
    """
    columns = list(YEAR_COLUMN_LABELS)
    decade = {c: (books[c] // 10 * 10).astype("Int64") for c in columns}
    low = min(int(d.min()) for d in decade.values())
    high = max(int(d.max()) for d in decade.values())
    decades = list(range(low, high + 10, 10))
    positions = list(range(len(decades))) + [len(decades) + 0.6]
    width = 0.4
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 5.8), sharex=True, sharey=True, gridspec_kw={"hspace": 0.32})
    for ax, role in zip(axes, ROLES):
        mask = books["corpus_role"] == role
        for offset, column in zip((-width / 2, width / 2), columns):
            counts = [int((decade[column][mask] == d).sum()) for d in decades] + [int(decade[column][mask].isna().sum())]
            x = [p + offset for p in positions]
            solid = column == "edition_year_clean"
            bars = ax.bar(x, counts, width=width, color=COLORS[role] if solid else "white",
                          edgecolor=COLORS[role], linewidth=1.2)
            bars[-1].set(facecolor="white", hatch="///")
            for xi, count in zip(x, counts):
                if count:
                    ax.text(xi, count + 1.5, str(count), ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_ylabel("Number of books")
        ax.grid(axis="y")
        ax.set_title(_corpus_label(role, int(mask.sum())), fontsize=10.5, color=COLORS[role])
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.1)
    axes[1].set_xticks(positions, [f"{d}s" for d in decades] + ["no clean\nyear"])
    axes[1].set_xlabel("Decade")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=GREY, edgecolor=GREY, label="year of the edition we hold"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GREY, linewidth=1.2, label="year of first publication"),
               plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GREY, hatch="///", label="no clean year")]
    axes[0].legend(handles=handles, loc="upper left", fontsize=9)
    fig.suptitle("Which decades do the books come from? (clean years only)", x=0.125, ha="left",
                 fontweight="bold", fontsize=12)
    return fig


def plot_edition_lag(books: pd.DataFrame, label_titles: tuple[str, ...] = ("Becoming a Writer", "The Writing of Fiction",
                                                                            "Writing Fiction")):
    """Years between first publication and the edition we hold, for books with both clean years."""
    rng = np.random.default_rng(4)
    both = books[books["edition_lag"].notna()]
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    labels = []
    for position, role in enumerate(ROLES):
        part = both[both["corpus_role"] == role]
        values = part["edition_lag"].astype(float)
        jitter = position + rng.uniform(-0.27, 0.27, len(values))
        ax.scatter(values, jitter, s=13, color=COLORS[role], alpha=0.35, linewidths=0, marker=MARKERS[role], zorder=2)
        later = int((values > 0).sum())
        labels.append(f"{ROLE_LABELS[role]}\n(n = {len(values)})")
        ax.text(1.01, position, f"{later} later edition{'s' if later != 1 else ''}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9.5, color=INK)
        for (_, row), y in zip(part.iterrows(), jitter):
            if row["title"] in label_titles and row["edition_lag"] >= 30:
                ax.annotate(f"{row['author'].split(';')[0].split()[-1]}, {row['work_year_clean']}\u2192{row['edition_year_clean']}",
                            (row["edition_lag"], y), xytext=(0, 9), textcoords="offset points", ha="center",
                            fontsize=8.5, color=INK)
    ax.set_yticks(range(len(ROLES)), labels)
    ax.set_ylim(len(ROLES) - 0.45, -0.7)
    ax.set_xlim(-3, max(10, float(both["edition_lag"].max()) + 8))
    ax.set_xlabel("Years from first publication to the edition we hold (0 = same year)")
    ax.grid(axis="x")
    ax.set_title("How old are the works behind the editions?")
    return fig


def plot_word_counts(books: pd.DataFrame):
    """Book length in words, by corpus and file format."""
    rng = np.random.default_rng(1)
    kind = books["format"].map(lambda f: "PDF" if f == "pdf" else "E-book")
    rows = [(role, k) for role in ROLES for k in ("E-book", "PDF")]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    labels = []
    for position, (role, k) in enumerate(rows):
        values = books.loc[(books["corpus_role"] == role) & (kind == k), "n_words"].astype(float)
        _strip(ax, position, values, role, rng)
        labels.append(f"{ROLE_LABELS[role]}, {k}\n(n = {len(values)})")
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, None)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
    ax.set_xlabel("Words per book (thousands)")
    ax.grid(axis="x")
    _strip_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=9)
    ax.set_title("How long are the books?")
    return fig


def plot_pov_density(books: pd.DataFrame):
    """POV terms per 10,000 words for every book, primary against comparison."""
    rng = np.random.default_rng(2)
    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    labels = []
    for position, role in enumerate(ROLES):
        values = books.loc[books["corpus_role"] == role, "pov_per_10k"]
        _strip(ax, position, values, role, rng)
        labels.append(f"{ROLE_LABELS[role]}\n(n = {len(values)})")
        ax.text(1.01, position, f"median {values.median():.1f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(ROLES)), labels)
    ax.set_ylim(len(ROLES) - 0.45, -0.55)
    _density_axis(ax, "x")
    ax.grid(axis="x")
    _strip_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=3, fontsize=9)
    ax.set_title("How often do point-of-view terms occur in each book?")
    return fig


def plot_pov_density_robustness(books: pd.DataFrame):
    """Same density plot, with primary split into fiction manuals vs pedagogy/genre craft."""
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    labels = []
    for position, key in enumerate(GROUP_ORDER):
        values = books.loc[books["primary_group"] == key, "pov_per_10k"]
        y = np.full(len(values), position, dtype=float)
        if len(values):
            y = y + rng.uniform(-0.18, 0.18, size=len(values))
            ax.scatter(values, y, s=18, alpha=0.55, color=GROUP_COLORS[key],
                       marker=GROUP_MARKERS[key], linewidths=0, zorder=2)
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            ax.plot([q1, q3], [position, position], color=INK, lw=3.2, solid_capstyle="butt", zorder=3)
            ax.plot(values.median(), position, marker="|", color=INK, markersize=14, markeredgewidth=2.4, zorder=4)
        labels.append(f"{PRIMARY_GROUP_LABELS[key]}\n(n = {len(values)})")
        if len(values):
            ax.text(1.01, position, f"median {values.median():.1f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=9.5, color=INK)
    ax.set_yticks(range(len(GROUP_ORDER)), labels)
    ax.set_ylim(len(GROUP_ORDER) - 0.45, -0.55)
    _density_axis(ax, "x")
    ax.grid(axis="x")
    handles = [
        Line2D([], [], marker=GROUP_MARKERS[k], color=GROUP_COLORS[k], lw=0, ms=6, label=PRIMARY_GROUP_LABELS[k])
        for k in GROUP_ORDER
    ]
    handles += [
        Line2D([], [], color=INK, lw=3.2, label="middle half of the books"),
        Line2D([], [], marker="|", color=INK, lw=0, ms=12, mew=2.2, label="median"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncols=3, fontsize=9, handletextpad=0.35, columnspacing=1.1)
    ax.set_title("Fiction manuals sit higher.\nPedagogy and the contrast group sit together.")
    return fig


def plot_threshold_curve(books: pd.DataFrame, role_column: str = "corpus_role",
                         roles: list[str] | None = None, role_labels: dict[str, str] | None = None):
    """Share of books at or above a density threshold, for every threshold from 0.1 to 100.

    Dotted lines mark the two example cut-offs (1 and 5 per 10,000 words).
    """
    roles = list(roles or ROLES)
    role_labels = role_labels or ROLE_LABELS
    thresholds = np.logspace(-1, 2, 121)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    # Place each label where that curve is still easy to read.
    label_x = {roles[0]: 12.0, roles[-1]: 1.6}
    for role in roles:
        values = books.loc[books[role_column] == role, "pov_per_10k"].to_numpy(dtype=float)
        n = len(values)
        if n == 0:
            continue
        hits = [(values >= t).sum() for t in thresholds]
        share = np.array([h / n for h in hits]) * 100
        bounds = np.array([wilson_interval(h, n) for h in hits]) * 100
        color = COLORS.get(role, GROUP_COLORS.get(role, INK))
        ax.fill_between(thresholds, bounds[:, 0], bounds[:, 1], color=color, alpha=0.18, linewidth=0)
        ax.plot(thresholds, share, color=color, lw=2, ls="-" if role == roles[0] else "--")
        x = label_x.get(role, 5.0)
        y = share[np.argmin(np.abs(thresholds - x))]
        ax.annotate(f"{role_labels.get(role, role)} (n = {n})", (x, y),
                    xytext=(8 if role == roles[0] else -8, 6 if role == roles[0] else -8),
                    textcoords="offset points", color=color, fontsize=10, fontweight="bold",
                    va="center", ha="left" if role == roles[0] else "right")
    for cut in (1, 5):
        ax.axvline(cut, color=GREY, lw=1, ls=":")
    ax.text(0.02, 0.06, "dotted lines: example cut-offs\n(1 and 5 per 10,000 words)",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CCCCCC"})
    ax.set_xscale("log")
    ax.set_xlim(0.1, 100)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1", "0.3", "1", "3", "10", "30", "100"])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"Threshold: at least this many {DENSITY_LABEL} (log scale)")
    ax.set_ylabel("Share of books at or above the threshold (%)")
    ax.grid(True)
    ax.text(0.99, 0.97, "shaded band: uncertainty from the number of books",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=GREY)
    ax.set_title("What share of books reaches a given density of POV terms?")
    return fig


def plot_dimension_shares(books: pd.DataFrame, role_column: str = "corpus_role",
                          roles: list[str] | None = None, role_labels: dict[str, str] | None = None):
    """Share of books with at least one term of each POV dimension, with uncertainty lines."""
    roles = list(roles or ROLES)
    role_labels = role_labels or ROLE_LABELS
    columns = dimension_columns(books)
    rows = []
    for role in roles:
        subset = books[books[role_column] == role]
        for dimension, column in columns.items():
            hits = int((subset[column] > 0).sum())
            low, high = wilson_interval(hits, len(subset)) if len(subset) else (0, 0)
            rows.append(dict(role=role, dimension=dimension, share=hits / len(subset) * 100 if len(subset) else 0,
                             low=low * 100, high=high * 100))
    table = pd.DataFrame(rows)
    order = table[table["role"] == roles[0]].sort_values("share", ascending=False)["dimension"].tolist()

    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    offsets = np.linspace(-0.16, 0.16, len(roles)) if len(roles) > 1 else [0]
    for offset, role in zip(offsets, roles):
        part = table[table["role"] == role].set_index("dimension").loc[order]
        y = np.arange(len(order)) + offset
        n = int((books[role_column] == role).sum())
        color = COLORS.get(role, GROUP_COLORS.get(role, INK))
        marker = MARKERS.get(role, GROUP_MARKERS.get(role, "o"))
        ax.errorbar(part["share"], y, xerr=[part["share"] - part["low"], part["high"] - part["share"]],
                    fmt=marker, color=color, ms=6, lw=1.5, capsize=0,
                    label=f"{role_labels.get(role, role)} (n = {n})")
    ax.set_yticks(np.arange(len(order)), [DIMENSION_LABELS[d] for d in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Share of books with at least one term of this kind (%)\nline = uncertainty from the number of books")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    ax.set_title("Which aspects of point of view appear?\n"
                 "A gap between the two marks is worth discussing; \"point of view\" in general is common in both")
    return fig


def plot_density_by_decade(books: pd.DataFrame, year_column: str = "work_year_clean", min_books: int = 20,
                           group_column: str = "corpus_role", groups: list[str] | None = None,
                           group_labels: dict[str, str] | None = None):
    """POV density by decade. Books with no year in `year_column` are left out."""
    groups = list(groups or ROLES)
    group_labels = group_labels or {**ROLE_LABELS, **PRIMARY_GROUP_LABELS}
    rng = np.random.default_rng(3)
    wanted = books[books[group_column].isin(groups)]
    dated = wanted[wanted[year_column].notna()].copy()
    left_out = int(len(wanted) - len(dated))
    dated["decade"] = (dated[year_column] // 10 * 10).astype(int)
    sizes = dated.groupby(["decade", group_column]).size()
    decades = sorted({d for (d, g), n in sizes.items() if g in groups and n >= min_books})
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    offsets = np.linspace(-0.18, 0.18, len(groups)) if len(groups) > 1 else [0.0]
    for x, decade in enumerate(decades):
        for offset, group in zip(offsets, groups):
            values = dated.loc[(dated["decade"] == decade) & (dated[group_column] == group), "pov_per_10k"]
            color = COLORS.get(group, GROUP_COLORS.get(group, INK))
            if len(values) >= min_books:
                # Reuse the strip marks, coloured by group rather than by corpus role.
                position = x + offset
                jitter = position + rng.uniform(-0.12, 0.12, len(values))
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                ax.scatter(jitter, values, s=13, color=color, alpha=0.45, linewidths=0,
                           marker=MARKERS.get(group, GROUP_MARKERS.get(group, "o")), zorder=2)
                ax.plot([position, position], [q1, q3], color=INK, lw=3.5, solid_capstyle="butt", zorder=3)
                ax.plot([position], [median], marker="_", ms=16, mew=2.2, color=INK, zorder=4)
                ax.text(position, 0.98, f"n = {len(values)}", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=8.5, color=color)
            elif len(groups) > 1:
                ax.text(x + offset, 0.5, f"fewer than {min_books}", transform=ax.get_xaxis_transform(),
                        ha="center", va="center", rotation=90, fontsize=8, color=GREY)
    ax.set_xticks(range(len(decades)), [f"{d}s" for d in decades])
    ax.set_xlim(-0.6, max(len(decades) - 0.4, 0.6))
    year_name = YEAR_COLUMN_LABELS.get(year_column, year_column)
    ax.set_xlabel(f"Decade of {year_name}")
    _density_axis(ax, "y")
    ax.grid(axis="y")
    handles = [
        Line2D([], [], marker=MARKERS.get(g, GROUP_MARKERS.get(g, "o")),
               color=COLORS.get(g, GROUP_COLORS.get(g, INK)), lw=0, ms=6,
               label=group_labels.get(g, g))
        for g in groups
    ]
    handles += [Line2D([], [], color=INK, lw=3.5, label="middle half of the books"),
                Line2D([], [], marker="_", color=INK, lw=0, ms=12, mew=2.5, label="median")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=3, fontsize=9)
    shown = ", ".join(group_labels.get(g, g) for g in groups)
    ax.set_title(f"POV terms by decade of {year_name}\n"
                 f"{shown}: {len(dated)} books with a year; {left_out} left out because they have none",
                 pad=14)
    return fig
