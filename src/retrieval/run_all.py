"""Orchestrate tests → baselines → report → eval sample → passage manifest."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.common.logging_util import setup_logger
from src.common.paths import DATA_INTERIM, ROOT
from src.retrieval.run_meta import base_run_meta, peak_rss_mb, write_json


def _run(cmd: list[str], log) -> None:
    log.info("RUN: " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--skip-algo-fix", action="store_true")
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    parser.add_argument("--with-passages", action="store_true")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument(
        "--pipeline",
        choices=["v03", "legacy_v02"],
        default="v03",
        help="v03: algo-fix + v0.3 lexical/BM25 (default). legacy_v02: old A/B/C stems.",
    )
    args = parser.parse_args()
    log = setup_logger(__file__)
    t0 = time.time()

    if not args.skip_tests:
        _run(
            [sys.executable, "-m", "pytest", "tests/retrieval", "-q", "--tb=short"],
            log,
        )

    py = sys.executable

    if args.pipeline == "legacy_v02":
        if not args.skip_a:
            cmd = [py, "-m", "src.retrieval.keyword_baselines", "--baseline", "a", "--lang", args.lang]
            if args.with_passages:
                cmd.append("--with-passages")
            _run(cmd, log)
        cmd = [
            py,
            "-m",
            "src.retrieval.keyword_baselines",
            "--baseline",
            "b",
            "--lang",
            args.lang,
            "--lexicon-version",
            "v02",
            "--overwrite-v02",
        ]
        if args.with_passages:
            cmd.append("--with-passages")
        _run(cmd, log)
        cmd = [
            py,
            "-m",
            "src.retrieval.bm25_passages",
            "--lang",
            args.lang,
            "--top-k",
            str(args.top_k),
            "--lexicon-version",
            "v02",
        ]
        if args.with_passages:
            cmd.append("--with-passages")
        _run(cmd, log)
        _run([py, "-m", "src.retrieval.report"], log)
        _run([py, "-m", "src.retrieval.eval_sample", "--version", "v02"], log)
    else:
        # Intermediate: v0.2 vocabulary + corrected algorithm
        if not args.skip_algo_fix:
            cmd = [
                py,
                "-m",
                "src.retrieval.keyword_baselines",
                "--baseline",
                "b",
                "--lang",
                args.lang,
                "--lexicon-version",
                "v02_algo_fix",
            ]
            if args.with_passages:
                cmd.append("--with-passages")
            _run(cmd, log)

        # Uncapped v0.3 lexical
        cmd = [
            py,
            "-m",
            "src.retrieval.keyword_baselines",
            "--baseline",
            "b",
            "--lang",
            args.lang,
            "--lexicon-version",
            "v03",
        ]
        if args.with_passages:
            cmd.append("--with-passages")
        _run(cmd, log)

        # v0.3 BM25 + BM25-only
        cmd = [
            py,
            "-m",
            "src.retrieval.bm25_passages",
            "--lang",
            args.lang,
            "--top-k",
            str(args.top_k),
            "--lexicon-version",
            "v03",
        ]
        if args.with_passages:
            cmd.append("--with-passages")
        _run(cmd, log)

        _run([py, "-m", "src.retrieval.report", "--version", "v03"], log)
        # Preserve frozen v0.3 sample; generate probability + diversity v0.4 designs
        # with lexicon-dev held-out books excluded by default.
        held = DATA_INTERIM / "retrieval" / "lexicon_dev_held_out_books_v03.txt"
        eval_cmd = [py, "-m", "src.retrieval.eval_sample", "--version", "v04"]
        if held.is_file():
            eval_cmd.extend(["--held-out-file", str(held)])
        _run(eval_cmd, log)

        if not args.skip_manifest:
            _run(
                [py, "-m", "src.retrieval.passage_manifest", "--lang", args.lang],
                log,
            )
            _run(
                [py, "-m", "src.retrieval.corpus_snapshot", "--lang", args.lang],
                log,
            )

    meta = base_run_meta(
        kind="run_all",
        pipeline=args.pipeline,
        duration_sec=round(time.time() - t0, 1),
        peak_rss_mb=round(peak_rss_mb(), 1),
        lang=args.lang,
        top_k=args.top_k,
        with_passages=args.with_passages,
    )
    write_json(DATA_INTERIM / "retrieval" / "run_all.meta.json", meta)
    log.info(f"done in {meta['duration_sec']}s peak_rss={meta['peak_rss_mb']}MB")
    log.info("Log finished.")


if __name__ == "__main__":
    main()
