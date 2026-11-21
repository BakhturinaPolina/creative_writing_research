"""Orchestrate tests → baselines A/B/C → report → eval sample."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from src.common.logging_util import setup_logger
from src.common.paths import ROOT
from src.retrieval.run_meta import base_run_meta, peak_rss_mb, write_json
from src.common.paths import DATA_INTERIM


def _run(cmd: list[str], log) -> None:
    log.info("RUN: " + " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--lang", choices=["en", "de", "all"], default="all")
    parser.add_argument("--with-passages", action="store_true")
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args()
    log = setup_logger(__file__)
    t0 = time.time()

    if not args.skip_tests:
        _run(
            [sys.executable, "-m", "pytest", "tests/retrieval", "-q", "--tb=short"],
            log,
        )

    py = sys.executable
    if not args.skip_a:
        cmd = [py, "-m", "src.retrieval.keyword_baselines", "--baseline", "a", "--lang", args.lang]
        if args.with_passages:
            cmd.append("--with-passages")
        _run(cmd, log)

    cmd = [py, "-m", "src.retrieval.keyword_baselines", "--baseline", "b", "--lang", args.lang]
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
    ]
    if args.with_passages:
        cmd.append("--with-passages")
    _run(cmd, log)

    _run([py, "-m", "src.retrieval.report"], log)
    _run([py, "-m", "src.retrieval.eval_sample"], log)

    meta = base_run_meta(
        kind="run_all",
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
