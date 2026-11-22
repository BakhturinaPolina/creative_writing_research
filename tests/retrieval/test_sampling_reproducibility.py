"""Cross-process sampling reproducibility (PYTHONHASHSEED-independent)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.common.paths import ROOT
from src.retrieval.sampling import (
    ReservoirSample,
    StratifiedReservoirs,
    derive_stratum_seed,
)


def test_derive_stratum_seed_stable():
    a = derive_stratum_seed(20261123, "en::fiction_manual")
    b = derive_stratum_seed(20261123, "en::fiction_manual")
    c = derive_stratum_seed(20261123, "de::fiction_manual")
    assert a == b
    assert a != c
    assert 0 <= a < 2**31 - 1


def test_stratified_meta_records_weights():
    s = StratifiedReservoirs(k_per_stratum=5, seed=99)
    for i in range(100):
        s.consider("en::g", f"p{i}")
    meta = s.meta()
    m = meta["en::g"]
    assert m["N_h"] == 100
    assert m["n_h"] == 5
    assert m["inclusion_probability"] == pytest.approx(0.05)
    assert m["sampling_weight"] == pytest.approx(20.0)
    assert m["seed"] == derive_stratum_seed(99, "en::g")


def test_reservoir_identical_under_different_pythonhashseed():
    """Spawn two interpreters with different PYTHONHASHSEED; compare samples."""
    script = textwrap.dedent(
        """
        import json, sys
        from src.retrieval.sampling import StratifiedReservoirs
        s = StratifiedReservoirs(k_per_stratum=10, seed=20261123)
        for i in range(500):
            s.consider("en::fiction", f"book{i%40}:p{i}")
            s.consider("de::fiction", f"de{i%20}:p{i}")
        ids = sorted(x for x in s.all_items())
        print(json.dumps({"ids": ids, "meta": s.meta()}))
        """
    )
    env_base = {**os.environ, "PYTHONPATH": str(ROOT)}
    results = []
    for hs in ("0", "1", "42", "random"):
        env = {**env_base, "PYTHONHASHSEED": hs}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        import json

        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    first = results[0]["ids"]
    for r in results[1:]:
        assert r["ids"] == first
    # Meta seeds must also match
    for r in results[1:]:
        assert r["meta"] == results[0]["meta"]


def test_reservoir_k_equals_n_gives_exact_pi():
    r = ReservoirSample(k=7, seed=3)
    r.fill_from(range(200))
    assert r.inclusion_probability == pytest.approx(7 / 200)
    assert r.sampling_weight == pytest.approx(200 / 7)
