"""Publication-frequency statistics (STUB).

Operates on included records only. Planned outputs:
- new works per year (first edition per work_id)
- editions per year
- both per 5-year bin, split by language and translation_status
"""
import pandas as pd


def new_works_per_period(included: pd.DataFrame, bin_years: int = 5) -> pd.DataFrame:
    raise NotImplementedError


def editions_per_period(included: pd.DataFrame, bin_years: int = 5) -> pd.DataFrame:
    raise NotImplementedError
