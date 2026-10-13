"""Two-stage deduplication (STUB).

Input: query-hit table (one row per record x query). Source provenance must be preserved.
1. Exact: ISBN, then catalogue identifiers (DNB id, OCLC number) -> edition_id.
2. Fuzzy: rapidfuzz on normalized title, same normalized author, plausible year range -> work_id.
Different editions of one work share a work_id but keep distinct edition_ids.
"""
import pandas as pd


def assign_edition_ids(hits: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError


def assign_work_ids(editions: pd.DataFrame, title_threshold: int = 92) -> pd.DataFrame:
    raise NotImplementedError
