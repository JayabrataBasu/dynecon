"""FRED download and parsing (standard library HTTP; pandas needed for the result).

Economics note:
Macro data from the St. Louis Fed's FRED database is the usual input for filtering and
for matching model moments to the data.

The download endpoint (``fredgraph.csv``) is FRED's public CSV export; a live download of
``GDPC1`` was checked manually on 2026-11-04. Tests never touch the network.
"""

from __future__ import annotations

import io
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dynecon.core.errors import DyneconError, ValidationError

__all__ = ["fetch_fred", "parse_fred_csv"]

_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def parse_fred_csv(text: str) -> Any:
    """Parse FRED CSV text (``DATE,SERIES`` header, ``.`` for missing) into a pandas Series."""
    import pandas as pd

    df = pd.read_csv(io.StringIO(text), na_values=["."], parse_dates=[0])
    if df.shape[1] != 2:
        raise ValidationError(f"expected two columns (date, value), got {list(df.columns)}")
    series = df.set_index(df.columns[0])[df.columns[1]].astype("float64")
    series.index.name = "date"
    series.name = str(df.columns[1])
    return series.sort_index()


def fetch_fred(
    series_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path | None = None,
    timeout: float = 30.0,
) -> Any:
    """Download one FRED series as a ``pandas.Series`` (cached under ``cache_dir`` if given)."""
    if not series_id.replace("_", "").isalnum():
        raise ValidationError(f"invalid FRED series id {series_id!r}")
    cache = None if cache_dir is None else Path(cache_dir) / f"{series_id}_{start}_{end}.csv"
    if cache is not None and cache.exists():
        return parse_fred_csv(cache.read_text(encoding="utf-8"))
    query = {"id": series_id}
    if start:
        query["cosd"] = start
    if end:
        query["coed"] = end
    url = f"{_URL}?{urllib.parse.urlencode(query)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DyneconError(f"could not download FRED series {series_id!r}: {exc}") from exc
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(text, encoding="utf-8")
    return parse_fred_csv(text)
