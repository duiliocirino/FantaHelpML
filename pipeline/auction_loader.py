"""Canonical auction data loader for Stage 2A (price model training).

Every auction source is normalized to one record schema:

    role, name, team, price

plus per-file metadata:

    source_file, source, league, auction_season, format_tag

Supported files (looked up in ``data/raw/{SEASON}/`` and prior-season dirs):

    Rose_*.xlsx                  local multi-league auction exports;
                                 Italian/English header row auto-detected
    auction_{league}_{fmt}.csv   external sources (e.g. website parser output),
                                 already in the canonical schema;
                                 ``fmt = {credits}_{players}`` (e.g. ``1000_10``)
    Quotazioni_*.xlsx            per-season FVM source (see ``load_season_fvm``);
                                 used to pair each auction with the FVM of its
                                 own season

Auction semantics: a file stored in ``data/raw/{t}/`` is the most recent completed
auction *before* season ``t`` starts (the t-1 season auction) — the primary training
input for season ``t``. See ``docs/phase3_stage2a2b_plan.md`` (section A3).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Canonical record schema (the 4 columns every auction source must provide)
RECORD_COLUMNS = ["role", "name", "team", "price"]
# Full unified schema returned by load_auctions()
UNIFIED_COLUMNS = RECORD_COLUMNS + [
    "source_file", "source", "league", "auction_season", "format_tag"
]

VALID_ROLES = {"P", "D", "C", "A"}

# Tokens used to auto-detect the header row inside Rose xlsx files
_HEADER_TOKENS = {"role", "ruolo", "name", "calciatore", "team", "squadra", "price", "costo"}
# Header cell (lower-cased) -> canonical column name
_HEADER_MAP = {
    "role": "role", "ruolo": "role",
    "name": "name", "calciatore": "name",
    "team": "team", "squadra": "team",
    "price": "price", "costo": "price",
}

# auction_fantaclub_1000_10.csv -> league="fantaclub", fmt="1000_10"
_AUCTION_CSV_RE = re.compile(r"^auction_(?P<league>.+)_(?P<fmt>\d{3,4}_\d{1,2})\.csv$")


def prev_season(season: str) -> str:
    """Return the season before the given one: ``'25-26' -> '24-25'``."""
    a, b = season.split("-")
    return f"{int(a) - 1}-{int(b) - 1}"

# Column tokens for Quotazioni files (FVM source per season)
_QUOT_NAME_TOKENS = {"nome", "name", "calciatore"}
_QUOT_ROLE_TOKENS = {"r", "role", "ruolo"}


def load_season_fvm(season: str, raw_root: str | Path = "data/raw") -> pd.DataFrame:
    """Load the per-player FVM map for a season from its Quotazioni file.

    Reads ``data/raw/{season}/Quotazioni_*.xlsx`` (header row auto-detected:
    some files have a title row before the header). Returns a DataFrame with
    columns ``name_key, role, fvm`` — the FVM contemporaneous with that
    season, used to pair auction records with the right market values.
    """
    d = Path(raw_root) / season
    files = sorted(d.glob("Quotazioni_*.xlsx"))
    if not files:
        raise ValueError(f"No Quotazioni file found for season {season} under {d}")
    path = files[0]
    head = pd.read_excel(path, header=None, nrows=3)
    header_row = None
    for i in range(len(head)):
        row = {str(v).strip().lower() for v in head.iloc[i].tolist() if isinstance(v, str)}
        if "fvm" in row and (row & _QUOT_NAME_TOKENS) and (row & _QUOT_ROLE_TOKENS):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not detect header row in {path}")
    raw = pd.read_excel(path, header=header_row)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    name_col = next(c for c in raw.columns if c in _QUOT_NAME_TOKENS)
    role_col = next(c for c in raw.columns if c in _QUOT_ROLE_TOKENS)
    out = pd.DataFrame({
        "name_key": raw[name_col].astype(str).str.strip().str.lower(),
        "role": raw[role_col].astype(str).str.strip(),
        "fvm": pd.to_numeric(raw["fvm"], errors="coerce"),
    }).dropna(subset=["fvm"])
    out = out[out["role"].isin(VALID_ROLES)]
    out["fvm"] = out["fvm"].astype(int)
    return out.reset_index(drop=True)


def _clean_records(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical cleaning rules to a raw auction table.

    Rules: drop rows with missing name/price, strip whitespace, remove asterisks
    from names, keep only valid roles, coerce price to int.
    """
    df = df.dropna(subset=["name", "price"]).copy()
    df["role"] = df["role"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip().str.replace("*", "", regex=False)
    df["team"] = df["team"].astype(str).str.strip()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["role"].isin(VALID_ROLES)].dropna(subset=["price"])
    df["price"] = df["price"].astype(int)
    return df[RECORD_COLUMNS]


def _detect_header_row(path: Path, max_rows: int = 3) -> int:
    """Find the 0-based header row inside a Rose xlsx by matching known tokens.

    Rose exports may have a junk first row (e.g. a leading team name), so the
    header is not assumed to be row 0.
    """
    head = pd.read_excel(path, header=None, nrows=max_rows)
    for i in range(len(head)):
        row = {str(v).strip().lower() for v in head.iloc[i].tolist() if isinstance(v, str)}
        if len(row & _HEADER_TOKENS) >= 3:
            return i
    raise ValueError(f"Could not detect header row in {path}")


def load_rose_xlsx(path: Path, format_tag: str, auction_season: str) -> pd.DataFrame:
    """Load one Rose xlsx export into the unified schema, tagged with metadata."""
    header_row = _detect_header_row(path)
    raw = pd.read_excel(path, header=header_row)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw = raw.rename(columns=_HEADER_MAP)

    recs = _clean_records(raw).reset_index(drop=True)
    recs["source_file"] = path.name
    recs["source"] = "rose"
    recs["league"] = path.stem.removeprefix("Rose_")
    recs["auction_season"] = auction_season
    recs["format_tag"] = format_tag
    return recs[UNIFIED_COLUMNS]


def load_auction_csv(path: Path, auction_season: str) -> pd.DataFrame:
    """Load one canonical auction CSV (external source) with filename-derived metadata."""
    m = _AUCTION_CSV_RE.match(path.name)
    if not m:
        raise ValueError(
            f"Unexpected auction CSV name: {path.name} "
            "(expected auction_<league>_<credits>_<players>.csv)"
        )
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    missing = set(RECORD_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing canonical columns {sorted(missing)}")

    recs = _clean_records(raw).reset_index(drop=True)
    recs["source_file"] = path.name
    recs["source"] = "website"
    recs["league"] = m.group("league")
    recs["auction_season"] = auction_season
    recs["format_tag"] = m.group("fmt")
    return recs[UNIFIED_COLUMNS]


def load_auctions(
    season: str,
    raw_root: str | Path = "data/raw",
    rose_dirs: tuple[str, ...] = ("current", "prior"),
    rose_format: str = "800_8",
) -> pd.DataFrame:
    """Load all auction files for a season from current + prior auction directories.

    Args:
        season: target season, e.g. ``'25-26'``. ``data/raw/{season}`` holds the
            t-1 season auction (the primary training input for season ``t``).
        raw_root: root directory containing per-season folders.
        rose_dirs: which directories to load, in recency order
            (``'current'`` and/or ``'prior'``).
        rose_format: default ``{credits}_{players}`` format tag for Rose xlsx files.

    Returns:
        Unified DataFrame with columns: ``role, name, team, price, source_file,
        source, league, auction_season, format_tag``.
    """
    raw_root = Path(raw_root)
    frames: list[pd.DataFrame] = []
    for kind in rose_dirs:
        if kind == "current":
            season_dir, auction_season = season, prev_season(season)
        elif kind == "prior":
            season_dir, auction_season = prev_season(season), prev_season(prev_season(season))
        else:
            raise ValueError(f"Unknown rose_dirs entry: {kind!r}")
        d = raw_root / season_dir
        if not d.is_dir():
            continue  # prior-season data may not exist yet
        for p in sorted(d.glob("Rose_*.xlsx")):
            frames.append(load_rose_xlsx(p, format_tag=rose_format, auction_season=auction_season))
        for p in sorted(d.glob("auction_*.csv")):
            frames.append(load_auction_csv(p, auction_season=auction_season))
    if not frames:
        raise ValueError(f"No auction files found for season {season} under {raw_root}")
    return pd.concat(frames, ignore_index=True)
