import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BASE_COLUMNS = [
    "품목코드",
    "품목명",
    "구분",
    "담당부서",
    "규격",
    "단위",
    "Lot.No",
]

COLUMN_ALIASES = {
    "품목코드": r"품목\s*코드",
    "품목명": r"품목\s*명",
    "구분": r"구\s*분",
    "담당부서": r"담당\s*부서",
    "규격": r"규\s*격",
    "단위": r"단\s*위",
    "Lot.No": r"lot|lot\.\s*no|lot\s*no",
}

PERIOD_LABELS = [
    "M",
    "M-1",
    "M-2",
    "M-3",
    "M-4",
    "M-5",
    "M-6",
    "M-7",
    "M-8",
    "M-9",
    "M-10",
    "M-11",
    "12개월이상",
]

PERIOD_TO_MONTHS = {
    "M": 0,
    "M-1": 1,
    "M-2": 2,
    "M-3": 3,
    "M-4": 4,
    "M-5": 5,
    "M-6": 6,
    "M-7": 7,
    "M-8": 8,
    "M-9": 9,
    "M-10": 10,
    "M-11": 11,
    "12개월이상": 12,
}


def _find_header_row(file_path: Path, expected_cols: List[str], max_scan: int = 6) -> int:
    preview = pd.read_excel(file_path, header=None, nrows=max_scan)
    for idx in range(len(preview)):
        row = preview.iloc[idx].astype(str).fillna("").tolist()
        hits = sum(1 for col in expected_cols if any(col in cell for cell in row))
        if hits >= 2:
            return idx
    return 0


def _normalize_columns(columns: List[str]) -> List[str]:
    normalized = []
    for col in columns:
        if col is None:
            normalized.append("")
            continue
        col_str = str(col).replace("\n", " ").strip()
        col_str = re.sub(r"\s+", " ", col_str)
        normalized.append(col_str)
    return normalized


def _map_base_columns(columns: List[str]) -> List[str]:
    mapped = []
    for col in columns:
        mapped_col = col
        for target, pattern in COLUMN_ALIASES.items():
            if re.search(pattern, col, flags=re.IGNORECASE):
                mapped_col = target
                break
        mapped.append(mapped_col)
    return mapped


def _rename_monthly_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [c for c in df.columns if c in BASE_COLUMNS]
    remaining = [c for c in df.columns if c not in base_cols]

    rename_map: Dict[str, str] = {}
    for i in range(0, len(remaining), 2):
        period_idx = i // 2
        period_label = (
            PERIOD_LABELS[period_idx] if period_idx < len(PERIOD_LABELS) else f"M-{period_idx}"
        )
        qty_col = remaining[i]
        rename_map[qty_col] = f"{period_label}_수량"
        if i + 1 < len(remaining):
            amt_col = remaining[i + 1]
            rename_map[amt_col] = f"{period_label}_금액"

    df = df.rename(columns=rename_map)
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c.endswith("_수량") or c.endswith("_금액")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _aging_bucket_from_row(row: pd.Series, amount_cols: List[Tuple[int, str]]) -> str:
    max_month = None
    for months, col in amount_cols:
        if row.get(col, 0) > 0:
            if max_month is None or months > max_month:
                max_month = months

    if max_month is None:
        return "미상"
    if max_month >= 12:
        return "12개월+"
    if max_month >= 6:
        return "6~12개월"
    if max_month >= 3:
        return "3~6개월"
    return "3개월 미만"


def load_inventory_data(file_path: Path) -> pd.DataFrame:
    header_row = _find_header_row(file_path, BASE_COLUMNS)
    df = pd.read_excel(file_path, header=header_row)
    df.columns = _normalize_columns(df.columns)
    df.columns = _map_base_columns(df.columns)

    df = df.dropna(how="all")

    df = _rename_monthly_columns(df)
    df = _coerce_numeric(df)

    amount_cols = []
    for period, months in PERIOD_TO_MONTHS.items():
        col = f"{period}_금액"
        if col in df.columns:
            amount_cols.append((months, col))

    df["총재고금액"] = 0
    for _, col in amount_cols:
        df["총재고금액"] += df[col]

    if "12개월이상_금액" in df.columns:
        df["12개월+_금액"] = df["12개월이상_금액"]
    else:
        df["12개월+_금액"] = 0

    df["에이징"] = df.apply(lambda row: _aging_bucket_from_row(row, amount_cols), axis=1)

    return df


def build_monthly_long(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in BASE_COLUMNS if c in df.columns]
    value_cols = [c for c in df.columns if re.match(r".+_(수량|금액)$", c)]
    if not value_cols:
        return pd.DataFrame()

    melted = df.melt(id_vars=id_cols, value_vars=value_cols, var_name="period_type", value_name="value")
    period_type = melted["period_type"].str.rsplit("_", n=1, expand=True)
    melted["period"] = period_type[0]
    melted["type"] = period_type[1]

    pivoted = (
        melted.pivot_table(index=id_cols + ["period"], columns="type", values="value", aggfunc="sum")
        .reset_index()
        .rename(columns={"금액": "금액", "수량": "수량"})
    )

    return pivoted
