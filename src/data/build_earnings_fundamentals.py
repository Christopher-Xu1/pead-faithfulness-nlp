from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.io import ensure_dir, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

GLOPARDO_NUMERIC_COLUMNS = [
    "eps12mtrailing_qavg",
    "eps12mtrailing_eoq",
    "eps12mfwd_qavg",
    "eps12mfwd_eoq",
    "eps_lt",
    "peforw_qavg",
    "peforw_eoq",
]
EVENT_LEVEL_COLUMN_ALIASES = {
    "reported_eps": ["reported_eps", "actual_eps", "eps_actual", "reportedEPS", "reported_eps_gaap"],
    "estimated_eps": ["estimated_eps", "eps_estimate", "eps_expected", "consensus_eps", "estimatedEPS"],
    "eps_surprise": ["eps_surprise", "surprise", "surprise_eps"],
    "eps_surprise_pct": ["eps_surprise_pct", "surprise_percentage", "surprisePercent", "surprise_pct"],
    "reported_revenue": ["reported_revenue", "actual_revenue", "revenue_actual"],
    "estimated_revenue": ["estimated_revenue", "revenue_estimate", "revenue_expected", "consensus_revenue"],
    "revenue_surprise": ["revenue_surprise", "revenue_surprise_abs"],
    "revenue_surprise_pct": ["revenue_surprise_pct", "revenue_surprise_percentage"],
    "reported_capex": ["reported_capex", "actual_capex", "capex_actual", "capital_expenditure", "capital_expenditures"],
    "estimated_capex": ["estimated_capex", "capex_estimate", "capex_expected", "consensus_capex"],
    "capex_surprise": ["capex_surprise", "capex_surprise_abs"],
    "capex_surprise_pct": ["capex_surprise_pct", "capex_surprise_percentage"],
}
EVENT_LEVEL_OUTPUT_COLUMNS = [
    "reported_eps",
    "estimated_eps",
    "eps_surprise",
    "eps_surprise_pct",
    "eps_beat_flag",
    "eps_miss_flag",
    "eps_meet_flag",
    "eps_beat_miss",
    "reported_revenue",
    "estimated_revenue",
    "revenue_surprise",
    "revenue_surprise_pct",
    "revenue_beat_flag",
    "revenue_miss_flag",
    "revenue_meet_flag",
    "revenue_beat_miss",
    "reported_capex",
    "estimated_capex",
    "capex_surprise",
    "capex_surprise_pct",
    "capex_beat_flag",
    "capex_miss_flag",
    "capex_meet_flag",
    "capex_beat_miss",
]
EVENT_LEVEL_PASSTHROUGH_COLUMNS = [
    "source_event_date",
    "date_diff_days",
    "match_status",
    "estimated_eps_source",
    "estimated_revenue_source",
    "estimated_capex_source",
    "estimated_capex_is_proxy",
    "estimated_capex_method",
    "prior_capex_to_revenue_ratio",
    "reported_revenue_source_concept",
    "reported_capex_source_concept",
]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    out = numerator / denominator.abs()
    out = out.where(denominator.abs() > 1e-12)
    return out.replace([np.inf, -np.inf], np.nan)


def _first_present(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    available = [name for name in candidates if name in frame.columns]
    if not available:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    out = pd.to_numeric(frame[available[0]], errors="coerce")
    for name in available[1:]:
        out = out.combine_first(pd.to_numeric(frame[name], errors="coerce"))
    return out


def _series_or_na(frame: pd.DataFrame, column: str, dtype: str | None = None) -> pd.Series:
    if column in frame.columns:
        series = frame[column].copy()
    else:
        series = pd.Series(pd.NA, index=frame.index)
    if dtype is not None:
        return series.astype(dtype)
    return series


def _label_surprise(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    surprise = pd.to_numeric(series, errors="coerce")
    beat = (surprise > 0).astype("Int64")
    miss = (surprise < 0).astype("Int64")
    meet = (surprise == 0).astype("Int64")
    label = pd.Series(pd.NA, index=surprise.index, dtype="string")
    label = label.mask(surprise > 0, "beat")
    label = label.mask(surprise < 0, "miss")
    label = label.mask(surprise == 0, "meet")
    return beat, miss, meet, label


def _load_glopardo_enrichment_rows(metadata_df: pd.DataFrame, raw_root: str | Path) -> pd.DataFrame:
    subset = metadata_df[metadata_df["source_id"] == "glopardo_sp500_earnings_transcripts"].copy()
    if subset.empty:
        return pd.DataFrame(columns=["call_id"] + GLOPARDO_NUMERIC_COLUMNS)

    rows_by_file: dict[str, set[int]] = {}
    for row in subset.itertuples(index=False):
        rows_by_file.setdefault(str(row.source_file), set()).add(int(row.source_row))

    extracted_rows: list[dict[str, Any]] = []
    base_root = Path(raw_root) / "glopardo_sp500_earnings_transcripts"
    requested_columns = ["ticker", "earnings_date"] + GLOPARDO_NUMERIC_COLUMNS
    for source_file, source_rows in rows_by_file.items():
        parquet_path = base_root / source_file
        parquet_file = pq.ParquetFile(parquet_path)
        available = set(parquet_file.schema.names)
        columns = [column for column in requested_columns if column in available]
        current_row = 0
        for batch in parquet_file.iter_batches(columns=columns, batch_size=256):
            for item in batch.to_pylist():
                if current_row in source_rows:
                    item["source_file"] = source_file
                    item["source_row"] = current_row
                    extracted_rows.append(item)
                current_row += 1
                if len(extracted_rows) >= len(subset):
                    break
            if len(extracted_rows) >= len(subset):
                break

    extracted_df = pd.DataFrame(extracted_rows)
    if extracted_df.empty:
        return pd.DataFrame(columns=["call_id"] + GLOPARDO_NUMERIC_COLUMNS)

    merge_cols = subset[["call_id", "source_file", "source_row"]].copy()
    enriched = merge_cols.merge(extracted_df, on=["source_file", "source_row"], how="left")
    keep_cols = ["call_id"] + [column for column in GLOPARDO_NUMERIC_COLUMNS if column in enriched.columns]
    return enriched[keep_cols]


def _load_single_external_earnings_events(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        LOGGER.warning("External earnings events file not found: %s", csv_path)
        return None

    frame = pd.read_csv(csv_path)
    if frame.empty:
        return None
    required = {"ticker", "event_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"External earnings events file is missing required columns: {sorted(missing)}")

    out = frame.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["event_date"] = pd.to_datetime(out["event_date"]).dt.strftime("%Y-%m-%d")
    key_columns = ["ticker", "event_date"]
    if "call_id" in out.columns:
        key_columns.insert(0, "call_id")
    standardized = out[key_columns].copy()
    for output_column, aliases in EVENT_LEVEL_COLUMN_ALIASES.items():
        standardized[output_column] = _first_present(out, aliases)
    for column in EVENT_LEVEL_PASSTHROUGH_COLUMNS:
        if column in out.columns:
            standardized[column] = out[column]
    return standardized.drop_duplicates(subset=key_columns, keep="last")


def _merge_external_event_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    merge_keys = ["call_id", "ticker", "event_date"] if "call_id" in left.columns and "call_id" in right.columns else ["ticker", "event_date"]
    merged = left.merge(right, on=merge_keys, how="outer", suffixes=("", "__new"))
    for column in right.columns:
        if column in merge_keys:
            continue
        new_column = f"{column}__new"
        if new_column in merged.columns:
            merged[column] = merged[column].combine_first(merged[new_column])
            merged = merged.drop(columns=[new_column])
    return merged


def load_external_earnings_events(paths: str | Path | list[str | Path] | None) -> pd.DataFrame | None:
    if paths is None:
        return None
    if isinstance(paths, (str, Path)):
        path_list = [paths]
    else:
        path_list = list(paths)

    frames = [
        frame
        for frame in (_load_single_external_earnings_events(path) for path in path_list)
        if frame is not None and not frame.empty
    ]
    if not frames:
        return None
    out = frames[0]
    for frame in frames[1:]:
        out = _merge_external_event_frames(out, frame)
    key_columns = ["call_id", "ticker", "event_date"] if "call_id" in out.columns else ["ticker", "event_date"]
    return out.drop_duplicates(subset=key_columns, keep="last")


def build_earnings_fundamentals(
    metadata_df: pd.DataFrame,
    raw_root: str | Path,
    external_events_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = metadata_df[["call_id", "ticker", "event_date", "source_id", "source_file", "source_row"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper().str.strip()
    base["event_date"] = pd.to_datetime(base["event_date"]).dt.strftime("%Y-%m-%d")

    glopardo_df = _load_glopardo_enrichment_rows(base, raw_root)
    out = base.merge(glopardo_df, on="call_id", how="left")

    if external_events_df is not None and not external_events_df.empty:
        merge_keys = ["call_id", "ticker", "event_date"] if "call_id" in external_events_df.columns else ["ticker", "event_date"]
        out = out.merge(external_events_df, on=merge_keys, how="left")
    else:
        for column in EVENT_LEVEL_OUTPUT_COLUMNS:
            if column not in out.columns:
                if column.endswith("_beat_miss"):
                    out[column] = pd.Series(pd.NA, index=out.index, dtype="string")
                else:
                    out[column] = np.nan

    out["eps_surprise"] = pd.to_numeric(out["eps_surprise"], errors="coerce").combine_first(
        pd.to_numeric(out["reported_eps"], errors="coerce") - pd.to_numeric(out["estimated_eps"], errors="coerce")
    )
    out["eps_surprise_pct"] = pd.to_numeric(out["eps_surprise_pct"], errors="coerce").combine_first(
        _safe_ratio(out["eps_surprise"], out["estimated_eps"])
    )
    out["reported_revenue"] = pd.to_numeric(out["reported_revenue"], errors="coerce")
    out["estimated_revenue"] = pd.to_numeric(out["estimated_revenue"], errors="coerce")
    out["revenue_surprise"] = pd.to_numeric(out["revenue_surprise"], errors="coerce").combine_first(
        out["reported_revenue"] - out["estimated_revenue"]
    )
    out["revenue_surprise_pct"] = pd.to_numeric(out["revenue_surprise_pct"], errors="coerce").combine_first(
        _safe_ratio(out["revenue_surprise"], out["estimated_revenue"])
    )
    out["reported_capex"] = pd.to_numeric(out["reported_capex"], errors="coerce")
    out["estimated_capex"] = pd.to_numeric(out["estimated_capex"], errors="coerce")
    out["capex_surprise"] = pd.to_numeric(out["capex_surprise"], errors="coerce").combine_first(
        out["reported_capex"] - out["estimated_capex"]
    )
    out["capex_surprise_pct"] = pd.to_numeric(out["capex_surprise_pct"], errors="coerce").combine_first(
        _safe_ratio(out["capex_surprise"], out["estimated_capex"])
    )
    out["estimated_capex_is_proxy"] = pd.to_numeric(
        _series_or_na(out, "estimated_capex_is_proxy"), errors="coerce"
    ).astype("Int64")

    eps_beat, eps_miss, eps_meet, eps_label = _label_surprise(out["eps_surprise"])
    revenue_beat, revenue_miss, revenue_meet, revenue_label = _label_surprise(out["revenue_surprise"])
    capex_beat, capex_miss, capex_meet, capex_label = _label_surprise(out["capex_surprise"])
    out["eps_beat_flag"] = pd.to_numeric(_series_or_na(out, "eps_beat_flag"), errors="coerce").astype("Int64").combine_first(
        eps_beat
    )
    out["eps_miss_flag"] = pd.to_numeric(_series_or_na(out, "eps_miss_flag"), errors="coerce").astype("Int64").combine_first(
        eps_miss
    )
    out["eps_meet_flag"] = pd.to_numeric(_series_or_na(out, "eps_meet_flag"), errors="coerce").astype("Int64").combine_first(
        eps_meet
    )
    out["eps_beat_miss"] = _series_or_na(out, "eps_beat_miss", dtype="string").combine_first(eps_label)
    out["revenue_beat_flag"] = pd.to_numeric(_series_or_na(out, "revenue_beat_flag"), errors="coerce").astype(
        "Int64"
    ).combine_first(revenue_beat)
    out["revenue_miss_flag"] = pd.to_numeric(_series_or_na(out, "revenue_miss_flag"), errors="coerce").astype(
        "Int64"
    ).combine_first(revenue_miss)
    out["revenue_meet_flag"] = pd.to_numeric(_series_or_na(out, "revenue_meet_flag"), errors="coerce").astype(
        "Int64"
    ).combine_first(revenue_meet)
    out["revenue_beat_miss"] = _series_or_na(out, "revenue_beat_miss", dtype="string").combine_first(revenue_label)
    out["capex_beat_flag"] = pd.to_numeric(_series_or_na(out, "capex_beat_flag"), errors="coerce").astype(
        "Int64"
    ).combine_first(capex_beat)
    out["capex_miss_flag"] = pd.to_numeric(_series_or_na(out, "capex_miss_flag"), errors="coerce").astype(
        "Int64"
    ).combine_first(capex_miss)
    out["capex_meet_flag"] = pd.to_numeric(_series_or_na(out, "capex_meet_flag"), errors="coerce").astype(
        "Int64"
    ).combine_first(capex_meet)
    out["capex_beat_miss"] = _series_or_na(out, "capex_beat_miss", dtype="string").combine_first(capex_label)

    out["earnings_surprise"] = out["eps_surprise_pct"].combine_first(out["eps_surprise"])
    out["fwd_minus_trailing_eps_eoq"] = pd.to_numeric(out.get("eps12mfwd_eoq"), errors="coerce") - pd.to_numeric(
        out.get("eps12mtrailing_eoq"), errors="coerce"
    )
    out["fwd_minus_trailing_eps_qavg"] = pd.to_numeric(out.get("eps12mfwd_qavg"), errors="coerce") - pd.to_numeric(
        out.get("eps12mtrailing_qavg"), errors="coerce"
    )
    out["fwd_vs_trailing_eps_growth_eoq"] = _safe_ratio(out["fwd_minus_trailing_eps_eoq"], out["eps12mtrailing_eoq"])
    out["fwd_vs_trailing_eps_growth_qavg"] = _safe_ratio(
        out["fwd_minus_trailing_eps_qavg"], out["eps12mtrailing_qavg"]
    )

    ordered_columns = [
        "call_id",
        "ticker",
        "event_date",
        "source_id",
        "reported_eps",
        "estimated_eps",
        "eps_surprise",
        "eps_surprise_pct",
        "eps_beat_flag",
        "eps_miss_flag",
        "eps_meet_flag",
        "eps_beat_miss",
        "reported_revenue",
        "estimated_revenue",
        "revenue_surprise",
        "revenue_surprise_pct",
        "revenue_beat_flag",
        "revenue_miss_flag",
        "revenue_meet_flag",
        "revenue_beat_miss",
        "reported_capex",
        "estimated_capex",
        "capex_surprise",
        "capex_surprise_pct",
        "capex_beat_flag",
        "capex_miss_flag",
        "capex_meet_flag",
        "capex_beat_miss",
        "source_event_date",
        "date_diff_days",
        "match_status",
        "estimated_eps_source",
        "estimated_revenue_source",
        "estimated_capex_source",
        "estimated_capex_is_proxy",
        "estimated_capex_method",
        "prior_capex_to_revenue_ratio",
        "reported_revenue_source_concept",
        "reported_capex_source_concept",
        "earnings_surprise",
        *GLOPARDO_NUMERIC_COLUMNS,
        "fwd_minus_trailing_eps_eoq",
        "fwd_minus_trailing_eps_qavg",
        "fwd_vs_trailing_eps_growth_eoq",
        "fwd_vs_trailing_eps_growth_qavg",
    ]
    out = out[[column for column in ordered_columns if column in out.columns]].copy()

    summary = {
        "rows": int(len(out)),
        "eps_surprise_coverage": float(out["eps_surprise"].notna().mean()),
        "eps_surprise_pct_coverage": float(out["eps_surprise_pct"].notna().mean()),
        "earnings_surprise_coverage": float(out["earnings_surprise"].notna().mean()),
        "reported_eps_coverage": float(out["reported_eps"].notna().mean()),
        "estimated_eps_coverage": float(out["estimated_eps"].notna().mean()),
        "reported_revenue_coverage": float(out["reported_revenue"].notna().mean()),
        "estimated_revenue_coverage": float(out["estimated_revenue"].notna().mean()),
        "revenue_surprise_coverage": float(out["revenue_surprise"].notna().mean()),
        "reported_capex_coverage": float(out["reported_capex"].notna().mean()),
        "estimated_capex_coverage": float(out["estimated_capex"].notna().mean()),
        "capex_surprise_coverage": float(out["capex_surprise"].notna().mean()),
        "estimated_capex_proxy_coverage": float(out["estimated_capex_is_proxy"].fillna(0).astype(bool).mean()),
        "eps_beat_miss_coverage": float(out["eps_beat_miss"].notna().mean()),
        "revenue_beat_miss_coverage": float(out["revenue_beat_miss"].notna().mean()),
        "capex_beat_miss_coverage": float(out["capex_beat_miss"].notna().mean()),
        "glopardo_forward_eps_coverage": float(out["eps12mfwd_eoq"].notna().mean()) if "eps12mfwd_eoq" in out else 0.0,
    }
    return out, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-input", default="data/raw/metadata/call_metadata.csv")
    parser.add_argument("--raw-root", default="data/raw/transcripts")
    parser.add_argument("--external-events-input", default=None)
    parser.add_argument("--output-dir", default="data/external/earnings_fundamentals")
    args = parser.parse_args()

    metadata_df = pd.read_csv(args.metadata_input)
    external_events_df = load_external_earnings_events(args.external_events_input)
    fundamentals_df, summary = build_earnings_fundamentals(
        metadata_df=metadata_df,
        raw_root=args.raw_root,
        external_events_df=external_events_df,
    )

    out_dir = ensure_dir(args.output_dir)
    write_csv(fundamentals_df, out_dir / "earnings_fundamentals.csv")
    save_json(summary, out_dir / "summary.json")
    LOGGER.info("Saved earnings fundamentals to %s", out_dir)


if __name__ == "__main__":
    main()
