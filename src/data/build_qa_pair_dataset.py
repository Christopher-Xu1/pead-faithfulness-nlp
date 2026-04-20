from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir, load_yaml, save_json, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)

SECTOR_MAP = {
    "AAPL": "information_technology",
    "AMZN": "consumer_discretionary",
    "GOOGL": "communication_services",
    "META": "communication_services",
    "MSFT": "information_technology",
    "NVDA": "information_technology",
    "TSLA": "consumer_discretionary",
}

QUESTION_MIN_CHARS = 40
SHORT_ACK_RE = re.compile(r"^(okay|ok|great|thanks|thank you|all right|right)\b", re.IGNORECASE)
QUESTION_STOP_PHRASES = (
    "next question",
    "next caller",
    "could we have the next question",
    "we'll take our next question",
    "operator, next question",
)
DEFAULT_QA_FILTERS: dict[str, dict[str, Any]] = {
    "broad": {
        "profile": "broad",
        "min_answer_chars": 0,
        "require_management_turn": False,
        "drop_answer_with_question_mark": False,
        "drop_operator_prompt_leak": False,
        "drop_analyst_only_answers": False,
        "max_answer_turns": None,
    },
    "strict": {
        "profile": "strict",
        "min_answer_chars": 120,
        "require_management_turn": True,
        "drop_answer_with_question_mark": True,
        "drop_operator_prompt_leak": True,
        "drop_analyst_only_answers": True,
        "max_answer_turns": None,
    },
}


def _as_float(row: Any, column: str) -> float:
    if not hasattr(row, column):
        return float("nan")
    value = getattr(row, column)
    return float(value) if pd.notna(value) else float("nan")


def _as_int(row: Any, column: str, default: int = 0) -> int:
    if not hasattr(row, column):
        return default
    value = getattr(row, column)
    return int(value) if pd.notna(value) else default


def _as_str(row: Any, column: str, default: str = "unknown") -> str:
    if not hasattr(row, column):
        return default
    value = getattr(row, column)
    return str(value) if pd.notna(value) and str(value).strip() else default


def _slug(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "unknown"


def _normalize_role(role: object) -> str:
    return str(role).strip().lower()


def _contains_question_stop_phrase(text: object) -> bool:
    lowered = str(text).strip().lower()
    return any(phrase in lowered for phrase in QUESTION_STOP_PHRASES)


def _is_question_turn(role: object, text: object) -> bool:
    normalized_role = _normalize_role(role)
    cleaned = str(text).strip()
    if normalized_role == "operator":
        return False
    if "analyst" not in normalized_role:
        return False
    if "?" not in cleaned:
        return False
    if len(cleaned) < QUESTION_MIN_CHARS:
        return False
    if _contains_question_stop_phrase(cleaned):
        return False
    return True


def _is_short_ack(text: object) -> bool:
    cleaned = str(text).strip()
    return len(cleaned) < 30 and "?" not in cleaned and bool(SHORT_ACK_RE.search(cleaned))


def _split_role_tokens(role_text: object) -> list[str]:
    tokens = [token.strip().lower() for token in str(role_text).split("|")]
    return [token for token in tokens if token]


def resolve_pair_filter_config(pair_filter_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_QA_FILTERS["broad"])
    if pair_filter_config:
        profile = str(pair_filter_config.get("profile", "broad")).strip().lower()
        if profile not in DEFAULT_QA_FILTERS:
            raise ValueError(f"Unsupported QA pair filter profile: {profile!r}")
        config = dict(DEFAULT_QA_FILTERS[profile])
        for key, value in pair_filter_config.items():
            if key == "profile":
                continue
            config[key] = value
    config["profile"] = str(config.get("profile", "broad")).strip().lower()
    if config["profile"] not in DEFAULT_QA_FILTERS:
        raise ValueError(f"Unsupported QA pair filter profile: {config['profile']!r}")
    return config


def annotate_pair_quality(pair_df: pd.DataFrame) -> pd.DataFrame:
    out = pair_df.copy()
    answer_roles = out["answer_roles"].fillna("")
    answer_text = out["answer_text"].fillna("")
    role_tokens = answer_roles.map(_split_role_tokens)
    out["answer_has_management_role"] = answer_roles.str.contains(r"\bmanagement\b", regex=True)
    out["answer_has_analyst_role"] = answer_roles.str.contains(r"\banalyst\b", regex=True)
    out["answer_has_mixed_roles"] = out["answer_has_management_role"] & out["answer_has_analyst_role"]
    out["answer_is_analyst_only"] = out["answer_has_analyst_role"] & ~out["answer_has_management_role"]
    out["answer_management_turn_count"] = role_tokens.map(lambda roles: sum(role == "management" for role in roles)).astype(int)
    out["answer_analyst_turn_count"] = role_tokens.map(lambda roles: sum(role == "analyst" for role in roles)).astype(int)
    out["answer_role_switch_count"] = role_tokens.map(
        lambda roles: sum(curr != prev for prev, curr in zip(roles, roles[1:])) if len(roles) >= 2 else 0
    ).astype(int)
    out["answer_management_turn_share"] = out["answer_management_turn_count"] / out["num_answer_turns"].clip(lower=1)
    out["answer_analyst_turn_share"] = out["answer_analyst_turn_count"] / out["num_answer_turns"].clip(lower=1)
    out["answer_starts_with_management"] = role_tokens.map(lambda roles: bool(roles) and roles[0] == "management")
    out["answer_ends_with_management"] = role_tokens.map(lambda roles: bool(roles) and roles[-1] == "management")
    out["answer_starts_with_analyst"] = role_tokens.map(lambda roles: bool(roles) and roles[0] == "analyst")
    out["answer_ends_with_analyst"] = role_tokens.map(lambda roles: bool(roles) and roles[-1] == "analyst")
    out["answer_contains_question_mark"] = answer_text.str.contains(r"\?")
    out["answer_contains_operator_prompt"] = answer_text.map(_contains_question_stop_phrase)
    out["answer_span_gt6_turns"] = out["num_answer_turns"] > 6
    out["answer_is_short_strict"] = out["answer_char_len"] < int(DEFAULT_QA_FILTERS["strict"]["min_answer_chars"])
    return out


def apply_pair_filters(
    pair_df: pd.DataFrame,
    pair_filter_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = resolve_pair_filter_config(pair_filter_config)
    keep_mask = pd.Series(True, index=pair_df.index)
    drop_counts: dict[str, int] = {}

    min_answer_chars = int(config.get("min_answer_chars") or 0)
    if min_answer_chars > 0:
        failed = pair_df["answer_char_len"] < min_answer_chars
        drop_counts["min_answer_chars"] = int(failed.sum())
        keep_mask &= ~failed

    if bool(config.get("require_management_turn", False)):
        failed = ~pair_df["answer_has_management_role"]
        drop_counts["require_management_turn"] = int(failed.sum())
        keep_mask &= ~failed

    if bool(config.get("drop_analyst_only_answers", False)):
        failed = pair_df["answer_is_analyst_only"]
        drop_counts["drop_analyst_only_answers"] = int(failed.sum())
        keep_mask &= ~failed

    if bool(config.get("drop_answer_with_question_mark", False)):
        failed = pair_df["answer_contains_question_mark"]
        drop_counts["drop_answer_with_question_mark"] = int(failed.sum())
        keep_mask &= ~failed

    if bool(config.get("drop_operator_prompt_leak", False)):
        failed = pair_df["answer_contains_operator_prompt"]
        drop_counts["drop_operator_prompt_leak"] = int(failed.sum())
        keep_mask &= ~failed

    max_answer_turns = config.get("max_answer_turns")
    if max_answer_turns is not None:
        failed = pair_df["num_answer_turns"] > int(max_answer_turns)
        drop_counts["max_answer_turns"] = int(failed.sum())
        keep_mask &= ~failed

    filtered = pair_df[keep_mask].copy().reset_index(drop=True)
    summary = {
        "pair_filter_profile": config["profile"],
        "pair_filter_config": config,
        "pair_rows_before_filtering": int(len(pair_df)),
        "pair_rows_after_filtering": int(len(filtered)),
        "pairs_removed_by_filtering": int(len(pair_df) - len(filtered)),
        "pair_retention_rate": float(len(filtered) / len(pair_df)) if len(pair_df) else 0.0,
        "drop_counts_if_applied_individually": drop_counts,
    }
    return filtered, summary


def extract_qa_pairs(parsed_df: pd.DataFrame) -> pd.DataFrame:
    pairs: list[dict[str, object]] = []
    qa_df = parsed_df.copy()
    qa_df = qa_df[qa_df["section"].str.contains("q", case=False, na=False)].copy()
    qa_df = qa_df.sort_values(["call_id", "turn_id"]).reset_index(drop=True)

    for call_id, group in qa_df.groupby("call_id", sort=False):
        turns = group.reset_index(drop=True)
        pair_index = 0
        i = 0
        while i < len(turns):
            turn = turns.iloc[i]
            if not _is_question_turn(turn["speaker_role"], turn["text"]):
                i += 1
                continue

            question_text = str(turn["text"]).strip()
            answer_texts: list[str] = []
            answer_roles: list[str] = []
            answer_turn_ids: list[int] = []

            j = i + 1
            while j < len(turns):
                next_turn = turns.iloc[j]
                next_role = _normalize_role(next_turn["speaker_role"])
                next_text = str(next_turn["text"]).strip()

                if next_role == "operator":
                    j += 1
                    continue
                if _is_question_turn(next_turn["speaker_role"], next_turn["text"]):
                    break
                if answer_texts and _is_short_ack(next_text):
                    j += 1
                    continue

                answer_texts.append(next_text)
                answer_roles.append(next_role)
                answer_turn_ids.append(int(next_turn["turn_id"]))
                j += 1

            answer_text = " ".join(answer_texts).strip()
            if answer_text:
                pairs.append(
                    {
                        "call_id": call_id,
                        "ticker": turn["ticker"],
                        "event_date": turn["event_date"],
                        "pair_index": pair_index,
                        "question_turn_id": int(turn["turn_id"]),
                        "answer_start_turn_id": answer_turn_ids[0],
                        "answer_end_turn_id": answer_turn_ids[-1],
                        "question_text": question_text,
                        "answer_text": answer_text,
                        "pair_text": f"Question: {question_text}\nAnswer: {answer_text}",
                        "question_char_len": len(question_text),
                        "answer_char_len": len(answer_text),
                        "num_answer_turns": len(answer_turn_ids),
                        "answer_roles": "|".join(answer_roles),
                    }
                )
                pair_index += 1
            i = j

    return pd.DataFrame(pairs)


def _window(values: np.ndarray, end_idx: int, window_size: int) -> np.ndarray:
    if end_idx < 0:
        return np.array([], dtype=float)
    start_idx = max(0, end_idx - window_size + 1)
    return np.asarray(values[start_idx : end_idx + 1], dtype=float)


def _forward_window(values: np.ndarray, start_idx: int, window_size: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if start_idx < 0 or start_idx >= len(values) or window_size <= 0:
        return np.array([], dtype=float)
    end_idx = min(len(values), start_idx + window_size)
    return np.asarray(values[start_idx:end_idx], dtype=float)


def _safe_std(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.std(values))


def _safe_beta(stock_returns: np.ndarray, market_returns: np.ndarray) -> float:
    stock_returns = np.asarray(stock_returns, dtype=float)
    market_returns = np.asarray(market_returns, dtype=float)
    mask = np.isfinite(stock_returns) & np.isfinite(market_returns)
    if mask.sum() < 5:
        return float("nan")
    stock_returns = stock_returns[mask]
    market_returns = market_returns[mask]
    market_var = float(np.var(market_returns))
    if market_var <= 0:
        return float("nan")
    cov = float(np.cov(stock_returns, market_returns, ddof=0)[0, 1])
    return cov / market_var


def _safe_cumulative_return(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.prod(1.0 + values) - 1.0)


def build_call_controls(
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    qa_summary_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    label_config_path: str | Path,
    earnings_fundamentals_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    label_cfg = load_yaml(label_config_path)
    event_lag_days = int(label_cfg.get("event_lag_days", 1))

    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    market = market_df.copy()
    market["date"] = pd.to_datetime(market["date"])
    market = market.sort_values("date").reset_index(drop=True)

    labels = labels_df.copy()
    labels["event_date"] = pd.to_datetime(labels["event_date"])

    qa_summary = qa_summary_df.copy()
    qa_summary["event_date"] = pd.to_datetime(qa_summary["event_date"])

    metadata = metadata_df.copy()
    metadata["event_date"] = pd.to_datetime(metadata["event_date"])

    earnings = None
    if earnings_fundamentals_df is not None and not earnings_fundamentals_df.empty:
        earnings = earnings_fundamentals_df.copy()
        earnings["event_date"] = pd.to_datetime(earnings["event_date"])

    base = metadata.merge(
        qa_summary[
            [
                "call_id",
                "ticker",
                "event_date",
                "num_questions",
                "num_qa_turns",
            ]
        ],
        on=["call_id", "ticker", "event_date"],
        how="left",
    ).merge(
        labels[["call_id", "ticker", "event_date", "car_horizon", "label"]],
        on=["call_id", "ticker", "event_date"],
        how="inner",
    )
    if earnings is not None:
        earnings_feature_cols = [
            column
            for column in earnings.columns
            if column not in {"call_id", "ticker", "event_date", "source_id"}
        ]
        base = base.merge(
            earnings[["call_id", "ticker", "event_date", *earnings_feature_cols]],
            on=["call_id", "ticker", "event_date"],
            how="left",
        )

    rows: list[dict[str, object]] = []
    for row in base.itertuples(index=False):
        ticker_prices = prices[prices["ticker"] == row.ticker]
        merged = ticker_prices.merge(market, on="date", how="inner").sort_values("date").reset_index(drop=True)
        target_date = pd.Timestamp(row.event_date) + pd.Timedelta(days=event_lag_days)
        candidate = merged[merged["date"] >= target_date]
        event_idx = int(candidate.index[0]) - 1 if not candidate.empty else -1

        stock_returns = merged["return"].to_numpy(dtype=float)
        market_returns = merged["market_return"].to_numpy(dtype=float)

        stock_5 = _window(stock_returns, event_idx, 5)
        stock_20 = _window(stock_returns, event_idx, 20)
        stock_60 = _window(stock_returns, event_idx, 60)
        market_20 = _window(market_returns, event_idx, 20)
        market_60 = _window(market_returns, event_idx, 60)
        pre_event_week_returns = _window(stock_returns, event_idx - 1, 5)
        post_event_3d_returns = _forward_window(stock_returns, event_idx + 1, 3)

        prior_momentum_5d = float(np.nansum(stock_5)) if len(stock_5) else float("nan")
        prior_momentum_20d = float(np.nansum(stock_20)) if len(stock_20) else float("nan")
        prior_momentum_60d = float(np.nansum(stock_60)) if len(stock_60) else float("nan")
        market_momentum_20d = float(np.nansum(market_20)) if len(market_20) else float("nan")
        market_momentum_60d = float(np.nansum(market_60)) if len(market_60) else float("nan")

        rows.append(
            {
                "call_id": row.call_id,
                "ticker": row.ticker,
                "event_date": pd.Timestamp(row.event_date).strftime("%Y-%m-%d"),
                "source_id": row.source_id,
                "company": row.company,
                "year": row.year,
                "quarter": row.quarter,
                "quality_score": row.quality_score,
                "soft_quality_flags": row.soft_quality_flags,
                "num_questions": row.num_questions,
                "num_qa_turns": row.num_qa_turns,
                "sector": _slug(_as_str(row, "universe_sector", SECTOR_MAP.get(row.ticker, "unknown"))),
                "universe_sector": _slug(_as_str(row, "universe_sector", SECTOR_MAP.get(row.ticker, "unknown"))),
                "universe_industry": _slug(_as_str(row, "universe_industry")),
                "universe_included_by": _slug(_as_str(row, "universe_included_by")),
                "snapshot_market_cap_usd": _as_float(row, "snapshot_market_cap_usd"),
                "snapshot_log_market_cap": _as_float(row, "snapshot_log_market_cap"),
                "snapshot_market_cap_percentile": _as_float(row, "snapshot_market_cap_percentile"),
                "universe_calls_in_gold_corpus": _as_int(row, "universe_calls_in_gold_corpus", default=0),
                "hist_market_cap": _as_float(row, "hist_market_cap"),
                "hist_log_market_cap": _as_float(row, "hist_log_market_cap"),
                "hist_market_cap_percentile": _as_float(row, "hist_market_cap_percentile"),
                "hist_market_cap_close": _as_float(row, "hist_market_cap_close"),
                "hist_market_cap_shares_outstanding": _as_float(row, "hist_market_cap_shares_outstanding"),
                "hist_market_cap_price_lag_days": _as_float(row, "hist_market_cap_price_lag_days"),
                "hist_market_cap_shares_staleness_days": _as_float(row, "hist_market_cap_shares_staleness_days"),
                "car_horizon": float(row.car_horizon),
                "label": int(row.label),
                "prior_momentum_5d": prior_momentum_5d,
                "prior_momentum_20d": prior_momentum_20d,
                "prior_momentum_60d": prior_momentum_60d,
                "pre_event_return_5d": _safe_cumulative_return(pre_event_week_returns),
                "post_event_return_3d": _safe_cumulative_return(post_event_3d_returns),
                "relative_momentum_20d": prior_momentum_20d - market_momentum_20d
                if np.isfinite(prior_momentum_20d) and np.isfinite(market_momentum_20d)
                else float("nan"),
                "relative_momentum_60d": prior_momentum_60d - market_momentum_60d
                if np.isfinite(prior_momentum_60d) and np.isfinite(market_momentum_60d)
                else float("nan"),
                "volatility_20d": _safe_std(stock_20),
                "volatility_60d": _safe_std(stock_60),
                "market_volatility_20d": _safe_std(market_20),
                "beta_60d": _safe_beta(stock_60, market_60),
                "reported_eps": float(row.reported_eps) if hasattr(row, "reported_eps") and pd.notna(row.reported_eps) else float("nan"),
                "estimated_eps": float(row.estimated_eps)
                if hasattr(row, "estimated_eps") and pd.notna(row.estimated_eps)
                else float("nan"),
                "eps_surprise": float(row.eps_surprise) if hasattr(row, "eps_surprise") and pd.notna(row.eps_surprise) else float("nan"),
                "eps_surprise_pct": float(row.eps_surprise_pct)
                if hasattr(row, "eps_surprise_pct") and pd.notna(row.eps_surprise_pct)
                else float("nan"),
                "eps_beat_flag": int(row.eps_beat_flag) if hasattr(row, "eps_beat_flag") and pd.notna(row.eps_beat_flag) else -1,
                "eps_miss_flag": int(row.eps_miss_flag) if hasattr(row, "eps_miss_flag") and pd.notna(row.eps_miss_flag) else -1,
                "eps_meet_flag": int(row.eps_meet_flag) if hasattr(row, "eps_meet_flag") and pd.notna(row.eps_meet_flag) else -1,
                "eps_beat_miss": str(row.eps_beat_miss)
                if hasattr(row, "eps_beat_miss") and pd.notna(row.eps_beat_miss)
                else "unknown",
                "reported_revenue": float(row.reported_revenue)
                if hasattr(row, "reported_revenue") and pd.notna(row.reported_revenue)
                else float("nan"),
                "estimated_revenue": float(row.estimated_revenue)
                if hasattr(row, "estimated_revenue") and pd.notna(row.estimated_revenue)
                else float("nan"),
                "revenue_surprise": float(row.revenue_surprise)
                if hasattr(row, "revenue_surprise") and pd.notna(row.revenue_surprise)
                else float("nan"),
                "revenue_surprise_pct": float(row.revenue_surprise_pct)
                if hasattr(row, "revenue_surprise_pct") and pd.notna(row.revenue_surprise_pct)
                else float("nan"),
                "revenue_beat_flag": int(row.revenue_beat_flag)
                if hasattr(row, "revenue_beat_flag") and pd.notna(row.revenue_beat_flag)
                else -1,
                "revenue_miss_flag": int(row.revenue_miss_flag)
                if hasattr(row, "revenue_miss_flag") and pd.notna(row.revenue_miss_flag)
                else -1,
                "revenue_meet_flag": int(row.revenue_meet_flag)
                if hasattr(row, "revenue_meet_flag") and pd.notna(row.revenue_meet_flag)
                else -1,
                "revenue_beat_miss": str(row.revenue_beat_miss)
                if hasattr(row, "revenue_beat_miss") and pd.notna(row.revenue_beat_miss)
                else "unknown",
                "reported_capex": _as_float(row, "reported_capex"),
                "estimated_capex": _as_float(row, "estimated_capex"),
                "capex_surprise": _as_float(row, "capex_surprise"),
                "capex_surprise_pct": _as_float(row, "capex_surprise_pct"),
                "capex_beat_flag": _as_int(row, "capex_beat_flag", default=-1),
                "capex_miss_flag": _as_int(row, "capex_miss_flag", default=-1),
                "capex_meet_flag": _as_int(row, "capex_meet_flag", default=-1),
                "capex_beat_miss": _as_str(row, "capex_beat_miss"),
                "estimated_capex_is_proxy": _as_int(row, "estimated_capex_is_proxy", default=0),
                "prior_capex_to_revenue_ratio": _as_float(row, "prior_capex_to_revenue_ratio"),
                "earnings_surprise": float(row.earnings_surprise)
                if hasattr(row, "earnings_surprise") and pd.notna(row.earnings_surprise)
                else float("nan"),
                "eps12mtrailing_qavg": float(row.eps12mtrailing_qavg)
                if hasattr(row, "eps12mtrailing_qavg") and pd.notna(row.eps12mtrailing_qavg)
                else float("nan"),
                "eps12mtrailing_eoq": float(row.eps12mtrailing_eoq)
                if hasattr(row, "eps12mtrailing_eoq") and pd.notna(row.eps12mtrailing_eoq)
                else float("nan"),
                "eps12mfwd_qavg": float(row.eps12mfwd_qavg)
                if hasattr(row, "eps12mfwd_qavg") and pd.notna(row.eps12mfwd_qavg)
                else float("nan"),
                "eps12mfwd_eoq": float(row.eps12mfwd_eoq)
                if hasattr(row, "eps12mfwd_eoq") and pd.notna(row.eps12mfwd_eoq)
                else float("nan"),
                "eps_lt": float(row.eps_lt) if hasattr(row, "eps_lt") and pd.notna(row.eps_lt) else float("nan"),
                "peforw_qavg": float(row.peforw_qavg)
                if hasattr(row, "peforw_qavg") and pd.notna(row.peforw_qavg)
                else float("nan"),
                "peforw_eoq": float(row.peforw_eoq)
                if hasattr(row, "peforw_eoq") and pd.notna(row.peforw_eoq)
                else float("nan"),
                "fwd_minus_trailing_eps_eoq": float(row.fwd_minus_trailing_eps_eoq)
                if hasattr(row, "fwd_minus_trailing_eps_eoq") and pd.notna(row.fwd_minus_trailing_eps_eoq)
                else float("nan"),
                "fwd_minus_trailing_eps_qavg": float(row.fwd_minus_trailing_eps_qavg)
                if hasattr(row, "fwd_minus_trailing_eps_qavg") and pd.notna(row.fwd_minus_trailing_eps_qavg)
                else float("nan"),
                "fwd_vs_trailing_eps_growth_eoq": float(row.fwd_vs_trailing_eps_growth_eoq)
                if hasattr(row, "fwd_vs_trailing_eps_growth_eoq") and pd.notna(row.fwd_vs_trailing_eps_growth_eoq)
                else float("nan"),
                "fwd_vs_trailing_eps_growth_qavg": float(row.fwd_vs_trailing_eps_growth_qavg)
                if hasattr(row, "fwd_vs_trailing_eps_growth_qavg") and pd.notna(row.fwd_vs_trailing_eps_growth_qavg)
                else float("nan"),
            }
        )

    out = pd.DataFrame(rows)
    return add_ticker_frequency_features(out)


def add_ticker_frequency_features(call_features_df: pd.DataFrame) -> pd.DataFrame:
    df = call_features_df.copy()
    if df.empty:
        return df
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values(["ticker", "event_date", "call_id"]).reset_index(drop=True)
    df["ticker_prior_call_count"] = df.groupby("ticker").cumcount()
    df["ticker_prev_event_date"] = df.groupby("ticker")["event_date"].shift(1)
    df["ticker_days_since_prev_call"] = (df["event_date"] - df["ticker_prev_event_date"]).dt.days

    prior_365_counts: list[int] = []
    mean_prior_gaps: list[float] = []
    for _, group in df.groupby("ticker", sort=False):
        dates = group["event_date"].tolist()
        for idx, current_date in enumerate(dates):
            prior_dates = dates[:idx]
            prior_365_counts.append(int(sum((current_date - prior_date).days <= 365 for prior_date in prior_dates)))
            if len(prior_dates) < 2:
                mean_prior_gaps.append(float("nan"))
            else:
                gaps = np.diff(pd.to_datetime(prior_dates).to_numpy()).astype("timedelta64[D]").astype(float)
                mean_prior_gaps.append(float(np.nanmean(gaps)))
    df["ticker_prior_call_count_365d"] = prior_365_counts
    df["ticker_mean_prior_call_gap_days"] = mean_prior_gaps
    df = df.drop(columns=["ticker_prev_event_date"]).sort_values(["event_date", "call_id"]).reset_index(drop=True)
    df["event_date"] = df["event_date"].dt.strftime("%Y-%m-%d")
    return df


def build_qa_pair_dataset(
    parsed_df: pd.DataFrame,
    qa_summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    market_df: pd.DataFrame,
    label_config_path: str | Path,
    earnings_fundamentals_df: pd.DataFrame | None = None,
    pair_filter_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    pair_df = extract_qa_pairs(parsed_df)
    if pair_df.empty:
        raise ValueError("No QA pairs were extracted from parsed calls")
    pair_df = annotate_pair_quality(pair_df)
    raw_pair_count = int(len(pair_df))
    raw_call_count = int(pair_df["call_id"].nunique())
    pair_df, filter_summary = apply_pair_filters(pair_df, pair_filter_config=pair_filter_config)
    if pair_df.empty:
        raise ValueError("QA pair filtering removed every extracted pair; relax the filter config")

    call_features_df = build_call_controls(
        metadata_df=metadata_df,
        labels_df=labels_df,
        qa_summary_df=qa_summary_df,
        prices_df=prices_df,
        market_df=market_df,
        label_config_path=label_config_path,
        earnings_fundamentals_df=earnings_fundamentals_df,
    )
    call_features_df = call_features_df.sort_values(["event_date", "call_id"]).reset_index(drop=True)

    pair_df["event_date"] = pd.to_datetime(pair_df["event_date"]).dt.strftime("%Y-%m-%d")
    call_features_df["event_date"] = pd.to_datetime(call_features_df["event_date"]).dt.strftime("%Y-%m-%d")

    pair_df = pair_df.merge(
        call_features_df,
        on=["call_id", "ticker", "event_date"],
        how="inner",
    )
    pair_counts = pair_df.groupby("call_id").size().rename("num_pairs").reset_index()
    role_aggregate_df = (
        pair_df.groupby("call_id", as_index=False)
        .agg(
            pair_management_turn_share_mean=("answer_management_turn_share", "mean"),
            pair_management_turn_share_max=("answer_management_turn_share", "max"),
            pair_analyst_turn_share_mean=("answer_analyst_turn_share", "mean"),
            pair_role_switch_count_mean=("answer_role_switch_count", "mean"),
            pair_role_switch_count_max=("answer_role_switch_count", "max"),
            pair_starts_with_management_rate=("answer_starts_with_management", "mean"),
            pair_ends_with_management_rate=("answer_ends_with_management", "mean"),
            pair_starts_with_analyst_rate=("answer_starts_with_analyst", "mean"),
            pair_ends_with_analyst_rate=("answer_ends_with_analyst", "mean"),
        )
    )
    pair_df = pair_df.merge(pair_counts, on="call_id", how="left")
    call_features_df = call_features_df.merge(pair_counts, on="call_id", how="left")
    call_features_df = call_features_df.merge(role_aggregate_df, on="call_id", how="left")
    call_features_df["num_pairs"] = call_features_df["num_pairs"].fillna(0).astype(int)

    summary = {
        "pair_rows": int(len(pair_df)),
        "calls_with_pairs": int(pair_df["call_id"].nunique()),
        "pair_rows_before_filtering": raw_pair_count,
        "calls_with_pairs_before_filtering": raw_call_count,
        "mean_pairs_per_call": float(pair_counts["num_pairs"].mean()) if len(pair_counts) else 0.0,
        "median_pairs_per_call": float(pair_counts["num_pairs"].median()) if len(pair_counts) else 0.0,
        "mean_question_chars": float(pair_df["question_char_len"].mean()),
        "mean_answer_chars": float(pair_df["answer_char_len"].mean()),
        "pre_event_return_5d_coverage": float(call_features_df["pre_event_return_5d"].notna().mean()),
        "post_event_return_3d_coverage": float(call_features_df["post_event_return_3d"].notna().mean()),
        "snapshot_market_cap_coverage": float(call_features_df["snapshot_market_cap_usd"].notna().mean())
        if "snapshot_market_cap_usd" in call_features_df
        else 0.0,
        "hist_market_cap_coverage": float(call_features_df["hist_market_cap"].notna().mean())
        if "hist_market_cap" in call_features_df
        else 0.0,
        "ticker_days_since_prev_call_coverage": float(call_features_df["ticker_days_since_prev_call"].notna().mean()),
        "ticker_prior_call_count_mean": float(call_features_df["ticker_prior_call_count"].mean()),
        "ticker_prior_call_count_365d_mean": float(call_features_df["ticker_prior_call_count_365d"].mean()),
        "positive_rate": float(call_features_df["label"].mean()) if len(call_features_df) else 0.0,
        "earnings_surprise_coverage": float(call_features_df["earnings_surprise"].notna().mean()),
        "reported_eps_coverage": float(call_features_df["reported_eps"].notna().mean()),
        "estimated_eps_coverage": float(call_features_df["estimated_eps"].notna().mean()),
        "eps_surprise_coverage": float(call_features_df["eps_surprise"].notna().mean()),
        "eps_surprise_pct_coverage": float(call_features_df["eps_surprise_pct"].notna().mean()),
        "reported_revenue_coverage": float(call_features_df["reported_revenue"].notna().mean()),
        "estimated_revenue_coverage": float(call_features_df["estimated_revenue"].notna().mean()),
        "revenue_surprise_coverage": float(call_features_df["revenue_surprise"].notna().mean()),
        "reported_capex_coverage": float(call_features_df["reported_capex"].notna().mean()),
        "estimated_capex_coverage": float(call_features_df["estimated_capex"].notna().mean()),
        "capex_surprise_coverage": float(call_features_df["capex_surprise"].notna().mean()),
        "estimated_capex_proxy_coverage": float(call_features_df["estimated_capex_is_proxy"].fillna(0).astype(bool).mean()),
        "eps_beat_miss_coverage": float((call_features_df["eps_beat_miss"] != "unknown").mean()),
        "revenue_beat_miss_coverage": float((call_features_df["revenue_beat_miss"] != "unknown").mean()),
        "capex_beat_miss_coverage": float((call_features_df["capex_beat_miss"] != "unknown").mean()),
        "glopardo_forward_eps_coverage": float(call_features_df["eps12mfwd_eoq"].notna().mean()),
        "answer_has_management_role_rate": float(pair_df["answer_has_management_role"].mean()),
        "answer_has_analyst_role_rate": float(pair_df["answer_has_analyst_role"].mean()),
        "answer_has_mixed_roles_rate": float(pair_df["answer_has_mixed_roles"].mean()),
        "answer_is_analyst_only_rate": float(pair_df["answer_is_analyst_only"].mean()),
        "answer_management_turn_share_mean": float(pair_df["answer_management_turn_share"].mean()),
        "answer_analyst_turn_share_mean": float(pair_df["answer_analyst_turn_share"].mean()),
        "answer_role_switch_count_mean": float(pair_df["answer_role_switch_count"].mean()),
        "answer_starts_with_management_rate": float(pair_df["answer_starts_with_management"].mean()),
        "answer_ends_with_management_rate": float(pair_df["answer_ends_with_management"].mean()),
        "answer_contains_question_mark_rate": float(pair_df["answer_contains_question_mark"].mean()),
        "answer_contains_operator_prompt_rate": float(pair_df["answer_contains_operator_prompt"].mean()),
        "answer_span_gt6_turns_rate": float(pair_df["answer_span_gt6_turns"].mean()),
        "answer_is_short_strict_rate": float(pair_df["answer_is_short_strict"].mean()),
    }
    summary.update(filter_summary)
    return pair_df, call_features_df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-input", default="data/interim/parsed_calls/parsed_calls.csv")
    parser.add_argument("--qa-summary-input", default="data/interim/qa_only/qa_dataset.csv")
    parser.add_argument("--metadata-input", default="data/raw/metadata/call_metadata.csv")
    parser.add_argument("--labels-input", default="data/interim/labels/pead_labels.csv")
    parser.add_argument("--prices-input", default="data/raw/prices/daily_returns.csv")
    parser.add_argument("--market-input", default="data/external/market_index/sp500_returns.csv")
    parser.add_argument(
        "--earnings-fundamentals-input",
        default="data/external/earnings_fundamentals/earnings_fundamentals.csv",
    )
    parser.add_argument("--label-config", default="configs/data/pead_20d.yaml")
    parser.add_argument("--output-dir", default="outputs/datasets/qa_pairs_mag7")
    parser.add_argument("--cleaning-profile", choices=sorted(DEFAULT_QA_FILTERS.keys()), default="broad")
    parser.add_argument("--min-answer-chars", type=int, default=None)
    parser.add_argument("--require-management-turn", action="store_true")
    parser.add_argument("--drop-answer-with-question-mark", action="store_true")
    parser.add_argument("--drop-operator-prompt-leak", action="store_true")
    parser.add_argument("--drop-analyst-only-answers", action="store_true")
    parser.add_argument("--max-answer-turns", type=int, default=None)
    args = parser.parse_args()

    parsed_df = pd.read_csv(args.parsed_input)
    qa_summary_df = pd.read_csv(args.qa_summary_input)
    metadata_df = pd.read_csv(args.metadata_input)
    labels_df = pd.read_csv(args.labels_input)
    prices_df = pd.read_csv(args.prices_input)
    market_df = pd.read_csv(args.market_input)
    earnings_fundamentals_df = (
        pd.read_csv(args.earnings_fundamentals_input) if Path(args.earnings_fundamentals_input).exists() else None
    )
    pair_filter_config: dict[str, Any] = {"profile": args.cleaning_profile}
    if args.min_answer_chars is not None:
        pair_filter_config["min_answer_chars"] = args.min_answer_chars
    if args.require_management_turn:
        pair_filter_config["require_management_turn"] = True
    if args.drop_answer_with_question_mark:
        pair_filter_config["drop_answer_with_question_mark"] = True
    if args.drop_operator_prompt_leak:
        pair_filter_config["drop_operator_prompt_leak"] = True
    if args.drop_analyst_only_answers:
        pair_filter_config["drop_analyst_only_answers"] = True
    if args.max_answer_turns is not None:
        pair_filter_config["max_answer_turns"] = args.max_answer_turns

    pair_df, call_features_df, summary = build_qa_pair_dataset(
        parsed_df=parsed_df,
        qa_summary_df=qa_summary_df,
        metadata_df=metadata_df,
        labels_df=labels_df,
        prices_df=prices_df,
        market_df=market_df,
        label_config_path=args.label_config,
        earnings_fundamentals_df=earnings_fundamentals_df,
        pair_filter_config=pair_filter_config,
    )

    out_dir = ensure_dir(args.output_dir)
    write_csv(pair_df, out_dir / "qa_pair_dataset.csv")
    write_csv(call_features_df, out_dir / "call_features.csv")
    save_json(summary, out_dir / "summary.json")
    LOGGER.info("Saved QA pair dataset to %s", out_dir)


if __name__ == "__main__":
    main()
