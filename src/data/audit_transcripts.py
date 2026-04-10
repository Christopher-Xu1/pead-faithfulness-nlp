from __future__ import annotations

import argparse

import pandas as pd

from src.data.transcript_corpus import build_audit_summary, build_record_metrics, inventory_raw_files, iter_source_records
from src.utils.io import ensure_dir, load_yaml, write_csv
from src.utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/transcript_curation.yaml")
    parser.add_argument("--raw-root", default="data/raw/transcripts")
    parser.add_argument("--audit-dir", default="data/interim/audit")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    source_ids = cfg.get("sources")
    ensure_dir(args.audit_dir)

    inventory_df = inventory_raw_files(args.raw_root)
    if source_ids:
        inventory_df = inventory_df[inventory_df["source_id"].isin(source_ids)].reset_index(drop=True)
    write_csv(inventory_df, f"{args.audit_dir}/raw_file_inventory.csv")

    metrics_rows = [build_record_metrics(record) for record in iter_source_records(args.raw_root, source_ids=source_ids)]
    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        raise RuntimeError("No transcript records were discovered. Run ingest first.")

    metrics_path = f"{args.audit_dir}/record_metrics.parquet"
    metrics_df.to_parquet(metrics_path, index=False)

    summary_df = build_audit_summary(metrics_df, inventory_df)
    write_csv(summary_df, f"{args.audit_dir}/audit_summary.csv")
    LOGGER.info("Audited %s records across %s sources.", len(metrics_df), metrics_df['source_id'].nunique())


if __name__ == "__main__":
    main()
