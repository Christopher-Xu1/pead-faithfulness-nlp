from __future__ import annotations

import fnmatch
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi

from src.utils.io import ensure_dir


@dataclass(frozen=True)
class TranscriptSource:
    source_id: str
    label: str
    source_type: str
    repo_id: str | None = None
    repo_type: str | None = None
    allow_patterns: tuple[str, ...] = ()
    kaggle_slug: str | None = None
    source_url: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["allow_patterns"] = list(self.allow_patterns)
        return out


SOURCE_REGISTRY: dict[str, TranscriptSource] = {
    "glopardo_sp500_earnings_transcripts": TranscriptSource(
        source_id="glopardo_sp500_earnings_transcripts",
        label="glopardo/sp500-earnings-transcripts",
        source_type="huggingface",
        repo_id="glopardo/sp500-earnings-transcripts",
        repo_type="dataset",
        allow_patterns=("README.md", "data/*.parquet", "transcript_coverage.csv"),
        source_url="https://huggingface.co/datasets/glopardo/sp500-earnings-transcripts",
        notes="Broad S&P 500 transcript corpus with strong Mag7 coverage.",
    ),
    "bose345_sp500_earnings_transcripts": TranscriptSource(
        source_id="bose345_sp500_earnings_transcripts",
        label="Bose345/sp500_earnings_transcripts",
        source_type="huggingface",
        repo_id="Bose345/sp500_earnings_transcripts",
        repo_type="dataset",
        allow_patterns=("README.md", "parquet_files/*.parquet"),
        source_url="https://huggingface.co/datasets/Bose345/sp500_earnings_transcripts",
        notes="Largest open full-transcript source in the current shortlist.",
    ),
    "lamini_earnings_calls_qa": TranscriptSource(
        source_id="lamini_earnings_calls_qa",
        label="lamini/earnings-calls-qa",
        source_type="huggingface",
        repo_id="lamini/earnings-calls-qa",
        repo_type="dataset",
        allow_patterns=("README.md", "filtered_predictions.jsonl"),
        source_url="https://huggingface.co/datasets/lamini/earnings-calls-qa",
        notes="Large QA-oriented JSONL source; useful as a supplemental raw corpus.",
    ),
    "jlh_ibm_earnings_call": TranscriptSource(
        source_id="jlh_ibm_earnings_call",
        label="jlh-ibm/earnings_call",
        source_type="huggingface",
        repo_id="jlh-ibm/earnings_call",
        repo_type="dataset",
        allow_patterns=("README.md", "data/transcripts/**/*.txt"),
        source_url="https://huggingface.co/datasets/jlh-ibm/earnings_call",
        notes="Small academic transcript set with explicit file-based call text.",
    ),
    "kaggle_meta_earnings_call_qa": TranscriptSource(
        source_id="kaggle_meta_earnings_call_qa",
        label="devaangbarthwal/meta-earnings-call-q-and-a-dataset",
        source_type="kaggle",
        kaggle_slug="devaangbarthwal/meta-earnings-call-q-and-a-dataset",
        source_url="https://www.kaggle.com/datasets/devaangbarthwal/meta-earnings-call-q-and-a-dataset",
        notes="Optional Meta-only Kaggle supplement. Requires Kaggle credentials.",
    ),
}


def resolve_sources(source_ids: list[str] | None = None) -> list[TranscriptSource]:
    if source_ids is None:
        return list(SOURCE_REGISTRY.values())
    missing = [source_id for source_id in source_ids if source_id not in SOURCE_REGISTRY]
    if missing:
        raise ValueError(f"Unknown transcript sources: {missing}")
    return [SOURCE_REGISTRY[source_id] for source_id in source_ids]


def _download_huggingface_source(source: TranscriptSource, destination: Path) -> dict[str, Any]:
    ensure_dir(destination)
    api = HfApi()
    info = api.dataset_info(source.repo_id, files_metadata=True)
    matched_files = [
        sibling
        for sibling in info.siblings
        if any(fnmatch.fnmatch(sibling.rfilename, pattern) for pattern in source.allow_patterns)
    ]
    if not matched_files:
        return {"status": "failed", "message": "No files matched the configured download patterns."}

    downloaded = 0
    skipped = 0
    for sibling in matched_files:
        target = destination / sibling.rfilename
        ensure_dir(target.parent)
        expected_size = getattr(sibling, "size", None)
        if target.exists() and expected_size is not None and target.stat().st_size == expected_size:
            skipped += 1
            continue

        url = f"https://huggingface.co/datasets/{source.repo_id}/resolve/main/{sibling.rfilename}"
        tmp_target = target.with_name(f"{target.name}.part")
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = requests.get(url, stream=True, timeout=(30, 300))
                response.raise_for_status()
                with tmp_target.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                tmp_target.replace(target)
                last_error = None
                break
            except requests.RequestException as exc:
                last_error = exc
                if tmp_target.exists():
                    tmp_target.unlink()
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)
        if last_error is not None:
            raise last_error
        downloaded += 1

    return {
        "status": "downloaded",
        "message": f"Downloaded {downloaded} files and reused {skipped} existing files.",
    }


def _download_kaggle_source(source: TranscriptSource, destination: Path) -> dict[str, Any]:
    kaggle_bin = shutil.which("kaggle")
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    config_creds = Path.home() / ".config" / "kaggle" / "kaggle.json"

    if kaggle_bin is None:
        return {"status": "skipped", "message": "kaggle CLI is not installed in this environment."}
    if not kaggle_creds.exists() and not config_creds.exists():
        return {"status": "skipped", "message": "Kaggle credentials are missing; source left un-downloaded."}

    ensure_dir(destination)
    cmd = [
        kaggle_bin,
        "datasets",
        "download",
        "-d",
        source.kaggle_slug,
        "-p",
        str(destination),
        "--unzip",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "Kaggle download failed."
        return {"status": "failed", "message": message}
    return {"status": "downloaded", "message": completed.stdout.strip() or "Kaggle dataset downloaded."}


def download_source(source: TranscriptSource, raw_root: str | Path) -> dict[str, Any]:
    raw_root = ensure_dir(raw_root)
    destination = raw_root / source.source_id
    if source.source_type == "huggingface":
        result = _download_huggingface_source(source, destination)
    elif source.source_type == "kaggle":
        result = _download_kaggle_source(source, destination)
    else:
        raise ValueError(f"Unsupported source_type={source.source_type!r}")

    result.update(
        {
            "source_id": source.source_id,
            "label": source.label,
            "source_type": source.source_type,
            "destination": str(destination),
            "source_url": source.source_url,
        }
    )
    return result
