"""Utility script to build a retraining payload from the FD001 dataset and
submit it to the API retraining endpoint.

Usage (powershell):
    python scripts/test_retrain_api.py --base-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests

# Ensure the repository root is on the import path so we can reuse app utilities.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.utils.cmapss_loader import (  # noqa: E402
    get_all_engine_ids,
    get_sensor_columns,
    load_rul_values,
    load_test_dataset,
)


def _row_to_datapoint(row: Dict[str, object], sensor_cols: Iterable[str]) -> Dict[str, object]:
    """Convert a dataframe row dict into the API datapoint structure."""
    datapoint = {
        "cycle": int(row["cycle"]),
        "setting_1": float(row["setting_1"]),
        "setting_2": float(row["setting_2"]),
        "setting_3": float(row["setting_3"]),
    }
    for sensor in sensor_cols:
        value = row.get(sensor)
        if pd.isna(value):
            datapoint[sensor] = None
        else:
            datapoint[sensor] = float(value)
    return datapoint


def _build_engine_payload(
    df: pd.DataFrame,
    engine_id: int,
    sensor_cols: List[str],
) -> List[Dict[str, object]]:
    engine_df = df[df["unit"] == engine_id].sort_values("cycle")
    columns = ["cycle", "setting_1", "setting_2", "setting_3", *sensor_cols]
    records = engine_df[columns].replace({pd.NA: None}).to_dict(orient="records")
    return [_row_to_datapoint(record, sensor_cols) for record in records]


def _build_retraining_payload(
    df: pd.DataFrame,
    rul_map: Dict[int, float],
    sensor_cols: List[str],
    engine_ids: List[int],
    *,
    include_regression: bool,
    include_classification: bool,
    partition: str,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    settings = get_settings()
    regression_samples: List[Dict[str, object]] = []
    classification_samples: List[Dict[str, object]] = []

    for engine_id in engine_ids:
        series_payload = _build_engine_payload(df, engine_id, sensor_cols)
        target_rul = float(rul_map[engine_id])
        if include_regression:
            regression_samples.append(
                {
                    "engine_id": str(engine_id),
                    "data": series_payload,
                    "target_rul": target_rul,
                }
            )
        if include_classification:
            label = 1 if target_rul <= settings.FAILURE_THRESHOLD else 0
            classification_samples.append(
                {
                    "engine_id": str(engine_id),
                    "data": series_payload,
                    "label": label,
                }
            )

    dataset_payload = {
        "partition": partition,
        "metadata": metadata or {},
        "regression_samples": regression_samples if include_regression else None,
        "classification_samples": classification_samples if include_classification else None,
    }

    request_payload = {
        "job_id": None,
        "retrain_regression": include_regression,
        "retrain_classification": include_classification,
        "dataset": dataset_payload,
    }
    return request_payload


def _post_retraining_request(base_url: str, payload: Dict[str, object]) -> Dict[str, object]:
    endpoint = f"{base_url.rstrip('/')}/api/v1/models/retrain"
    response = requests.post(endpoint, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _poll_job_status(
    base_url: str,
    job_id: str,
    *,
    poll_interval: float,
    timeout: float,
) -> Dict[str, object]:
    endpoint = f"{base_url.rstrip('/')}/api/v1/models/retrain/{job_id}"
    deadline = time.time() + timeout
    last_status: Optional[str] = None

    while True:
        response = requests.get(endpoint, timeout=15)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status")
        if status != last_status:
            progress = payload.get("progress")
            message = payload.get("progress_message")
            print(f"Status: {status} | progress={progress} | message={message}")
            last_status = status
        if status in {"completed", "failed"}:
            return payload
        if time.time() >= deadline:
            raise TimeoutError(f"Polling timed out after {timeout} seconds")
        time.sleep(poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger model retraining with FD001 samples")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--dataset-dir", default="datasets", help="Path to C-MAPSS dataset directory")
    parser.add_argument("--max-engines", type=int, default=5, help="Number of engines to include in the payload")
    parser.add_argument("--partition", default="fd001-test", help="Partition label to store with the dataset")
    parser.add_argument("--skip-regression", action="store_true", help="Do not include regression samples")
    parser.add_argument("--skip-classification", action="store_true", help="Do not include classification samples")
    parser.add_argument("--poll", dest="poll", action="store_true", help="Poll job status until completion")
    parser.add_argument("--no-poll", dest="poll", action="store_false", help="Return after scheduling the job")
    parser.set_defaults(poll=True)
    parser.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between status checks")
    parser.add_argument("--timeout", type=float, default=600.0, help="Maximum time to wait when polling (seconds)")
    parser.add_argument("--dump-payload", action="store_true", help="Print the payload JSON before sending")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading C-MAPSS FD001 dataset...")
    df = load_test_dataset(args.dataset_dir)
    rul_df = load_rul_values(args.dataset_dir)
    sensor_cols = get_sensor_columns()
    all_engine_ids = get_all_engine_ids(df)

    if len(rul_df) != len(all_engine_ids):
        raise ValueError("Mismatch between test dataset engines and RUL entries")

    engine_ids = all_engine_ids[:args.max_engines]
    rul_map = {engine_id: float(rul_df.iloc[idx, 0]) for idx, engine_id in enumerate(all_engine_ids)}

    include_regression = not args.skip_regression
    include_classification = not args.skip_classification
    if not include_regression and not include_classification:
        raise ValueError("At least one of regression or classification retraining must be requested")

    metadata = {
        "source": "FD001 test split",
        "engine_ids": engine_ids,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample_id": uuid.uuid4().hex,
    }

    payload = _build_retraining_payload(
        df,
        rul_map,
        sensor_cols,
        engine_ids,
        include_regression=include_regression,
        include_classification=include_classification,
        partition=args.partition,
        metadata=metadata,
    )

    if args.dump_payload:
        print(json.dumps(payload, indent=2))

    print(
        f"Triggering retraining for engines {engine_ids} | "
        f"regression={include_regression} classification={include_classification}"
    )

    try:
        response = _post_retraining_request(args.base_url, payload)
    except requests.RequestException as exc:
        raise SystemExit(f"Failed to trigger retraining: {exc}") from exc

    job_id = response.get("job_id")
    if not job_id:
        raise SystemExit(f"Unexpected response: {response}")

    print(f"Retraining job scheduled: {job_id}")
    if not args.poll:
        print(json.dumps(response, indent=2))
        return

    try:
        final_status = _poll_job_status(
            args.base_url,
            job_id,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
        )
    except (requests.RequestException, TimeoutError) as exc:
        raise SystemExit(f"Error while polling job status: {exc}") from exc

    print("Final job status:")
    print(json.dumps(final_status, indent=2))


if __name__ == "__main__":
    main()
