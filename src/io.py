"""IO utilities for logging and results."""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_logger(log_dir: str, name: str = "runner") -> logging.Logger:
    """Create a logger that writes to file and stdout."""
    ensure_dir(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def append_results(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    """Append results to CSV, creating header if needed."""
    ensure_dir(os.path.dirname(csv_path))
    df = pd.DataFrame(rows)
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
        existing_cols = header.split(",") if header else []
        new_cols = list(df.columns)

        if set(new_cols).issubset(existing_cols):
            df = df.reindex(columns=existing_cols)
            df.to_csv(csv_path, mode="a", index=False, header=False)
            return

        # Schema expanded: rewrite file with union columns.
        df_old = pd.read_csv(csv_path)
        all_cols = existing_cols + [c for c in new_cols if c not in existing_cols]
        df_old = df_old.reindex(columns=all_cols)
        df = df.reindex(columns=all_cols)
        df_all = pd.concat([df_old, df], ignore_index=True)
        df_all.to_csv(csv_path, index=False)


def write_meta(meta_path: str, payload: Dict[str, Any]) -> None:
    """Write meta information to JSON (overwrite)."""
    ensure_dir(os.path.dirname(meta_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_trace_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write trace rows to a CSV file."""
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def collect_meta(config_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect run metadata for reproducibility."""
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config_path": config_path,
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
    }

    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            meta["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    return meta
