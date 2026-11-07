"""Seeding, configuration, result files and summaries.

set_seed and model_conf are derived from GLNN (Copyright Snap Inc. 2021, MIT
License, see LICENSE).
"""
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# cuBLAS reads this on first use. torch.use_deterministic_algorithms requires it
# for matrix products on a GPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"
G2MLP_CONFIG = REPO_ROOT / "configs" / "g2mlp.yaml"

KEY_COLUMNS = ["config_hash", "config_json", "dataset", "teacher", "setting", "seed",
               "teacher_id"]
RESULT_COLUMNS = KEY_COLUMNS + ["val", "test", "test_tran", "test_ind", "test_prod"]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(index):
    if index >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def accuracy(logits, labels):
    """Fraction of nodes whose argmax matches the label."""
    return logits.argmax(dim=1).eq(labels).float().mean().item()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_conf(model, dataset, path=MODELS_CONFIG):
    """Model settings: the `global` block overlaid with dataset and model."""
    full = load_yaml(path)
    conf = dict(full["global"])
    if dataset not in full or model not in full[dataset]:
        raise KeyError(f"No {model} block for {dataset} in {path}")
    conf.update(full[dataset][model] or {})
    conf["model_name"] = model
    return conf


def resolve_config(dataset, overrides=None, path=G2MLP_CONFIG):
    """Method settings for one dataset.

    Order: `defaults`, then the block of the dataset under `datasets`, then the
    overrides whose value is not None. The result also holds `student_model`,
    the MLP settings of configs/models.yaml, so the config hash covers them.
    """
    full = load_yaml(path)
    cfg = dict(full["defaults"])
    cfg.update((full.get("datasets") or {}).get(dataset) or {})
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise KeyError(f"Unknown config key '{key}'. Known keys: {sorted(cfg)}")
        if value is not None:
            cfg[key] = value
    cfg["student_model"] = model_conf("MLP", dataset)
    return cfg


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def result_row(cfg, cfg_hash, dataset, teacher, setting, seed, teacher_id, scores):
    row = {"config_hash": cfg_hash, "config_json": json.dumps(cfg, sort_keys=True),
           "dataset": dataset, "teacher": teacher, "setting": setting, "seed": seed,
           "teacher_id": teacher_id}
    row.update({key: f"{value:.6f}" for key, value in scores.items()})
    return row


def read_rows(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_row(path, row, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    if not is_new:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle))
        if header != columns:
            raise ValueError(f"{path} has columns {header}, expected {columns}. "
                             "Write to a new file.")
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def cell_rows(rows, dataset, teacher, setting, teacher_ids, cfg_hash=None):
    """Rows of one cell whose teacher_id matches teacher_ids[seed].

    cfg_hash, when given, narrows the result to one config.
    """
    return [r for r in rows
            if r["dataset"] == dataset and r["teacher"] == teacher and r["setting"] == setting
            and int(r["seed"]) in teacher_ids
            and r.get("teacher_id") == teacher_ids[int(r["seed"])]
            and (cfg_hash is None or r["config_hash"] == cfg_hash)]


def result_metrics(setting):
    if setting == "ind":
        return ["val", "test_tran", "test_ind", "test_prod"]
    return ["val", "test"]


def format_stats(values):
    """Return (mean, std with ddof=1, n)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    return mean, std, n


def summarize(rows, metrics, seeds, label=""):
    """Print mean, std (ddof=1) and n per metric.

    Raises RuntimeError if a requested seed has no row.
    """
    requested = sorted(set(seeds))
    by_seed = {int(r["seed"]): r for r in rows if int(r["seed"]) in requested}
    if len(by_seed) < len(requested):
        missing = sorted(set(requested) - set(by_seed))
        raise RuntimeError(f"{label}: n={len(by_seed)} below the {len(requested)} "
                           f"requested seeds, missing {missing}")
    summary = {}
    for metric in metrics:
        values = [float(by_seed[s][metric]) for s in requested]
        mean, std, n = format_stats(values)
        summary[metric] = (mean, std, n)
        print(f"{label} {metric}: {100 * mean:.2f} +- {100 * std:.2f} (n={n})", flush=True)
    return summary
