"""Teacher artifacts: layout, saving and verified loading.

A teacher directory is
    outputs/teachers/nc-<setting>/<dataset>/<teacher>/seed_<s>/
and holds
    out.npz         log_softmax outputs
    logits.npz      raw network outputs
    h_teacher.npz   penultimate hidden representation
    model.pth       teacher state dict
    meta.json       dataset, task, setting, seed, teacher, num_nodes, split
                    fingerprint, val score, test scores, config, versions

One seed value sets the teacher, the split and the student.

Inductive artifacts are full-size with NaN at inductive test nodes, so a
student cannot read them by accident.
"""
import hashlib
import json
import math
import platform
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .utils import REPO_ROOT

TEACHER_ROOT = REPO_ROOT / "outputs" / "teachers"
VAL_TOLERANCE = 1e-4


class TeacherArtifactError(RuntimeError):
    """A teacher artifact is missing, inconsistent or unusable."""


@dataclass
class TeacherArtifacts:
    out: torch.Tensor
    h: torch.Tensor
    logits: torch.Tensor
    meta: dict
    identity: str


def teacher_dir(setting, dataset, teacher, seed, root=None):
    root = Path(root) if root is not None else TEACHER_ROOT
    return root / f"nc-{setting}" / dataset / teacher / f"seed_{seed}"


def package_versions():
    import dgl
    import scipy

    return {"python": platform.python_version(), "torch": torch.__version__,
            "dgl": dgl.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


def save_teacher(directory, meta, out, h_teacher, model_state, logits):
    """Write all artifacts. meta.json goes last, so it marks a complete directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").unlink(missing_ok=True)
    np.savez(directory / "out", out.detach().cpu().numpy())
    np.savez(directory / "h_teacher", h_teacher.detach().cpu().numpy())
    np.savez(directory / "logits", logits.detach().cpu().numpy())
    torch.save(model_state, directory / "model.pth")
    meta = dict(meta, versions=package_versions())
    with open(directory / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def _fail(directory, message):
    """Raise with a retrain command read off the teacher directory layout."""
    setting = directory.parent.parent.parent.name.partition("-")[2]
    command = (f"python train_teacher.py --teacher {directory.parent.name} "
               f"--dataset {directory.parent.parent.name} --setting {setting} "
               f"--seeds {directory.name.replace('seed_', '')} --overwrite")
    raise TeacherArtifactError(f"{directory}: {message}\nFix: retrain with `{command}`.")


def _load_array(directory, name):
    try:
        with np.load(directory / f"{name}.npz") as data:
            return torch.from_numpy(data["arr_0"]).float()
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
        _fail(directory, f"{name}.npz is unreadable ({err})")


def read_meta(directory):
    directory = Path(directory)
    path = directory / "meta.json"
    if not directory.is_dir():
        _fail(directory, "teacher directory does not exist")
    if not path.is_file():
        _fail(directory, "meta.json is missing, the directory is incomplete")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as err:
        _fail(directory, f"meta.json is unreadable ({err})")


def _check_finite(directory, name, tensor, index):
    values = tensor if index is None else tensor[index]
    if not torch.isfinite(values).all():
        _fail(directory, f"{name} holds NaN or Inf in the slice the student reads")


def check_teacher_config(directory, meta, expected):
    """Raise unless meta.json stores the values of `expected`."""
    stored = meta.get("config") or {}
    differs = {key: (stored.get(key), value) for key, value in expected.items()
               if stored.get(key) != value}
    if differs:
        _fail(Path(directory), "teacher config differs from the current one, as "
              f"key: (stored, current): {differs}")


def _identity(*tensors):
    """Short digest of the teacher outputs, to tie a result to the teacher behind it."""
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:12]


def load_teacher(directory, expected_fingerprint, expected_meta, expected_config, labels,
                 idx_val, num_classes, idx_read=None, idx_masked=None):
    """Load a teacher and verify it, raising TeacherArtifactError on any problem.

    Checks: files present; meta present; meta names the dataset, task, setting,
    seed and teacher asked for; the stored config holds the current model
    settings; the split fingerprint matches; shapes agree; no NaN or Inf in the
    slices the student reads; the val score recomputed from out.npz agrees with
    a finite meta value within 1e-4; the val score is not near chance.

    expected_meta holds dataset, task, setting, seed and teacher.
    expected_config holds the model settings the teacher must have used.
    idx_read limits the finiteness check to the nodes a student reads, and
    idx_masked lists nodes that must be NaN in every array (inductive test
    nodes).
    """
    directory = Path(directory)
    meta = read_meta(directory)
    for key, value in expected_meta.items():
        if meta.get(key) != value:
            _fail(directory, f"meta.json has {key}={meta.get(key)!r}, expected {value!r}")
    check_teacher_config(directory, meta, expected_config)

    required = ["out.npz", "h_teacher.npz", "model.pth", "logits.npz"]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        _fail(directory, f"missing files: {missing}")

    if meta.get("split_fingerprint") != expected_fingerprint:
        _fail(directory, "split fingerprint differs: this teacher was trained on another "
              "split")

    out = _load_array(directory, "out")
    h = _load_array(directory, "h_teacher")
    logits = _load_array(directory, "logits")

    num_nodes = meta.get("num_nodes")
    for name, tensor in (("out", out), ("h_teacher", h), ("logits", logits)):
        if tensor.dim() != 2 or tensor.shape[0] != num_nodes:
            _fail(directory, f"{name} has shape {tuple(tensor.shape)}, "
                  f"expected {num_nodes} rows")
    if logits.shape != out.shape:
        _fail(directory, f"logits shape {tuple(logits.shape)} differs from out "
              f"{tuple(out.shape)}")

    for name, tensor in (("out", out), ("h_teacher", h), ("logits", logits)):
        _check_finite(directory, name, tensor, idx_read)
        if idx_masked is not None and not torch.isnan(tensor[idx_masked]).all():
            _fail(directory, f"inductive test nodes are not masked in {name}.npz")

    stored = meta.get("val_score")
    if not isinstance(stored, (int, float)) or not math.isfinite(stored):
        _fail(directory, f"meta.json has no finite val_score (found {stored!r})")
    if labels.shape[0] != num_nodes:
        _fail(directory, f"labels do not fit {num_nodes} nodes")
    val = out[idx_val].argmax(dim=1).eq(labels[idx_val]).float().mean().item()
    chance = 1.5 / num_classes
    if abs(val - stored) > VAL_TOLERANCE:
        _fail(directory, f"val score recomputed from out.npz is {val:.6f}, meta says "
              f"{stored:.6f}: the file is corrupted or stale")
    if val < chance:
        _fail(directory, f"val score {val:.4f} is near chance (threshold {chance:.4f})")

    return TeacherArtifacts(out=out, h=h, logits=logits, meta=meta,
                            identity=_identity(out, h, logits))
