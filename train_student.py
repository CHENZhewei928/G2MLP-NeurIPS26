"""Train G2MLP students.

    python train_student.py --dataset cora --seeds 0 1 2
    python train_student.py --dataset cora --setting ind --seeds 0
    python train_student.py --dataset a-photo --seeds 0 --lambda_caw 0.5 --caw_transform student

The student objective is Eqs. (10)-(13) of the paper with the settings in
configs/g2mlp.yaml. --lambda_caw and --caw_transform override the config.

Every requested seed has its teacher verified. A seed is skipped only when the
results file holds a row with the same config and the same teacher, and, with
--save_ckpt, its checkpoint exists.
"""
import argparse
import json
from pathlib import Path

import torch

from g2mlp import nc
from g2mlp.teacher import SPLIT_RATE
from g2mlp.utils import (RESULT_COLUMNS, REPO_ROOT, append_row, cell_rows, config_hash,
                         get_device, read_rows, resolve_config, result_metrics, result_row,
                         summarize)

RESULTS_DIR = REPO_ROOT / "outputs" / "results"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher", default="SAGE")
    parser.add_argument("--setting", choices=["tran", "ind"], default="tran")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--lambda_caw", type=float, default=None,
                        help="Default from configs/g2mlp.yaml")
    parser.add_argument("--caw_transform", choices=["none", "student", "both"], default=None,
                        help="Default from configs/g2mlp.yaml")
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--device", type=int, default=0, help="GPU index, -1 for CPU")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--teacher_root", default=None)
    parser.add_argument("--results", default=None, help="Results CSV path")
    return parser.parse_args()


def checkpoint_path(setting, dataset, teacher, cfg_hash, teacher_id, seed):
    return (CHECKPOINT_DIR / f"nc-{setting}" / dataset / teacher
            / f"{cfg_hash}_{teacher_id}_seed_{seed}.pt")


def main():
    args = parse_args()
    device = get_device(args.device)
    setting = args.setting
    cfg = resolve_config(args.dataset, {"lambda_caw": args.lambda_caw,
                                        "caw_transform": args.caw_transform})
    cfg_hash = config_hash(cfg)
    results = Path(args.results) if args.results else RESULTS_DIR / "nc.csv"
    label = f"{args.dataset} {args.teacher} {setting} [{cfg_hash}]"
    print(f"{label}\n{json.dumps(cfg, sort_keys=True)}", flush=True)

    teacher_ids = {}
    for seed in args.seeds:
        ctx = nc.build_context(args.dataset, args.teacher, setting, seed, device,
                               args.data_path, args.teacher_root, SPLIT_RATE)
        teacher_ids[seed] = ctx.teacher_id
        path = checkpoint_path(setting, args.dataset, args.teacher, cfg_hash, ctx.teacher_id,
                               seed)
        done = cell_rows(read_rows(results), args.dataset, args.teacher, setting,
                         {seed: ctx.teacher_id}, cfg_hash)
        if done and (path.exists() or not args.save_ckpt):
            print(f"[student] seed {seed} already in {results}, skipping", flush=True)
            continue
        scores, checkpoint = nc.run_seed(ctx, cfg)
        shown = " ".join(f"{k}={v:.4f}" for k, v in scores.items())
        print(f"[student] {args.dataset} {args.teacher} {setting} seed={seed} {shown}", flush=True)
        append_row(results, result_row(cfg, cfg_hash, args.dataset, args.teacher, setting, seed,
                                       ctx.teacher_id, scores), RESULT_COLUMNS)
        if args.save_ckpt:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)
            print(f"[student] checkpoint {path}", flush=True)

    rows = cell_rows(read_rows(results), args.dataset, args.teacher, setting, teacher_ids,
                     cfg_hash)
    summarize(rows, result_metrics(setting), args.seeds, label)


if __name__ == "__main__":
    main()
