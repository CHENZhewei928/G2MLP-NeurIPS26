"""Train teachers and write verified artifacts.

Derived from GLNN (Copyright Snap Inc. 2021, MIT License, see LICENSE).

    python train_teacher.py --teacher SAGE --dataset cora --setting tran --seeds 0 1 2

A seed with artifacts is skipped only if they were trained with the current
settings. Otherwise the run stops and asks for --overwrite.
"""
import argparse

from g2mlp import teacher
from g2mlp.artifacts import check_teacher_config, read_meta, teacher_dir
from g2mlp.utils import get_device, summarize


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher", default="SAGE")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--setting", choices=["tran", "ind"], default="tran")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--device", type=int, default=0, help="GPU index, -1 for CPU")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--teacher_root", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Retrain even if the seed already has artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    setting = args.setting
    expected = teacher.teacher_conf(args.teacher, args.dataset)
    rows = []
    for seed in args.seeds:
        directory = teacher_dir(setting, args.dataset, args.teacher, seed, args.teacher_root)
        if (directory / "meta.json").exists() and not args.overwrite:
            meta = read_meta(directory)
            check_teacher_config(directory, meta, expected)
            print(f"[teacher] seed {seed} exists, skipping", flush=True)
        else:
            meta = teacher.train_nc_teacher(args.dataset, args.teacher, setting, seed, device,
                                            args.data_path, args.teacher_root)
        scores = " ".join(f"{k}={v:.4f}" for k, v in meta["test_scores"].items())
        print(f"[teacher] {args.dataset} {args.teacher} {setting} seed={seed} "
              f"val={meta['val_score']:.4f} {scores}", flush=True)
        rows.append(dict(meta["test_scores"], seed=seed, val=meta["val_score"]))
    metrics = ["val"] + list(rows[0].keys() - {"seed", "val"})
    summarize(rows, sorted(metrics), args.seeds, f"{args.dataset} {args.teacher} {setting}")


if __name__ == "__main__":
    main()
