"""Fill the curvature cache before training.

ORC can take a long time on the larger graphs, and the inductive setting uses one
observed graph per seed. Training reads the cache.

    python scripts/precompute_curvature.py --setting ind \\
        --dataset a-computer --seeds 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from g2mlp.curvature import KINDS, graph_name, node_curvature  # noqa: E402
from g2mlp.data import graph_split, load_data  # noqa: E402
from g2mlp.nc import observed_graph  # noqa: E402
from g2mlp.teacher import SPLIT_RATE  # noqa: E402
from g2mlp.utils import resolve_config  # noqa: E402


def graph_of(setting, dataset, seed, data_path):
    """The graph whose curvature the student of this cell reads."""
    g, _, idx_train, idx_val, idx_test = load_data(dataset, data_path, seed=seed)
    if setting == "tran":
        return g
    idx_obs = graph_split(idx_train, idx_val, idx_test, SPLIT_RATE, seed)[3]
    return observed_graph(g, idx_obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--setting", choices=["tran", "ind"], default="tran")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=None,
                        help="Default: the curv_kind training reads from configs/g2mlp.yaml")
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()
    cfg = resolve_config(args.dataset)
    kinds = args.kinds or [cfg["curv_kind"]]
    for seed in args.seeds:
        graph = graph_of(args.setting, args.dataset, seed, args.data_path)
        for kind in kinds:
            start = time.time()
            node_curvature(graph, kind, args.dataset, graph_name(args.setting, seed),
                           alpha=cfg["curv_alpha"])
            print(f"[precompute] {args.dataset} {args.setting} seed {seed} {kind}: "
                  f"{time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
