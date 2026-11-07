"""Evaluate a saved student checkpoint from node features alone.

    python evaluate.py --ckpt outputs/checkpoints/nc-tran/cora/SAGE/<hash>_<teacher_id>_seed_0.pt
"""
import argparse

import torch

from g2mlp import nc
from g2mlp.utils import get_device


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--device", type=int, default=0, help="GPU index, -1 for CPU")
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
    metrics = nc.evaluate_checkpoint(checkpoint, device, args.data_path)
    print(f"{checkpoint['dataset']} teacher={checkpoint['teacher']} "
          f"setting={checkpoint['setting']} seed={checkpoint['seed']}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
