#!/usr/bin/env bash
# Create a clean conda environment for G2MLP.
#
#   bash prepare_env.sh          torch 2.6.0 / dgl 2.5.0, CUDA 12.4, listed in the paper
#   bash prepare_env.sh cpu      torch 2.2.2 / dgl 2.2.1, CPU only
set -euo pipefail

STACK="${1:-cu124}"
case "$STACK" in
    cu124) TORCH=2.6.0; DGL=2.5.0; NUMPY=2.2.6;  SCIPY=1.13.1; TORCHDATA=0.6.1 ;;
    cpu)   TORCH=2.2.2; DGL=2.2.1; NUMPY=1.26.4; SCIPY=1.11.1; TORCHDATA=0.7.1 ;;
    *) echo "Unknown stack: $STACK (expected cu124 or cpu)" >&2; exit 1 ;;
esac
DGL_WHEELS="https://data.dgl.ai/wheels/torch-${TORCH%.*}"
[ "$STACK" = cpu ] || DGL_WHEELS="$DGL_WHEELS/$STACK"

conda create -y -n g2mlp python=3.10
# The conda hook reads unset variables.
set +u
eval "$(conda shell.bash hook)"
conda activate g2mlp
set -u

# Old pip versions fail on some wheels of the PyTorch index.
python -m pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir "torch==$TORCH" --index-url "https://download.pytorch.org/whl/$STACK"
pip install --no-cache-dir "numpy==$NUMPY" "scipy==$SCIPY" requests
# torchdata metadata can pin an older torch, and dgl needs its datapipes module.
pip install --no-cache-dir --no-deps "torchdata==$TORCHDATA"
pip install --no-cache-dir "dgl==$DGL" -f "$DGL_WHEELS/repo.html"
pip install --no-cache-dir -r requirements.txt
python -c "import dgl, torch; print('torch', torch.__version__, 'dgl', dgl.__version__)"
