#!/usr/bin/env bash
#
# setup.sh — one-shot VM setup for the FluxRT-TouchDesigner backend.
#
# Automates: miniconda install, git-lfs, cloning FluxRT, and running its
# installer. Intended for a fresh rented GPU box (RunPod / similar, Ubuntu).
#
# Usage:
#   bash setup.sh
#
# After it finishes, restart your shell (or `source ~/.bashrc`), then:
#   conda activate fluxrt        # if FluxRT's installer made this env
#   cd FluxRT
#   python server-tcp.py --config configs/stream_processor_config.json --port 8080
#
# NOTE: FluxRT also requires downloading model weights (FLUX.2-klein-4B,
# and RIFE). Follow FluxRT's own README for the weight downloads — this
# script sets up the environment and code, not the multi-GB model files,
# since those steps vary (HF auth, int8 vs full, etc.).

set -e  # stop on first error

echo "=== [1/4] Installing Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
    mkdir -p ~/miniconda3
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    echo "Miniconda installed."
else
    echo "Miniconda already present at ~/miniconda3, skipping."
fi

echo "=== [2/4] Installing git-lfs ==="
apt-get update -qq
apt-get install -y -qq git-lfs
git lfs install

echo "=== [3/4] Cloning FluxRT ==="
if [ ! -d "FluxRT" ]; then
    git clone https://github.com/tensorforger/FluxRT
else
    echo "FluxRT directory already exists, skipping clone."
fi

echo "=== [4/4] Running FluxRT installer ==="
cd FluxRT
bash scripts/install.sh

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Restart your shell or run: source ~/.bashrc"
echo "  2. Download FluxRT model weights (see FluxRT's README)"
echo "  3. Copy server-tcp.py into the FluxRT directory"
echo "  4. Run: python server-tcp.py --config configs/stream_processor_config.json --port 8080"
