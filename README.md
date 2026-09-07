# BiGAT-Fusion

PyTorch implementation for drug–disease association prediction, combining dual-view
graphs, direction-specific bidirectional attention, node-wise fusion gates, and a
hybrid residual decoder.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/xpu
```

## Training and evaluation

```powershell
python run_fullcv.py --device xpu
```

Defaults: `data/Gdataset.mat`, 10 repetitions of 10-fold pair-level
cross-validation, 4,000 epochs per fold.

## Code structure

- `data.py`: dataset loading and similarity graphs
- `layers.py`: graph attention and residual decoder
- `model.py`: dual-view encoding and fusion
- `protocols.py`: cross-validation partitions
- `sampling.py`: negative sampling
- `training.py`: optimization and checkpoint selection
- `metrics.py`: prediction scores and evaluation
- `artifacts.py`: checkpoints, metrics, attention weights, and gates
- `runtime.py`, `cli.py`: device setup and command-line options

Modules are in `bigat_fusion/`.

## Outputs

- `results/Gdataset/`: fold metrics
- `train/`: model checkpoints
- `interpret/`: attention weights and fusion gates

## License

MIT
