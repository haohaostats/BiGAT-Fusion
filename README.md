# BiGAT-Fusion

PyTorch implementation of BiGAT-Fusion for drug–disease association prediction.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/xpu
```

## Evaluation

The package supports pair-level, drug cold-start, and disease cold-start
evaluation. Cold-start AP and AUROC are calculated per held-out entity and
macro-averaged within each fold.

```powershell
# Pair-level: 10 repetitions of 10-fold cross-validation
python run_fullcv.py --protocol pair --folds 10 --repeats 10 --device xpu

# Drug cold-start: 10 repetitions of 10-fold cross-validation
python run_fullcv.py --protocol drug_cold --folds 10 --cold_repeats 10 --device xpu

# Disease cold-start: 10 repetitions of 10-fold cross-validation
python run_fullcv.py --protocol disease_cold --folds 10 --cold_repeats 10 --device xpu
```

Use `--protocol all` to run all protocols.

## Code structure

- `bigat_fusion/data.py`: dataset loading and similarity graphs
- `bigat_fusion/layers.py`: graph attention and decoder layers
- `bigat_fusion/model.py`: BiGAT-Fusion model
- `bigat_fusion/protocols.py`: pair-level and cold-start partitions
- `bigat_fusion/metrics.py`: AUROC and AUPRC calculation
- `bigat_fusion/training.py`: fold training and model selection
- `bigat_fusion/artifacts.py`: checkpoints, metrics, and interpretation outputs
- `bigat_fusion/cli.py`: command-line configuration

## Outputs

- `results/Gdataset/`: fold metrics
- `train/`: model checkpoints
- `interpret/`: attention weights and fusion gates

## License

MIT
