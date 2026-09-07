"""Command-line interface for cross-validation experiments."""

import argparse

from .data import load_dataset
from .runtime import seed_everything, select_device
from .training import run_protocol


def parse_fold_ids(value):
    return [int(item) for item in value.split(",")]


def build_parser():
    """Define experiment command-line arguments."""
    parser = argparse.ArgumentParser(description="Run BiGAT-Fusion cross-validation")
    parser.add_argument("--mat_path", default="data/Gdataset.mat")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_batch_size", type=int, default=65536)
    parser.add_argument("--neg_k", type=int, default=3)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument(
        "--fold_ids",
        type=parse_fold_ids,
        help="optional comma-separated zero-based folds (0 or 0,1)",
    )
    parser.add_argument("--repeats", type=int, default=10, help="pair-level repetitions")
    parser.add_argument("--run_tag", default="", help="label used to keep outputs separate")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "xpu"], default="xpu")
    parser.add_argument("--wd_backbone", type=float, default=1e-4)
    parser.add_argument("--wd_gate", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--lr_patience", type=int, default=20)
    return parser


def validate_args(parser, args):
    """Validate arguments that depend on other options."""
    if args.fold_ids is not None and any(
        fold < 0 or fold >= args.folds for fold in args.fold_ids
    ):
        parser.error("every --fold_ids value must be between 0 and --folds-1")
    if args.folds < 3 or args.repeats < 1 or args.epochs < 1 or args.eval_every < 1:
        parser.error("folds must be at least 3; repeats, epochs and eval_every must be positive")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    seed_everything(args.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")
    data = load_dataset(args.mat_path, k=args.k)
    run_protocol(args, data, device)


if __name__ == "__main__":
    main()
