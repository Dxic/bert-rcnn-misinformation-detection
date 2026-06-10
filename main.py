from __future__ import annotations

import argparse
from typing import Optional

from config import ExperimentConfig
from experiments import (
    run_baseline_rcnn_w2v_ce,
    run_bert_rcnn,
    print_result,
)


VALID_EXPERIMENTS = {
    "baseline_rcnn_w2v_ce",
    "bert_rcnn_ce",
    "bert_rcnn_focal",
    "bert_rcnn_ce_imbalanced",
    "bert_rcnn_focal_imbalanced",
}


def run_colab(
    csv_path: str,
    experiment: str,
    epochs: int = 5,
    batch_size: int = 8,
    max_seq_len: int = 128,
    grad_accum_steps: int = 1,
    focal_gamma: float = 2.0,
    num_workers: int = 2,
    seed: int = 42,
):
    if experiment not in VALID_EXPERIMENTS:
        raise ValueError(
            "Experiment tidak valid. Pilih salah satu: "
            + ", ".join(sorted(VALID_EXPERIMENTS))
        )

    cfg = ExperimentConfig(
        csv_path=csv_path,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        grad_accum_steps=grad_accum_steps,
        focal_gamma=focal_gamma,
        num_workers=num_workers,
        seed=seed,
    )

    if experiment == "baseline_rcnn_w2v_ce":
        result = run_baseline_rcnn_w2v_ce(
            cfg,
            dataset_name="balanced_100k",
        )

    elif experiment == "bert_rcnn_ce":
        result = run_bert_rcnn(
            cfg,
            use_focal_loss=False,
            dataset_name="balanced_100k",
            experiment_id="B1",
        )

    elif experiment == "bert_rcnn_focal":
        result = run_bert_rcnn(
            cfg,
            use_focal_loss=True,
            dataset_name="balanced_100k",
            experiment_id="B2",
        )

    elif experiment == "bert_rcnn_ce_imbalanced":
        result = run_bert_rcnn(
            cfg,
            use_focal_loss=False,
            dataset_name="imbalanced_controlled",
            experiment_id="C1",
        )

    else:
        result = run_bert_rcnn(
            cfg,
            use_focal_loss=True,
            dataset_name="imbalanced_controlled",
            experiment_id="C2",
        )

    print_result(result)

    return result


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv-path", required=True)

    parser.add_argument(
        "--experiment",
        required=True,
        choices=sorted(VALID_EXPERIMENTS),
    )

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_colab(
        csv_path=args.csv_path,
        experiment=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        grad_accum_steps=args.grad_accum_steps,
        focal_gamma=args.focal_gamma,
        num_workers=args.num_workers,
        seed=args.seed,
    )
