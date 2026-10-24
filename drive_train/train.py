#!/usr/bin/env python3
"""Run DSFN driving finetuning as warp, then fusion, from one command.

The stages stay separate. This script only sequences them: warp trains to
completion and writes drive_train/checkpoints/<dataset>/warp/, then fusion
loads that checkpoint, freezes it, and trains the fusion network.

    python drive_train/train.py --dataset nuscenes --gpu 4
    python drive_train/train.py --dataset lyft --gpu 4 --stage warp
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import dataset_names  # noqa: E402
from train_fusion import DEFAULT_BATCH  # noqa: E402

WARP_BATCH = 4


def add_flag(command, name, value):
    if value is None:
        return
    command.extend([name, str(value)])


def common_args(args, batch_size):
    command = [
        "--dataset",
        args.dataset,
        "--gpu",
        args.gpu,
        "--mode",
        args.mode,
        "--batch-size",
        str(batch_size),
        "--workers",
        str(args.workers),
        "--save-every",
        str(args.save_every),
        "--max-scenes",
        str(args.max_scenes),
        "--max-pairs",
        str(args.max_pairs),
        "--seed",
        str(args.seed),
    ]
    add_flag(command, "--max-epoch", args.max_epoch)
    add_flag(command, "--lr", args.lr)
    add_flag(command, "--bench-steps", args.bench_steps or None)
    if args.allow_val:
        command.append("--allow-val")
    if args.no_exchange:
        command.append("--no-exchange")
    if args.refresh_index:
        command.append("--refresh-index")
    return command


def run_stage(script, command):
    print(f"\n########## {script} ##########", flush=True)
    subprocess.run([sys.executable, str(HERE / script), *command], check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="DSFN warp then fusion finetuning")
    parser.add_argument("--dataset", required=True, choices=dataset_names())
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--stage", choices=("both", "warp", "fusion"), default="both")
    parser.add_argument("--mode", choices=("finetune", "scratch"), default="finetune")
    parser.add_argument(
        "--warp-batch-size",
        type=int,
        default=WARP_BATCH,
        help="official warp training batch",
    )
    parser.add_argument(
        "--fusion-batch-size",
        type=int,
        default=None,
        help="official fusion training batch is 1",
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="override the default: nuscenes 1, lyft 3, widedrive3 3",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--warp-pretrained", default="")
    parser.add_argument("--fusion-pretrained", default="")
    parser.add_argument(
        "--warp-checkpoint",
        default="",
        help="fusion only; default is the newest checkpoint from the warp stage",
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-val", action="store_true")
    parser.add_argument("--no-exchange", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--max-canvas", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=520000)
    parser.add_argument("--bench-steps", type=int, default=0)
    args = parser.parse_args()
    if args.fusion_batch_size is None:
        args.fusion_batch_size = DEFAULT_BATCH.get(args.dataset, 1)
    return args


def main():
    args = parse_args()
    run_dataset(args)


def run_dataset(args):
    if args.stage in ("both", "warp"):
        command = common_args(args, args.warp_batch_size)
        add_flag(command, "--pretrained", args.warp_pretrained or None)
        run_stage("train_warp.py", command)
    if args.stage in ("both", "fusion"):
        command = common_args(args, args.fusion_batch_size)
        add_flag(command, "--pretrained", args.fusion_pretrained or None)
        add_flag(command, "--warp-checkpoint", args.warp_checkpoint or None)
        add_flag(command, "--max-canvas", args.max_canvas)
        add_flag(command, "--max-pixels", args.max_pixels)
        run_stage("train_fusion.py", command)


if __name__ == "__main__":
    main()
