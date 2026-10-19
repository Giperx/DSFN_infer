#!/usr/bin/env python3
"""Finetune or train the DSFN warp stage on driving pairs.

Same objective as Warp/newCodes/train.py: two images, overlap / grid / depth
losses, Adam 1e-4 (1e-5 for finetune) with ExponentialLR gamma 0.97.
Samples are cam5+cam4 and cam5+cam3 from train scenes, shuffled by the loader.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _prelaunch():
    if os.environ.get("DSFN_DRIVE_GPU_READY") == "1":
        return
    gpu = "1"
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--gpu" and i + 1 < len(args):
            gpu = args[i + 1]
        elif arg.startswith("--gpu="):
            gpu = arg.split("=", 1)[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["DSFN_DRIVE_GPU_READY"] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":
    _prelaunch()

import argparse
import glob
import importlib.util
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from data import DrivePairDataset, dataset_names, default_epochs  # noqa: E402


def load_module(name, file_path, search_path):
    sys.path.insert(0, search_path)
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def latest_checkpoint(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.pth")))
    return paths[-1] if paths else None


def train(args):
    warp = load_module(
        "dsfn_warp_network",
        os.path.join(ROOT, "Warp/newCodes/network.py"),
        os.path.join(ROOT, "Warp/newCodes"),
    )
    loss_fn = load_module(
        "dsfn_warp_loss",
        os.path.join(ROOT, "Warp/newCodes/loss.py"),
        os.path.join(ROOT, "Warp/newCodes"),
    )

    if args.max_epoch is None:
        args.max_epoch = default_epochs(args.dataset)
    data = DrivePairDataset(
        args.dataset,
        allow_val=args.allow_val,
        max_scenes=args.max_scenes,
        max_pairs=args.max_pairs,
        exchange=not args.no_exchange,
        seed=args.seed,
        refresh_index=args.refresh_index,
    )
    print(
        f"{args.dataset}: sources={data.sources} scenes={data.scenes} pairs={len(data)} "
        f"epochs={args.max_epoch} mode={args.mode} skipped_frames={data.skipped} "
        f"cache={data.cache_path}"
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True,
    )
    if len(loader) == 0:
        raise SystemExit("no full batch; lower --batch-size or raise --max-pairs")

    net = warp.Network()
    if torch.cuda.is_available():
        net = net.cuda()
    lr = args.lr if args.lr is not None else (1e-5 if args.mode == "finetune" else 1e-4)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    ckpt_dir = os.path.join(ROOT, "drive_train", "checkpoints", args.dataset, "warp")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = latest_checkpoint(ckpt_dir)
    start_epoch = 0
    glob_iter = 0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        net.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0))
        glob_iter = int(checkpoint.get("glob_iter", 0))
        scheduler.last_epoch = start_epoch
        print(f"resume {resume_path} at epoch {start_epoch}")
    elif args.mode == "finetune":
        pretrained = args.pretrained or os.path.join(ROOT, "pretrained/epoch001_model_warp.pth")
        checkpoint = torch.load(pretrained, map_location="cpu")
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        net.load_state_dict(state)
        print(f"finetune from {pretrained} lr={lr}")
    else:
        print(f"train warp from scratch lr={lr}")

    if args.bench_steps > 0:
        start_epoch = 0

    print("################## start warp training ##################")
    score_print_fre = 50 if args.bench_steps else 300
    net.train()
    bench_times = []
    for epoch in range(start_epoch, args.max_epoch):
        print("start epoch", epoch, "lr={:.8f}".format(optimizer.param_groups[0]["lr"]))
        loss_sigma = 0.0
        overlap_sigma = 0.0
        nonoverlap_sigma = 0.0
        depth_sigma = 0.0
        epoch_start = time.time()
        for i, batch in enumerate(loader):
            step_start = time.time()
            input1 = batch[0].float()
            input2 = batch[1].float()
            depth1 = batch[2].float()
            depth2 = batch[3].float()
            if torch.cuda.is_available():
                input1 = input1.cuda(non_blocking=True)
                input2 = input2.cuda(non_blocking=True)
                depth1 = depth1.cuda(non_blocking=True)
                depth2 = depth2.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            batch_out = warp.build_model(net, input1, input2)
            overlap_loss = loss_fn.cal_lp_loss(
                input1,
                input2,
                batch_out["output_H"],
                batch_out["output_H_inv"],
                batch_out["warp_mesh"],
                batch_out["warp_mesh_mask"],
            )
            nonoverlap_loss = 10 * loss_fn.inter_grid_loss(
                batch_out["overlap"], batch_out["mesh2"]
            ) + 10 * loss_fn.intra_grid_loss(batch_out["mesh2"])
            depth_loss = 0.3 * loss_fn.cal_depth_loss(
                depth1,
                depth2,
                batch_out["output_H"],
                batch_out["output_H_inv"],
                batch_out["warp_mesh"],
                batch_out["warp_mesh_mask"],
            )
            total = overlap_loss + nonoverlap_loss + depth_loss
            if not torch.isfinite(total):
                print(f"skip non-finite loss at epoch {epoch} iter {i}")
                optimizer.zero_grad(set_to_none=True)
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()

            overlap_sigma += overlap_loss.item()
            nonoverlap_sigma += nonoverlap_loss.item()
            depth_sigma += depth_loss.item()
            loss_sigma += total.item()
            glob_iter += 1
            step_sec = time.time() - step_start
            if args.bench_steps:
                if i >= 1:
                    bench_times.append(step_sec)
                print(
                    f"bench {i + 1}/{args.bench_steps} loss={total.item():.4f} "
                    f"overlap={overlap_loss.item():.4f} grid={nonoverlap_loss.item():.4f} "
                    f"depth={depth_loss.item():.4f} sec={step_sec:.3f}"
                )
                if i + 1 >= args.bench_steps:
                    warm = sum(bench_times) / max(len(bench_times), 1)
                    print(f"BENCH_SEC_PER_STEP {warm:.4f} steps={len(bench_times)} batch={args.batch_size}")
                    return
            elif i % score_print_fre == 0 and i != 0:
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] "
                    "Total Loss: {:.4f}  Overlap Loss: {:.4f}  Non-overlap Loss: {:.4f} "
                    "depth_loss: {:.4f} lr={:.8f} sec/step={:.3f}".format(
                        epoch + 1,
                        args.max_epoch,
                        i + 1,
                        len(loader),
                        loss_sigma / score_print_fre,
                        overlap_sigma / score_print_fre,
                        nonoverlap_sigma / score_print_fre,
                        depth_sigma / score_print_fre,
                        optimizer.param_groups[0]["lr"],
                        step_sec,
                    )
                )
                loss_sigma = overlap_sigma = nonoverlap_sigma = depth_sigma = 0.0

        scheduler.step()
        print(f"epoch {epoch + 1} done in {(time.time() - epoch_start) / 60:.1f} min")
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.max_epoch:
            path = os.path.join(ckpt_dir, f"epoch{epoch + 1:03d}_model.pth")
            torch.save(
                {
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "glob_iter": glob_iter,
                },
                path,
            )
            print("saved", path)
    print("################## end warp training ##################")


def parse_args():
    parser = argparse.ArgumentParser(description="DSFN warp training on driving pairs")
    parser.add_argument("--dataset", required=True, choices=dataset_names())
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--mode", choices=("finetune", "scratch"), default="finetune")
    parser.add_argument("--batch-size", type=int, default=4, help="official warp training batch")
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="override the default: nuscenes 1, lyft (1224+1920) 3, widedrive3 3",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--max-scenes", type=int, default=0, help="debug cap; 0 means all train scenes")
    parser.add_argument("--max-pairs", type=int, default=0, help="random subset per run; 0 means every pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-val", action="store_true")
    parser.add_argument("--no-exchange", action="store_true", help="keep cam5 as the first image")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--bench-steps", type=int, default=0, help="time a few steps and exit")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
