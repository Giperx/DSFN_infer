#!/usr/bin/env python3
"""Finetune or train the DSFN fusion stage on driving pairs.

The published fusion weights are the deployed (fused) RepConv. Finetune loads
that checkpoint and trains every parameter. Scratch keeps the train-time
two-branch RepConv.

Warp is frozen. Each step warps one cam5+cam4 or cam5+cam3 pair with
build_depth_output_model, then applies the original fusion losses. The canvas
is the max over the batch. The official fusion batch size is 1.
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


def state_dict_of(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def is_fused_state(state):
    return not any(key.endswith("conv1.weight") or key.endswith("conv2.weight") for key in state)


def fuse_repconv(net, repconv_cls):
    for module in net.modules():
        if isinstance(module, repconv_cls):
            module.fuse_convs()
            module.forward = module.forward_fuse


def enable_grad(net):
    for parameter in net.parameters():
        parameter.requires_grad_(True)


def fusion_forward(net, warp1, warp2, mask1, mask2):
    # Same composition as Fusion build_model, without fuse/detach on every step.
    out = net(warp1, warp2, mask1, mask2)
    learned_mask1 = (mask1 - mask1 * mask2) + mask1 * mask2 * out
    learned_mask2 = (mask2 - mask1 * mask2) + mask1 * mask2 * (1 - out)
    stitched = (warp1 + 1.0) * learned_mask1 + (warp2 + 1.0) * learned_mask2 - 1.0
    return learned_mask1, learned_mask2, stitched


def train(args):
    warp = load_module(
        "dsfn_warp_network",
        os.path.join(ROOT, "Warp/newCodes/network.py"),
        os.path.join(ROOT, "Warp/newCodes"),
    )
    fusion = load_module(
        "dsfn_fusion_network",
        os.path.join(ROOT, "Fusion/newCodes/network.py"),
        os.path.join(ROOT, "Fusion/newCodes"),
    )
    loss_fn = load_module(
        "dsfn_fusion_loss",
        os.path.join(ROOT, "Fusion/newCodes/loss.py"),
        os.path.join(ROOT, "Fusion/newCodes"),
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
        f"epochs={args.max_epoch} mode={args.mode} skipped_frames={data.skipped}"
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=args.batch_size > 1,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    warp_net = warp.Network().cuda().eval()
    for parameter in warp_net.parameters():
        parameter.requires_grad_(False)
    warp_dir = os.path.join(ROOT, "drive_train", "checkpoints", args.dataset, "warp")
    warp_path = args.warp_checkpoint or latest_checkpoint(warp_dir)
    if warp_path is None:
        warp_path = os.path.join(ROOT, "pretrained/epoch001_model_warp.pth")
    warp_ckpt = torch.load(warp_path, map_location="cpu")
    warp_net.load_state_dict(state_dict_of(warp_ckpt))
    print(f"frozen warp {warp_path}")

    net = fusion.Network().cuda()
    lr = args.lr if args.lr is not None else (1e-5 if args.mode == "finetune" else 1e-4)
    ckpt_dir = os.path.join(ROOT, "drive_train", "checkpoints", args.dataset, "fusion")
    os.makedirs(ckpt_dir, exist_ok=True)
    resume_path = latest_checkpoint(ckpt_dir)
    start_epoch = 0
    glob_iter = 0
    resume_optimizer = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu")
        state = state_dict_of(checkpoint)
        if is_fused_state(state):
            fuse_repconv(net, fusion.RepConvN)
        net.load_state_dict(state)
        enable_grad(net)
        start_epoch = int(checkpoint.get("epoch", 0))
        glob_iter = int(checkpoint.get("glob_iter", 0))
        resume_optimizer = checkpoint.get("optimizer")
        print(f"resume fusion {resume_path} at epoch {start_epoch}")
    elif args.mode == "finetune":
        pretrained = args.pretrained or os.path.join(ROOT, "pretrained/epoch002_model_fusion.pth")
        checkpoint = torch.load(pretrained, map_location="cpu")
        state = state_dict_of(checkpoint)
        if is_fused_state(state):
            fuse_repconv(net, fusion.RepConvN)
        net.load_state_dict(state)
        enable_grad(net)
        print(f"finetune fusion from {pretrained} lr={lr}")
    else:
        print(f"train fusion from scratch lr={lr}")

    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    if resume_optimizer is not None:
        optimizer.load_state_dict(resume_optimizer)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    scheduler.last_epoch = start_epoch
    if args.bench_steps > 0:
        start_epoch = 0

    print("################## start fusion training ##################")
    score_print_fre = 20 if args.bench_steps else 300
    bench_times = []
    skipped = 0
    for epoch in range(start_epoch, args.max_epoch):
        print("start epoch", epoch, "lr={:.8f}".format(optimizer.param_groups[0]["lr"]))
        net.train()
        totals = [0.0, 0.0, 0.0, 0.0, 0.0]
        counted = 0
        epoch_start = time.time()
        for i, batch in enumerate(loader):
            step_start = time.time()
            input1 = batch[0].float().cuda(non_blocking=True)
            input2 = batch[1].float().cuda(non_blocking=True)
            depth1 = batch[2].float().cuda(non_blocking=True)
            depth2 = batch[3].float().cuda(non_blocking=True)
            try:
                with torch.no_grad():
                    warped = warp.build_depth_output_model(warp_net, input1, input2, depth1, depth2)
                warp1 = warped["final_warp1"]
                warp2 = warped["final_warp2"]
                mask1 = warped["final_warp1_mask"][:, :1]
                mask2 = warped["final_warp2_mask"][:, :1]
                depth_warp1 = warped["depth_warp1"].unsqueeze(1) / 255.0
                depth_warp2 = warped["depth_warp2"].unsqueeze(1) / 255.0
                canvas_h, canvas_w = int(warp1.shape[-2]), int(warp1.shape[-1])
                if max(canvas_h, canvas_w) > args.max_canvas or canvas_h * canvas_w > args.max_pixels:
                    raise RuntimeError(f"canvas {canvas_w}x{canvas_h}")
                optimizer.zero_grad(set_to_none=True)
                learned1, learned2, stitched = fusion_forward(net, warp1, warp2, mask1, mask2)
                boundary, _boundary_mask = loss_fn.cal_boundary_term(
                    warp1, warp2, mask1, mask2, stitched
                )
                boundary = 10000 * boundary
                smooth1 = 1000 * loss_fn.cal_smooth_term_stitch(stitched, learned1)
                smooth2 = 1000 * loss_fn.cal_smooth_term_diff_new(
                    warp1, warp2, learned1, learned2, mask1 * mask2
                )
                overlap_px = int(((depth_warp1 > 0) & (depth_warp2 > 0)).sum().item())
                if overlap_px > 10:
                    depth_loss = 10 * loss_fn.cal_depth_diff_loss(
                        depth_warp1, depth_warp2, learned1, learned2
                    )
                else:
                    depth_loss = stitched.new_zeros(())
                total = boundary + smooth1 + smooth2 + depth_loss
                if not torch.isfinite(total):
                    raise RuntimeError("non-finite fusion loss")
                total.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
                optimizer.step()
            except RuntimeError as exc:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"skip iter {i}: {exc}")
                if args.bench_steps and i + 1 >= args.bench_steps:
                    warm = sum(bench_times) / max(len(bench_times), 1)
                    print(
                        f"BENCH_SEC_PER_STEP {warm:.4f} steps={len(bench_times)} "
                        f"skipped={skipped}"
                    )
                    return
                continue

            totals[0] += total.item()
            totals[1] += boundary.item()
            totals[2] += smooth1.item()
            totals[3] += smooth2.item()
            totals[4] += float(depth_loss.item())
            counted += 1
            glob_iter += 1
            step_sec = time.time() - step_start
            if args.bench_steps:
                if counted >= 2:
                    bench_times.append(step_sec)
                print(
                    f"bench {i + 1}/{args.bench_steps} loss={total.item():.4f} "
                    f"boundary={boundary.item():.4f} smooth={smooth1.item():.4f} "
                    f"diff={smooth2.item():.4f} depth={float(depth_loss.item()):.4f} "
                    f"canvas={warp1.shape[-1]}x{warp1.shape[-2]} sec={step_sec:.3f}"
                )
                if i + 1 >= args.bench_steps:
                    warm = sum(bench_times) / max(len(bench_times), 1)
                    print(
                        f"BENCH_SEC_PER_STEP {warm:.4f} steps={len(bench_times)} "
                        f"skipped={skipped}"
                    )
                    return
            elif counted % score_print_fre == 0:
                print(
                    "Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] "
                    "Total Loss: {:.4f}  boundary: {:.4f}  smooth: {:.4f}  "
                    "diff: {:.4f}  depth: {:.4f} lr={:.8f} sec/step={:.3f}".format(
                        epoch + 1,
                        args.max_epoch,
                        i + 1,
                        len(loader),
                        totals[0] / score_print_fre,
                        totals[1] / score_print_fre,
                        totals[2] / score_print_fre,
                        totals[3] / score_print_fre,
                        totals[4] / score_print_fre,
                        optimizer.param_groups[0]["lr"],
                        step_sec,
                    )
                )
                totals = [0.0, 0.0, 0.0, 0.0, 0.0]

        scheduler.step()
        print(
            f"epoch {epoch + 1} done in {(time.time() - epoch_start) / 60:.1f} min "
            f"skipped={skipped}"
        )
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
    print("################## end fusion training ##################")


# Official fusion training batch.
DEFAULT_BATCH = {
    "nuscenes": 1,
    "lyft": 1,
    "widedrive3": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="DSFN fusion training on driving pairs")
    parser.add_argument("--dataset", required=True, choices=dataset_names())
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--mode", choices=("finetune", "scratch"), default="finetune")
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="override the default: nuscenes 1, lyft (1224+1920) 3, widedrive3 3",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="official fusion training batch is 1",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--pretrained", default="")
    parser.add_argument("--warp-checkpoint", default="")
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-val", action="store_true")
    parser.add_argument("--no-exchange", action="store_true")
    parser.add_argument("--refresh-index", action="store_true")
    parser.add_argument("--max-canvas", type=int, default=1024, help="skip a warp whose long side exceeds this")
    parser.add_argument("--max-pixels", type=int, default=520000, help="skip a warp whose canvas area exceeds this")
    parser.add_argument("--bench-steps", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH.get(args.dataset, 1)
    return args


if __name__ == "__main__":
    train(parse_args())
