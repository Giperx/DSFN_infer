#!/usr/bin/env python3
"""Stitch one driving scene and score it against wide-FOV sparse GT.

Each method writes stitched/ for viewing and crop/ resized to the sparse GT size for scoring.
  chain: cam5+cam4, then that panorama with cam3
  pairwise: cam5+cam4 and cam5+cam3, aligned on cam5

The metric crop is one camera-5 width on each side of the unwarped camera-5
rectangle, so the native crop is three camera-5 widths wide and one camera-5
high. That crop is then resized to the sparse GT size. Scores use only the
sparse GT mask.
"""
import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

import metrics_hm
import stitch_drive as sd


METHODS = ("chain", "pairwise")


def parse_args():
    parser = argparse.ArgumentParser(description="DSFN wide-FOV sparse-GT HM eval")
    parser.add_argument("--dataset", default="nuscenes", choices=[*sorted(sd.DATASET_REL), "lyft"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--all", action="store_true", help="every scene in the dataset val list")
    parser.add_argument("--methods", nargs="+", default=["pairwise"], choices=["pairwise", "chain"])
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--max-height", type=int, default=480)
    parser.add_argument("--frame", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0, help="score only the first N frames; 0 means all")
    parser.add_argument("--save-root", default=str(sd.ROOT / "outputs" / "metrics_wide"))
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def sparse_scene(dataset, scene_name):
    relative = sd.SPARSE_WIDE_REL[dataset]
    return sd.resolve_data_root(relative) / scene_name


def sparse_name(frame, cam):
    return f"{frame}_{cam}_sparse_wide.png"


def list_frames(scene_dir, gt_scene, requested, cam):
    suffix = f"_{cam}_sparse_wide.png"
    if requested:
        tokens = [sd.frame_token(frame) for frame in requested]
    else:
        tokens = []
        for path in sorted((gt_scene / "rgb").glob(f"*{suffix}")):
            tokens.append(path.name[: -len(suffix)])
    kept = []
    for token in tokens:
        rgb = gt_scene / "rgb" / sparse_name(token, cam)
        mask = gt_scene / "mask" / sparse_name(token, cam)
        if not rgb.is_file() or not mask.is_file():
            print(f"skip {token}: missing sparse gt")
            continue
        try:
            sd.load_cam(scene_dir, token, 3)
            sd.load_cam(scene_dir, token, 4)
            sd.load_cam(scene_dir, token, 5)
        except SystemExit as exc:
            print(f"skip {token}: {exc}")
            continue
        kept.append(token)
    return kept


def gt_size(gt_scene, frame, cam):
    image = cv2.imread(str(gt_scene / "rgb" / sparse_name(frame, cam)), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"missing sparse gt rgb for {frame}")
    return image.shape[1], image.shape[0]


def save_view(method_dir, frame, stitched, valid, cam5_mask, gt_hw):
    """Save the full stitch and the GT-sized crop. Masks are not written."""
    _native, metric = sd.prepare_metric_view(stitched, valid, cam5_mask, gt_hw)
    for folder in ("stitched", "crop"):
        (method_dir / folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(method_dir / "stitched" / f"{frame}.png"), stitched)
    cv2.imwrite(str(method_dir / "crop" / f"{frame}.png"), metric)
    return {
        "stitched": [int(stitched.shape[1]), int(stitched.shape[0])],
        "crop": [int(metric.shape[1]), int(metric.shape[0])],
        "cam5_pixels": int(cam5_mask.sum()),
        "white_ratio": float(np.mean(np.all(metric == 255, axis=2))),
    }


def stitch_views(modules, scene_dir, frame, max_height, methods):
    warp, fusion, warp_net, fusion_net = modules
    cam5, _p5 = sd.load_cam(scene_dir, frame, 5)
    cam4, _p4 = sd.load_cam(scene_dir, frame, 4)
    cam3, _p3 = sd.load_cam(scene_dir, frame, 3)
    height = min(max_height, cam5.shape[0])
    cam5_s = sd.resize_height(cam5, height)
    cam4_s = sd.resize_height(cam4, height)
    cam3_s = sd.resize_height(cam3, height)
    print("  stitch cam5 + cam4")
    pair54, valid54, cam5_54, ok54 = sd.stitch_pair(
        warp, fusion, warp_net, fusion_net, cam5_s, cam4_s, height
    )
    print("  stitch cam5 + cam3")
    pair53, valid53, cam5_53, ok53 = sd.stitch_pair(
        warp, fusion, warp_net, fusion_net, cam5_s, cam3_s, height
    )
    print("  align the two panoramas on cam5")
    pairwise, valid_pair, cam5_pair, ok_pair = sd.compose_panoramas(
        pair54, valid54, pair53, valid53, cam5_54, cam5_53
    )
    chain, valid_chain, cam5_chain, ok_chain = None, None, None, False
    if "chain" in methods:
        print("  chain: stitch (cam5+cam4) + cam3")
        for chain_height in (height, 360, 280, 220):
            if chain_height > pair54.shape[0]:
                continue
            chain, valid_chain, cam5_chain, ok_chain = sd.stitch_pair(
                warp, fusion, warp_net, fusion_net, pair54, cam3_s, chain_height, ref_extra_mask=cam5_54
            )
            if ok_chain:
                break
    flags = {
        "ok_cam5_cam4": bool(ok54),
        "ok_cam5_cam3": bool(ok53),
        "ok_pairwise": bool(ok_pair),
        "ok_chain": bool(ok_chain),
    }
    views = {"pairwise": (pairwise, valid_pair, cam5_pair)}
    if "chain" in methods:
        views["chain"] = (chain, valid_chain, cam5_chain)
    return views, flags


def score_method(method, resized_dir, gt_scene, frames, cam):
    full_rows = []
    region_rows = {name: [] for name in metrics_hm.REGION_NAMES}
    per_frame = []
    skipped = []
    blank = 0
    for frame in frames:
        render_path = resized_dir / f"{frame}.png"
        gt_path = gt_scene / "rgb" / sparse_name(frame, cam)
        mask_path = gt_scene / "mask" / sparse_name(frame, cam)
        if not render_path.is_file():
            skipped.append(frame)
            continue
        gt = metrics_hm.load_rgb(gt_path)
        render = metrics_hm.load_rgb(render_path)
        if render.shape[:2] != gt.shape[:2]:
            render = cv2.resize(render, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
        if metrics_hm.blank_side(render):
            blank += 1
            skipped.append(frame)
            print(f"  reject {frame}: blank side", flush=True)
            continue
        mask = metrics_hm.load_mask(mask_path, gt.shape[:2])
        direct = cam == 2
        if direct:
            mask = np.ones(gt.shape[:2], dtype=bool)
        full, regions = metrics_hm.score_pair(render, gt, mask, match=not direct)
        if full is None:
            skipped.append(frame)
            continue
        full_rows.append(full)
        row = {"frame": frame, **{f"full_{key}": value for key, value in full.items()}}
        for name, result in regions.items():
            if result is None:
                continue
            region_rows[name].append(result)
            for key, value in result.items():
                row[f"{name.lower()}_{key}"] = value
        per_frame.append(row)
    print(metrics_hm.success_line(len(full_rows), blank), flush=True)
    return full_rows, region_rows, per_frame, skipped


def write_frame_csv(path, rows):
    keys = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    lines = [",".join(keys)]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append(",".join(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_reports(report_dir, method, meta, full_rows, region_rows, skipped):
    full_stats = metrics_hm.mean_rows(full_rows)
    region_stats = {name: metrics_hm.mean_rows(region_rows[name]) for name in metrics_hm.REGION_NAMES}
    lr = metrics_hm.mean_of_region_means(region_rows, ("Left", "Right"))
    lrc = metrics_hm.mean_of_region_means(region_rows, ("Left", "Center", "Right"))
    header = [
        f"Method: {method}",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        *meta,
        "Mask: sparse GT mask only. LPIPS is not computed.",
        "HM-SSIM is the per-pixel luminance term. MAE and RMSE are on the 0-255 scale.",
        "Crop: one camera-5 width on each side of the camera-5 rectangle, then resize to sparse GT.",
        f"Skipped frames: {len(skipped)}",
        "Crops with a side third at least 90% white are rejected and excluded from the scores.",
    ]
    full_lines = header + ["", *metrics_hm.format_stats(full_stats, "Full")]
    metrics_hm.write_lines(report_dir / f"HM_full_{method}.txt", full_lines)
    region_lines = list(header)
    region_lines.append("Regions are width thirds: Left, Center, Right. L+R and L+R+C average the region means.")
    region_lines.append("")
    for name in metrics_hm.REGION_NAMES:
        region_lines.append(f"[{name}]")
        region_lines.extend(metrics_hm.format_stats(region_stats[name], name))
        region_lines.append("")
    region_lines.append("[L+R]")
    region_lines.extend(metrics_hm.format_stats(lr, "L+R"))
    region_lines.append("")
    region_lines.append("[L+R+C]")
    region_lines.extend(metrics_hm.format_stats(lrc, "L+R+C"))
    metrics_hm.write_lines(report_dir / f"HM_regions_{method}.txt", region_lines)
    return full_stats, region_stats, lr, lrc


def self_check_crop():
    image = np.zeros((100, 400, 3), np.uint8)
    image[20:80, 0:100] = (4, 5, 6)
    image[20:80, 100:200] = (1, 2, 3)
    cam5 = np.zeros((100, 400), dtype=bool)
    cam5[20:80, 100:200] = True
    crop = sd.crop_cam5_triple(image, cam5)
    if crop.shape[:2] != (60, 300):
        raise SystemExit(f"crop self-check failed: {crop.shape}")
    if not np.all(crop[:, 0:100] == (4, 5, 6)) or not np.all(crop[:, 100:200] == (1, 2, 3)):
        raise SystemExit("crop self-check failed: camera 5 is not the center third")
    if not np.all(crop[:, 200:300] == 0):
        raise SystemExit("crop self-check failed: right third")
    narrow = np.full((30, 80, 3), 7, np.uint8)
    narrow_cam5 = np.zeros((30, 80), dtype=bool)
    narrow_cam5[:, 10:60] = True
    padded = sd.crop_cam5_triple(narrow, narrow_cam5)
    if padded.shape[:2] != (30, 150) or not np.all(padded[:, :40] == 255) or not np.all(padded[:, 40:120] == 7):
        raise SystemExit(f"pad self-check failed: {padded.shape}")
    resized = sd.resize_to_gt(crop, (1554, 294))
    if resized.shape[:2] != (294, 1554):
        raise SystemExit(f"resize self-check failed: {resized.shape}")


def scene_names(args):
    if args.all:
        return sd.val_scenes(args.dataset)
    if not args.scene:
        raise SystemExit("pass --scene or --all")
    return [args.scene]


def score_saved_dataset(dataset, save_root, method, cam):
    """Score every saved crop under one dataset and return the pooled rows."""
    root = Path(save_root) / dataset
    full_rows = []
    region_rows = {name: [] for name in metrics_hm.REGION_NAMES}
    skipped = []
    if not root.is_dir():
        return full_rows, region_rows, skipped
    for scene in sorted(path.name for path in root.iterdir() if path.is_dir()):
        gt_scene = sparse_scene(dataset, scene)
        crop_dir = root / scene / method / "crop"
        if not crop_dir.is_dir():
            continue
        frames = sorted(path.stem for path in crop_dir.glob("*.png"))
        scene_full, scene_regions, _per_frame, scene_skipped = score_method(
            method, crop_dir, gt_scene, frames, cam
        )
        full_rows.extend(scene_full)
        for name in metrics_hm.REGION_NAMES:
            region_rows[name].extend(scene_regions[name])
        skipped.extend(f"{scene}/{frame}" for frame in scene_skipped)
        print(f"  {dataset}/{scene}: {len(scene_full)} frames", flush=True)
    return full_rows, region_rows, skipped


def main():
    args = parse_args()
    self_check_crop()
    methods = tuple(args.methods)
    started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.dataset == "lyft":
        report_dir = Path(args.save_root) / "lyft" / "reports"
        for method in methods:
            full_rows = []
            region_rows = {name: [] for name in metrics_hm.REGION_NAMES}
            skipped = []
            extra = ["Pool: every lyft1920 and lyft1224 frame. Subsets are not averaged."]
            for name in ("lyft1920", "lyft1224"):
                print(f"scoring {name} {method}", flush=True)
                part_full, part_regions, part_skipped = score_saved_dataset(
                    name, args.save_root, method, sd.gt_cam(name)
                )
                full_rows.extend(part_full)
                for region in metrics_hm.REGION_NAMES:
                    region_rows[region].extend(part_regions[region])
                skipped.extend(part_skipped)
                stats = metrics_hm.mean_rows(part_full)
                extra.append(name + ": " + ", ".join(metrics_hm.format_stats(stats, "Full")))
            if not full_rows:
                raise SystemExit(f"no scored lyft frames for {method}")
            meta = [f"Dataset: lyft", f"Method: {method}", f"Run time: {started}", *extra]
            write_reports(report_dir, method, meta, full_rows, region_rows, skipped)
        return

    cam = sd.gt_cam(args.dataset)
    names = scene_names(args)
    modules = None if args.score_only else sd.load_nets(args.gpu)
    pooled = {
        method: {"full": [], "regions": {name: [] for name in metrics_hm.REGION_NAMES}, "skipped": []}
        for method in methods
    }
    for scene_index, scene_name in enumerate(names, start=1):
        scene_dir = sd.scene_dir(args.dataset, scene_name)
        gt_scene = sparse_scene(args.dataset, scene_dir.name)
        frames = list_frames(scene_dir, gt_scene, args.frame, cam)
        if args.limit > 0:
            frames = frames[:args.limit]
        print(f"scene {scene_index}/{len(names)} {scene_dir.name}: {len(frames)} frames", flush=True)
        if not frames:
            continue
        size = gt_size(gt_scene, frames[0], cam)
        out_dir = Path(args.save_root) / args.dataset / scene_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        if modules is not None:
            log_path = out_dir / "stitch_log.jsonl"
            with log_path.open("a") as handle:
                for index, frame in enumerate(frames, start=1):
                    resized_ready = all(
                        (out_dir / method / "crop" / f"{frame}.png").is_file() for method in methods
                    )
                    if args.skip_existing and resized_ready:
                        print(f"[{index}/{len(frames)}] {frame} exists, skip", flush=True)
                        continue
                    print(f"[{index}/{len(frames)}] {frame}", flush=True)
                    try:
                        views, flags = stitch_views(modules, scene_dir, frame, args.max_height, methods)
                    except Exception as exc:
                        print(f"  frame failed: {exc}", flush=True)
                        handle.write(json.dumps({"frame": frame, "error": str(exc)}) + "\n")
                        handle.flush()
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    record = {"frame": frame, **flags}
                    for method, (image, valid, cam5_mask) in views.items():
                        if image is None or valid is None or cam5_mask is None:
                            record[method] = None
                            continue
                        if max(image.shape[:2]) > 65500:
                            print(
                                f"  {method} canvas {image.shape[1]}x{image.shape[0]} exceeds the image limit, skip",
                                flush=True,
                            )
                            record[method] = None
                            continue
                        try:
                            record[method] = save_view(out_dir / method, frame, image, valid, cam5_mask, size)
                        except (OSError, cv2.error) as exc:
                            print(f"  {method} save failed: {exc}", flush=True)
                            record[method] = None
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    del views
                    gc.collect()
                    torch.cuda.empty_cache()
        for method in methods:
            full_rows, region_rows, per_frame, skipped = score_method(
                method, out_dir / method / "crop", gt_scene, frames, cam
            )
            report_dir = out_dir / "reports"
            write_frame_csv(report_dir / f"frames_{method}.csv", per_frame)
            meta = [
                f"Dataset: {args.dataset}",
                f"Scene: {scene_dir.name}",
                f"GT root: {gt_scene}",
                f"Stitched dir: {out_dir / method / 'stitched'}",
                f"Metric crop dir: {out_dir / method / 'crop'}",
                f"GT size: {size[0]}x{size[1]}",
                f"Stitch max height: {args.max_height}",
                f"Run time: {started}",
            ]
            write_reports(report_dir, method, meta, full_rows, region_rows, skipped)
            pooled[method]["full"].extend(full_rows)
            for name in metrics_hm.REGION_NAMES:
                pooled[method]["regions"][name].extend(region_rows[name])
            pooled[method]["skipped"].extend(f"{scene_dir.name}/{frame}" for frame in skipped)
    dataset_report = Path(args.save_root) / args.dataset / "reports"
    for method in methods:
        meta = [
            f"Dataset: {args.dataset}",
            "Pool: every validation frame. Scenes are not averaged.",
            f"Run time: {started}",
        ]
        write_reports(
            dataset_report,
            method,
            meta,
            pooled[method]["full"],
            pooled[method]["regions"],
            pooled[method]["skipped"],
        )


if __name__ == "__main__":
    main()
