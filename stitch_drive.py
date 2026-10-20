#!/usr/bin/env python3
"""Stitch three rear cameras with DSFN (warp, then fusion).

DSFN aligns one pair. cam3 overlaps cam5, not cam4, so the wide view is
two pairs against the back camera: cam5+cam4 and cam5+cam3. Those panoramas
are then aligned on the shared cam5 content.

Depth maps are only used by the training loss. Inference uses the warp
checkpoint and the fusion checkpoint.

Examples:
  python stitch_drive.py --dataset nuscenes --scene 078 --frame 19
  python stitch_drive.py --dataset lyft1920 --scene 004 --frame 2 --frame 62
"""
import argparse
import gc
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
WORK = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub")
DATASET_REL = {
    "nuscenes": Path("nuscenes/processed_10Hz_v2"),
    "lyft1920": Path("lyft/lyft_val1920_3cams"),
    "lyft1224": Path("lyft/lyft_val1224_3cams"),
    "widedrive3": Path("WideDrive3_processed/WideDriveVal"),
    "ddad": Path("ddad_process/train"),
    "syn_nuscenes": Path("syn_nuscenes/processed_10Hz"),
}
SPARSE_WIDE_REL = {
    "nuscenes": Path("nuscenes/sparseWideFOVImages3_1554x294"),
    "lyft1920": Path("lyft/1920_sparseWideFOVImages3_1554x294"),
    "lyft1224": Path("lyft/1224_sparseWideFOVImages3_1554x434"),
    "widedrive3": Path("WideDrive_processed/sparseWideFOVImages3_1554x294"),
}
VAL_LIST_REL = {
    "nuscenes": Path("nuscenes/processed_10Hz_v2/nuScenes_Val.txt"),
    "lyft1920": Path("lyft/lyft_val1920_3cams/lyft_val1920.txt"),
    "lyft1224": Path("lyft/lyft_val1224_3cams/lyft_val1224.txt"),
    "widedrive3": Path("WideDrive3_processed/WideDriveVal/val.txt"),
}
GT_CAM = {"widedrive3": 2}
WARP_CKPT = ROOT / "pretrained/epoch001_model_warp.pth"
FUSION_CKPT = ROOT / "pretrained/epoch002_model_fusion.pth"


def load_module(name, file_path, search_path):
    sys.path.insert(0, str(search_path))
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="DSFN 3-camera driving stitch")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_REL))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frame", action="append", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-height", type=int, default=480)
    parser.add_argument("--save-root", default=str(ROOT / "outputs"))
    parser.add_argument("--chain-only", action="store_true", help="only stitch cam5+cam4, then that result with cam3")
    return parser.parse_args()


def resolve_data_root(relative):
    local = ROOT / "datasets" / relative
    if local.is_dir():
        return local
    shared = WORK / "datasets" / relative
    if shared.is_dir():
        return shared
    raise SystemExit(f"dataset not found: {local} or {shared}")


def resolve_data_path(relative):
    """Resolve a file or directory under the local dataset root, then the shared one."""
    for root in (ROOT / "datasets", WORK / "datasets"):
        path = root / relative
        if path.exists():
            return path
    raise SystemExit(f"path not found: {relative}")


def gt_cam(dataset):
    return GT_CAM.get(dataset, 5)


def val_scenes(dataset):
    listing = resolve_data_path(VAL_LIST_REL[dataset])
    return [line.strip() for line in listing.read_text().splitlines() if line.strip()]


def scene_dir(dataset, scene):
    root = resolve_data_root(DATASET_REL[dataset])
    name = str(scene)
    candidates = [name, name.zfill(3)]
    if name.isdigit():
        candidates.append(str(int(name)).zfill(3))
    for cand in candidates:
        path = root / cand
        if path.is_dir():
            return path
    raise SystemExit(f"scene not found under {root}: tried {candidates}")


def frame_token(frame):
    text = str(frame)
    return f"{int(text):03d}" if text.isdigit() else text


def load_cam(scene, frame, cam):
    token = frame_token(frame)
    path = scene / "images" / f"{token}_{cam}.jpg"
    if not path.is_file():
        alt = scene / "images" / f"{token}_{cam}.png"
        path = alt if alt.is_file() else path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"missing image: {path}")
    return image, path


def resize_height(image, height):
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * (height / h))))
    interp = cv2.INTER_AREA if height < h else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def resize_mask(mask, height, width=None):
    mask_u8 = mask.astype(np.uint8)
    h, w = mask_u8.shape[:2]
    if width is None:
        if h == height:
            return mask.astype(bool)
        width = max(1, int(round(w * (height / h))))
    if h == height and w == width:
        return mask.astype(bool)
    resized = cv2.resize(mask_u8, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    return resized > 0


def bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def crop_layers(image, *masks, margin=2):
    """Crop to the first mask. Later masks stay aligned with the image."""
    valid = masks[0]
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return (image, *masks)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    cropped = [image[y0:y1, x0:x1]]
    for mask in masks:
        cropped.append(mask[y0:y1, x0:x1])
    return tuple(cropped)


def paste_bool(canvas, mask, origin):
    if mask is None or mask.size == 0:
        return
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    height, width = mask.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas.shape[1], x + width), min(canvas.shape[0], y + height)
    if x1 <= x0 or y1 <= y0:
        return
    canvas[y0:y1, x0:x1] |= mask[y0 - y:y0 - y + (y1 - y0), x0 - x:x0 - x + (x1 - x0)]


def embed_extra_mask(ref_mask, extra_mask):
    """Place extra_mask into the axis-aligned footprint of the reference warp."""
    canvas = np.zeros(ref_mask.shape[:2], dtype=bool)
    box = bbox_from_mask(ref_mask)
    if box is None:
        return canvas
    y0, y1, x0, x1 = box
    if y1 <= y0 or x1 <= x0:
        return canvas
    resized = resize_mask(extra_mask, y1 - y0, x1 - x0)
    canvas[y0:y1, x0:x1] = resized
    canvas &= ref_mask
    return canvas


def paint_white(image, valid):
    painted = image.copy()
    painted[~valid] = 255
    return painted


def crop_cam5_triple(image, cam5_mask):
    """One camera-5 width on each side of the camera-5 rectangle.

    Camera 5 is only translated, so its mask is still that original rectangle.
    The crop keeps its height and is three times its width, with camera 5 in
    the center third. Pixels outside the stitch stay white.
    """
    box = bbox_from_mask(cam5_mask)
    if box is None:
        raise RuntimeError("camera 5 mask is empty")
    origin_y, y1, origin_x, x1 = box
    ref_h = y1 - origin_y
    ref_w = x1 - origin_x
    if ref_h < 1 or ref_w < 1:
        raise RuntimeError("camera 5 rectangle is empty")
    output = np.full((ref_h, ref_w * 3, image.shape[2]), 255, dtype=image.dtype)
    src_y0 = origin_y
    src_x0 = origin_x - ref_w
    dst_y0 = max(0, -src_y0)
    dst_x0 = max(0, -src_x0)
    dst_y1 = min(ref_h, image.shape[0] - src_y0)
    dst_x1 = min(ref_w * 3, image.shape[1] - src_x0)
    if dst_y1 > dst_y0 and dst_x1 > dst_x0:
        output[dst_y0:dst_y1, dst_x0:dst_x1] = image[
            src_y0 + dst_y0:src_y0 + dst_y1,
            src_x0 + dst_x0:src_x0 + dst_x1,
        ]
    return output


def resize_to_gt(image, size_wh):
    width, height = int(size_wh[0]), int(size_wh[1])
    if image.shape[1] == width and image.shape[0] == height:
        return image
    shrinking = image.shape[1] > width or image.shape[0] > height
    interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interp)


def prepare_metric_view(image, valid, cam5_mask, gt_size):
    """Return the native 3x camera-5 crop and that crop resized to the GT."""
    del valid
    crop = crop_cam5_triple(image, cam5_mask)
    return crop, resize_to_gt(crop, gt_size)


def to_tensor(image):
    norm = image.astype(np.float32)
    norm = (norm / 127.5) - 1.0
    return torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)


def crop_valid(image, mask, margin=2):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image, mask
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def translation(base, other, mask_base, mask_other):
    sift = cv2.SIFT_create(nfeatures=3000)
    gray_base = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    gray_other = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(gray_base, mask_base.astype(np.uint8) * 255)
    kp2, des2 = sift.detectAndCompute(gray_other, mask_other.astype(np.uint8) * 255)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    pts_base = np.float32([kp1[m.queryIdx].pt for m in good])
    pts_other = np.float32([kp2[m.trainIdx].pt for m in good])
    delta = pts_base - pts_other
    median = np.median(delta, axis=0)
    inliers = np.linalg.norm(delta - median, axis=1) < 15
    if int(inliers.sum()) < 10:
        return None
    shift = np.median(delta[inliers], axis=0)
    return float(shift[0]), float(shift[1]), int(inliers.sum())


def place_layer(canvas_h, canvas_w, image, mask, origin):
    color = np.zeros((canvas_h, canvas_w, 3), np.float32)
    weight = np.zeros((canvas_h, canvas_w), np.float32)
    x = int(round(origin[0]))
    y = int(round(origin[1]))
    height, width = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(canvas_w, x + width), min(canvas_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return color, weight
    ix0, iy0 = x0 - x, y0 - y
    ix1, iy1 = ix0 + (x1 - x0), iy0 + (y1 - y0)
    sub_mask = mask[iy0:iy1, ix0:ix1].astype(np.uint8)
    if sub_mask.max() == 0:
        return color, weight
    dist = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5).astype(np.float32) ** 2
    color[y0:y1, x0:x1] = image[iy0:iy1, ix0:ix1].astype(np.float32) * dist[..., None]
    weight[y0:y1, x0:x1] = dist
    return color, weight


def compose_panoramas(base, mask_base, other, mask_other, cam5_base=None, cam5_other=None):
    found = translation(base, other, mask_base, mask_other)
    if found is None:
        print("  compose alignment failed, placing cam3 panorama on the left")
        dx, dy, inliers = float(-other.shape[1] // 2), float((base.shape[0] - other.shape[0]) / 2), 0
    else:
        dx, dy, inliers = found
        print(f"  compose shift=({dx:.1f},{dy:.1f}) inliers={inliers}")
    x0 = min(0.0, dx)
    y0 = min(0.0, dy)
    x1 = max(float(base.shape[1]), dx + other.shape[1])
    y1 = max(float(base.shape[0]), dy + other.shape[0])
    canvas_w = int(round(x1 - x0))
    canvas_h = int(round(y1 - y0))
    color_base, weight_base = place_layer(canvas_h, canvas_w, base, mask_base, (-x0, -y0))
    color_other, weight_other = place_layer(canvas_h, canvas_w, other, mask_other, (dx - x0, dy - y0))
    weight = weight_base + weight_other
    valid = weight > 0
    fused = np.full((canvas_h, canvas_w, 3), 255, np.float32)
    fused[valid] = (color_base[valid] + color_other[valid]) / weight[valid, None]
    fused = np.clip(fused, 0, 255).astype(np.uint8)
    # One camera-5 rectangle, from the cam5+cam4 panorama. The other copy is
    # only used to estimate the shift; including it would widen the crop.
    cam5 = np.zeros((canvas_h, canvas_w), dtype=bool)
    paste_bool(cam5, cam5_base, (-x0, -y0))
    fused, valid, cam5 = crop_layers(fused, valid, cam5)
    return fused, valid, cam5, inliers > 0


def load_nets(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    warp = load_module(
        "dsfn_warp_network",
        ROOT / "Warp/newCodes/network.py",
        ROOT / "Warp/newCodes",
    )
    fusion = load_module(
        "dsfn_fusion_network",
        ROOT / "Fusion/newCodes/network.py",
        ROOT / "Fusion/newCodes",
    )
    if not torch.cuda.is_available():
        raise SystemExit("DSFN warping calls .cuda(); no GPU is visible")
    warp_net = warp.Network().cuda().eval()
    fusion_net = fusion.Network().cuda().eval()
    # The fusion checkpoint stores the deployed RepConv, not the train-time pair.
    for module in fusion_net.modules():
        if isinstance(module, fusion.RepConvN):
            module.fuse_convs()
            module.forward = module.forward_fuse
    warp_ckpt = torch.load(WARP_CKPT, map_location="cpu")
    fusion_ckpt = torch.load(FUSION_CKPT, map_location="cpu")
    warp_net.load_state_dict(warp_ckpt["model"], strict=True)
    fusion_net.load_state_dict(fusion_ckpt["model"], strict=True)
    print(f"loaded {WARP_CKPT.name} and {FUSION_CKPT.name}")
    return warp, fusion, warp_net, fusion_net


def _fallback_pair(reference, side, ref_extra_mask):
    canvas = np.concatenate([side, reference], axis=1)
    valid = np.ones(canvas.shape[:2], dtype=bool)
    extra = np.ones(reference.shape[:2], dtype=bool) if ref_extra_mask is None else ref_extra_mask
    cam5 = np.zeros(canvas.shape[:2], dtype=bool)
    cam5[:, side.shape[1]:side.shape[1] + extra.shape[1]] = extra
    return canvas, valid, cam5, False


@torch.no_grad()
def stitch_pair(warp, fusion, warp_net, fusion_net, reference, side, max_height, ref_extra_mask=None):
    target = getattr(stitch_pair, "input_size", None)
    if target:
        reference = cv2.resize(reference, target, interpolation=cv2.INTER_AREA)
    else:
        reference = resize_height(reference, min(max_height, reference.shape[0]))
    if ref_extra_mask is None:
        extra = None
    else:
        extra = resize_mask(ref_extra_mask, reference.shape[0], reference.shape[1])
    side = cv2.resize(side, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    print(f"  size={reference.shape[1]}x{reference.shape[0]}")
    ref_t = to_tensor(reference).cuda()
    side_t = to_tensor(side).cuda()
    try:
        warped = warp.build_output_model(warp_net, ref_t, side_t)
    except RuntimeError as exc:
        if "canvas too large" in str(exc):
            print(f"  {exc}", flush=True)
            del ref_t, side_t
            torch.cuda.empty_cache()
            raise
        print(f"  warp failed: {exc}")
        return _fallback_pair(reference, side, extra)
    height = int(warped["final_warp1"].shape[2])
    width = int(warped["final_warp1"].shape[3])
    if height * width > 5000 * 2000 or max(height, width) > 8000:
        print(f"  warp canvas too large {width}x{height}", flush=True)
        del warped, ref_t, side_t
        gc.collect()
        torch.cuda.empty_cache()
        raise RuntimeError(f"canvas too large {width}x{height}")
    mask1 = warped["final_warp1_mask"][:, :1]
    mask2 = warped["final_warp2_mask"][:, :1]
    fused = fusion.build_model(fusion_net, warped["final_warp1"], warped["final_warp2"], mask1, mask2)
    image = fused["stitched_image"][0].detach().float().cpu().numpy().transpose(1, 2, 0)
    image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
    valid = ((mask1[0, 0] + mask2[0, 0]) > 0.5).detach().cpu().numpy()
    ref_mask = (mask1[0, 0] > 0.5).detach().cpu().numpy()
    tracked = np.ones(reference.shape[:2], dtype=bool) if extra is None else extra
    cam5 = embed_extra_mask(ref_mask, tracked)
    image = paint_white(image, valid)
    image, valid, cam5 = crop_layers(image, valid, cam5)
    del warped, fused, ref_t, side_t, mask1, mask2
    gc.collect()
    torch.cuda.empty_cache()
    return image, valid, cam5, True


def montage(cam3, cam4, cam5, step4, step3, final):
    target_h = 220

    def row(images):
        return np.concatenate([resize_height(im, target_h) for im in images], axis=1)

    top = row([cam3, cam5, cam4])
    bottom = row([step3, step4, final])
    width = max(top.shape[1], bottom.shape[1])

    def pad(image):
        if image.shape[1] == width:
            return image
        canvas = np.zeros((image.shape[0], width, 3), dtype=image.dtype)
        canvas[:, : image.shape[1]] = image
        return canvas

    return np.concatenate([pad(top), pad(bottom)], axis=0)


def stitch_frame(modules, dataset, scene, frame, save_root, max_height, chain_only=False):
    warp, fusion, warp_net, fusion_net = modules
    cam5, p5 = load_cam(scene, frame, 5)
    cam4, p4 = load_cam(scene, frame, 4)
    cam3, p3 = load_cam(scene, frame, 3)
    print(f"{dataset} {scene.name} frame {frame_token(frame)}")
    print(f"  inputs {p5.name} {p4.name} {p3.name} {cam5.shape[1]}x{cam5.shape[0]}")
    height = min(max_height, cam5.shape[0])
    cam5_s = resize_height(cam5, height)
    cam4_s = resize_height(cam4, height)
    cam3_s = resize_height(cam3, height)
    print("  stitch cam5 + cam4")
    step4, mask4, cam5_4, ok4 = stitch_pair(warp, fusion, warp_net, fusion_net, cam5_s, cam4_s, height)
    if chain_only:
        ok3, ok_align = False, False
        final, valid, cam5 = step4, mask4, cam5_4
    else:
        print("  stitch cam5 + cam3")
        _step3, _mask3, _cam5_3, ok3 = stitch_pair(warp, fusion, warp_net, fusion_net, cam5_s, cam3_s, height)
        print("  align the two panoramas on cam5")
        final, valid, cam5, ok_align = compose_panoramas(step4, mask4, _step3, _mask3, cam5_4, _cam5_3)
    out_dir = Path(save_root) / dataset / scene.name / f"frame_{frame_token(frame)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "stitched.jpg"), final)
    native = crop_cam5_triple(final, cam5)
    gt_file = None
    if dataset in SPARSE_WIDE_REL:
        gt_file = resolve_data_root(SPARSE_WIDE_REL[dataset]) / scene.name / "rgb" / f"{frame_token(frame)}_5_sparse_wide.png"
    if gt_file is not None and gt_file.is_file():
        gt = cv2.imread(str(gt_file), cv2.IMREAD_COLOR)
        crop = resize_to_gt(native, (gt.shape[1], gt.shape[0]))
    else:
        crop = native
    cv2.imwrite(str(out_dir / "crop.png"), crop)
    print(
        f"  saved {out_dir / 'stitched.jpg'} {final.shape[1]}x{final.shape[0]} "
        f"and {out_dir / 'crop.png'} {crop.shape[1]}x{crop.shape[0]} ok=({ok4},{ok3},{ok_align})"
    )
    torch.cuda.empty_cache()
    return out_dir


def main():
    args = parse_args()
    stitch_pair.input_size = (518, 434) if "1224" in args.dataset else (518, 294)
    scene = scene_dir(args.dataset, args.scene)
    modules = load_nets(args.gpu)
    for frame in args.frame:
        stitch_frame(modules, args.dataset, scene, frame, args.save_root, args.max_height, args.chain_only)


if __name__ == "__main__":
    main()
