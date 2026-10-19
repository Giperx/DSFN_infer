"""Histogram-matched scores against a sparse wide-FOV GT mask.

Full-image scores follow metrics/full_hm.py in the reference project.
L/C/R scores follow the regional HM path: each third of the width is matched
on its own sparse-GT mask, then L+R and L+R+C are means of those region means.
LPIPS is not computed. The render mask is not used.
"""
import numpy as np
from PIL import Image


REGION_NAMES = ("Left", "Center", "Right")


def match_channel(source, template):
    """Per-channel CDF match, the same mapping as skimage.exposure.match_histograms."""
    _src_values, src_inverse, src_counts = np.unique(
        source.ravel(), return_inverse=True, return_counts=True
    )
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size
    mapped = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return mapped[src_inverse].reshape(source.shape)


def match_histograms(pred, target):
    matched = np.empty_like(pred, dtype=np.float64)
    for channel in range(pred.shape[-1]):
        matched[..., channel] = match_channel(pred[..., channel], target[..., channel])
    return matched


def frame_hm(render, gt, mask):
    """Match rendered pixels to GT pixels, then score that pair. Images are [0, 1]."""
    if mask.sum() == 0:
        return None
    pred = render[mask]
    target = gt[mask]
    matched = match_histograms(pred, target)
    diff = matched - target
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    c1 = 0.01 ** 2
    ssim = float(np.mean((2 * matched * target + c1) / (matched ** 2 + target ** 2 + c1)))
    return {
        "hm_mae": mae,
        "hm_rmse": float(np.sqrt(mse)),
        "hm_psnr": float(10.0 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "hm_ssim": ssim,
    }


def mean_rows(rows):
    if not rows:
        return None
    finite = [row["hm_psnr"] for row in rows if np.isfinite(row["hm_psnr"])]
    return {
        "hm_psnr": float(np.mean(finite)) if finite else float("inf"),
        "hm_ssim": float(np.mean([row["hm_ssim"] for row in rows])),
        "hm_mae": float(np.mean([row["hm_mae"] for row in rows])),
        "hm_rmse": float(np.mean([row["hm_rmse"] for row in rows])),
        "n": len(rows),
    }


def format_stats(stats, suffix):
    if stats is None:
        return [
            f"HM-PSNR-{suffix}: n/a",
            f"HM-SSIM-{suffix}: n/a",
            f"HM-MAE-{suffix}: n/a",
            f"HM-RMSE-{suffix}: n/a",
            "Frames: 0",
        ]
    return [
        f"HM-PSNR-{suffix}: {stats['hm_psnr']:.2f}",
        f"HM-SSIM-{suffix}: {stats['hm_ssim']:.4f}",
        f"HM-MAE-{suffix}: {stats['hm_mae'] * 255:.2f}",
        f"HM-RMSE-{suffix}: {stats['hm_rmse'] * 255:.2f}",
        f"Frames: {stats['n']}",
    ]


def region_bounds(width):
    third = width // 3
    return {
        "Left": (0, third),
        "Center": (third, third * 2),
        "Right": (third * 2, width),
    }


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def load_mask(path, shape_hw):
    mask = np.asarray(Image.open(path).convert("L"))
    if mask.shape != tuple(shape_hw):
        mask = np.asarray(Image.fromarray(mask).resize((shape_hw[1], shape_hw[0]), Image.NEAREST))
    return mask > 127


def score_pair(render, gt, mask):
    full = frame_hm(render, gt, mask)
    regions = {}
    for name, (x0, x1) in region_bounds(render.shape[1]).items():
        regions[name] = frame_hm(render[:, x0:x1], gt[:, x0:x1], mask[:, x0:x1])
    return full, regions


def mean_of_region_means(region_rows, names):
    stats = []
    for name in names:
        summary = mean_rows(region_rows.get(name) or [])
        if summary is None:
            return None
        stats.append(summary)
    return {
        "hm_psnr": float(np.mean([item["hm_psnr"] for item in stats])),
        "hm_ssim": float(np.mean([item["hm_ssim"] for item in stats])),
        "hm_mae": float(np.mean([item["hm_mae"] for item in stats])),
        "hm_rmse": float(np.mean([item["hm_rmse"] for item in stats])),
        "n": int(np.mean([item["n"] for item in stats])),
    }


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    print(text, end="")
    print(f"Wrote {path}", flush=True)
    return text
