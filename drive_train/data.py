"""Two-image driving pairs for DSFN warp / fusion training.

Each training item is one baseline pair: camera 5 with camera 4, or camera 5
with camera 3. The loader shuffles scenes and frames. Default finetune runs:

  nuscenes    1 epoch, nuScenes train scenes
  lyft        3 epochs, lyft 1224 and 1920 train scenes mixed
  widedrive3  3 epochs, WideDriveTrain (train.txt, sibling of WideDriveVal)
"""
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
_LEGACY_WORK = Path("/home/test/LIVA/XZP/FeedForward/work2/0923_synGithub")
WORK = _LEGACY_WORK if (_LEGACY_WORK / "datasets").exists() else ROOT.parent
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# split=train uses that list or every scene directory under root.
# WideDriveTrain is a sibling of WideDriveVal and uses train.txt.
DATASETS = {
    "nuscenes": {
        "root": "nuscenes/processed_10Hz_v2",
        "list": "nuscenes/processed_10Hz_v2/nuScenes_Train.txt",
        "split": "train",
    },
    "lyft1920": {
        "root": "lyft/lyft_train1920_3cams",
        "list": None,
        "split": "train",
    },
    "lyft1224": {
        "root": "lyft/lyft_train1224_3cams",
        "list": None,
        "split": "train",
    },
    "ddad": {
        "root": "ddad_process/train",
        "list": None,
        "split": "train",
    },
    "syn_nuscenes": {
        "root": "syn_nuscenes/processed_10Hz/trainval",
        "list": "syn_nuscenes/processed_10Hz/trainval/syn_nuscenes_train.txt",
        "split": "train",
    },
    "widedrive3": {
        "root": "WideDrive3_processed/WideDriveTrain",
        "list": "WideDrive3_processed/WideDriveTrain/train.txt",
        "split": "train",
    },
}

# Public training targets. lyft is both resolutions in one shuffled loader.
GROUPS = {
    "lyft": ("lyft1920", "lyft1224"),
}
DEFAULT_EPOCHS = {
    "nuscenes": 1,
    "lyft": 3,
    "lyft1920": 3,
    "lyft1224": 3,
    "widedrive3": 3,
}

PAIRS = ((5, 4), (5, 3))
def input_hw(name):
    return (518, 434) if "1224" in name else (518, 294)


INPUT_SIZE = (518, 294)
# Sparse lidar depth is stored in meters. Map valid pixels into the 0-255
# grayscale range the original DepthAnything loss expects. 0 stays invalid.
DEPTH_MAX_M = 80.0


def resolve_data_path(relative):
    relative = Path(relative)
    for root in (ROOT / "datasets", WORK / "datasets"):
        path = root / relative
        if path.exists():
            return path
    raise SystemExit(f"path not found: {relative}")


def dataset_names():
    return ["nuscenes", "lyft", "widedrive3", "lyft1920", "lyft1224", "ddad", "syn_nuscenes"]


def resolve_sources(name):
    if name in GROUPS:
        return list(GROUPS[name])
    if name in DATASETS:
        return [name]
    raise SystemExit(f"unknown dataset {name}; choose from {dataset_names()}")


def default_epochs(name):
    if name not in DEFAULT_EPOCHS:
        raise SystemExit(
            f"{name} has no default epoch count; pass --max-epoch. Defaults: {DEFAULT_EPOCHS}"
        )
    return DEFAULT_EPOCHS[name]


def uses_val_split(name):
    return bool(DATASETS[name].get("allow_val"))


def _read_list(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def train_scenes(dataset, allow_val=False):
    spec = DATASETS[dataset]
    allow_val = allow_val or uses_val_split(dataset)
    if spec["split"] != "train" and not allow_val:
        raise SystemExit(
            f"{dataset} has no train split on disk (only {spec['split']}: {spec['root']}). "
            "Pass --allow-val to use it anyway."
        )
    root = resolve_data_path(spec["root"])
    if spec["list"]:
        names = _read_list(resolve_data_path(spec["list"]))
    else:
        names = sorted(
            p.name for p in root.iterdir() if p.is_dir() and (p / "images").is_dir()
        )
    scenes = []
    for name in names:
        path = root / name
        if not (path / "images").is_dir():
            continue
        scenes.append(path)
    if not scenes:
        raise SystemExit(f"no scenes for {dataset} under {root}")
    return scenes


def _stem_map(folder, suffixes):
    found = {}
    if not folder.is_dir():
        return found
    with os.scandir(folder) as it:
        for entry in it:
            name = entry.name
            for suffix in suffixes:
                if name.endswith(suffix):
                    found[name[: -len(suffix)]] = name
                    break
    return found


def build_index(dataset, allow_val=False, max_scenes=0):
    scenes = train_scenes(dataset, allow_val=allow_val)
    if max_scenes and max_scenes > 0:
        scenes = scenes[:max_scenes]
    samples = []
    skipped = 0
    for scene in scenes:
        images = _stem_map(scene / "images", (".jpg", ".png"))
        depths = _stem_map(scene / "depth_map", (".npz", ".png", ".jpg"))
        frames = {}
        for stem in images:
            if "_" not in stem:
                continue
            frame, cam = stem.rsplit("_", 1)
            if cam in {"3", "4", "5"}:
                frames.setdefault(frame, set()).add(cam)
        for frame, cams in frames.items():
            if cams != {"3", "4", "5"}:
                skipped += 1
                continue
            if any(f"{frame}_{cam}" not in depths for cam in ("3", "4", "5")):
                skipped += 1
                continue
            for cam_a, cam_b in PAIRS:
                samples.append(
                    {
                        "scene": scene.name,
                        "frame": frame,
                        "cam_a": cam_a,
                        "cam_b": cam_b,
                    }
                )
    return samples, len(scenes), skipped


def load_or_build_index(dataset, allow_val=False, max_scenes=0, refresh=False):
    # A scene cap is a debug slice, so do not reuse or overwrite the full cache.
    use_cache = not max_scenes
    allow_val = allow_val or uses_val_split(dataset)
    cache_path = CACHE_DIR / f"{dataset}_{'valok' if allow_val else 'train'}.json"
    if use_cache and cache_path.is_file() and not refresh:
        payload = json.loads(cache_path.read_text())
        return payload["samples"], payload["scenes"], payload["skipped"], cache_path
    samples, scenes, skipped = build_index(dataset, allow_val=allow_val, max_scenes=max_scenes)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"samples": samples, "scenes": scenes, "skipped": skipped})
        )
    return samples, scenes, skipped, cache_path


def _image_file(scene, frame, cam):
    folder = scene / "images"
    for suffix in (".jpg", ".png"):
        path = folder / f"{frame}_{cam}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing image {folder}/{frame}_{cam}")


def _depth_file(scene, frame, cam):
    folder = scene / "depth_map"
    for suffix in (".npz", ".png", ".jpg"):
        path = folder / f"{frame}_{cam}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing depth {folder}/{frame}_{cam}")


def load_rgb(path, size=INPUT_SIZE):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = (image / 127.5) - 1.0
    return np.transpose(image, (2, 0, 1))


def load_depth(path, size=INPUT_SIZE):
    if path.suffix == ".npz":
        depth = np.load(path)["depth"].astype(np.float32)
        out = np.zeros(depth.shape, dtype=np.float32)
        valid = depth > 0
        out[valid] = np.clip(depth[valid], 0.0, DEPTH_MAX_M) / DEPTH_MAX_M * 255.0
        depth = out
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if depth is None:
            raise FileNotFoundError(path)
        depth = depth.astype(np.float32)
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (512, 512), interpolation=cv2.INTER_NEAREST)
    return depth[np.newaxis, :, :]


class DrivePairDataset(Dataset):
    """Baseline pair dataset. __getitem__ randomly swaps the two views."""

    def __init__(
        self,
        dataset,
        allow_val=False,
        max_scenes=0,
        max_pairs=0,
        exchange=True,
        seed=0,
        refresh_index=False,
    ):
        sources = resolve_sources(dataset)
        samples = []
        scenes = 0
        skipped = 0
        caches = []
        roots = {}
        for source in sources:
            part, n_scenes, n_skip, cache_path = load_or_build_index(
                source,
                allow_val=allow_val or uses_val_split(source),
                max_scenes=max_scenes,
                refresh=refresh_index,
            )
            for item in part:
                row = dict(item)
                row["source"] = source
                samples.append(row)
            scenes += n_scenes
            skipped += n_skip
            caches.append(cache_path)
            roots[source] = resolve_data_path(DATASETS[source]["root"])
        if not samples:
            raise SystemExit(f"{dataset} produced no cam5+cam4 / cam5+cam3 pairs")
        if max_pairs and max_pairs > 0 and max_pairs < len(samples):
            rng = random.Random(seed)
            rng.shuffle(samples)
            samples = samples[:max_pairs]
        self.samples = samples
        self.scenes = scenes
        self.skipped = skipped
        self.cache_path = caches[0] if len(caches) == 1 else caches
        self.exchange = exchange
        self.roots = roots
        self.sources = sources
        self.dataset = dataset

    def batch_sizes(self):
        return [input_hw(item["source"]) for item in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        scene = self.roots[item["source"]] / item["scene"]
        frame = item["frame"]
        cam_a = item["cam_a"]
        cam_b = item["cam_b"]
        size = input_hw(item["source"])
        image_a = load_rgb(_image_file(scene, frame, cam_a), size)
        image_b = load_rgb(_image_file(scene, frame, cam_b), size)
        depth_a = load_depth(_depth_file(scene, frame, cam_a), size)
        depth_b = load_depth(_depth_file(scene, frame, cam_b), size)
        image_a = torch.from_numpy(image_a)
        image_b = torch.from_numpy(image_b)
        depth_a = torch.from_numpy(depth_a)
        depth_b = torch.from_numpy(depth_b)
        if self.exchange and random.randint(0, 1) == 1:
            return image_b, image_a, depth_b, depth_a
        return image_a, image_b, depth_a, depth_b
