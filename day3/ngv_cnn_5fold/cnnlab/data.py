from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from .common import read_json, write_json, sha_file, ids_hash

CLASSES = {
    "mnist": [str(i) for i in range(10)],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "catsdogs": ["cat", "dog"],
}
SHAPES = {"mnist": (28, 28, 1), "cifar10": (32, 32, 3), "catsdogs": (96, 96, 3)}
DATASET_SIZES = {"mnist": 2000, "cifar10": 10000, "catsdogs": 1000}
SPLIT_SEED = 20260929


def validate_arrays(images, labels, ids, domain: str, expected: int | None = None) -> None:
    expected = DATASET_SIZES[domain] if expected is None else expected
    if images.dtype != np.uint8 or images.shape != (expected, *SHAPES[domain]):
        raise ValueError(f"{domain}: require uint8 {(expected, *SHAPES[domain])}, got {images.dtype} {images.shape}")
    if labels.shape != (expected,) or labels.dtype.kind not in "iu":
        raise ValueError("Invalid labels")
    k = len(CLASSES[domain])
    if not np.array_equal(np.bincount(labels, minlength=k), np.repeat(expected // k, k)):
        raise ValueError("Dataset must have exactly balanced class counts")
    if len(ids) != expected or len(set(map(str, ids))) != expected:
        raise ValueError("Duplicate/missing source IDs")
    hashes = [hashlib.sha256(x.tobytes()).hexdigest() for x in images]
    if len(set(hashes)) != expected:
        raise ValueError("Exact duplicate image pixels detected; do not duplicate examples to reach the required count")


def make_split(labels, ids, seed: int = SPLIT_SEED) -> dict:
    idx = np.arange(len(labels))
    train, test = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    train, test = np.sort(train), np.sort(test)
    return {"seed": seed, "train_pool": train.tolist(), "test": test.tolist(),
            "train_pool_hash": ids_hash(ids[train]), "test_hash": ids_hash(ids[test])}


def save_bundle(out: Path, domain: str, images, labels, ids, source: dict) -> None:
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    ids = np.asarray(ids, dtype=str)
    validate_arrays(images, labels, ids, domain)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{domain}.npz"
    with p.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, images=images, labels=labels, ids=ids)
    p.with_suffix(".tmp").replace(p)
    meta = {"domain": domain, "n": len(labels), "class_names": CLASSES[domain],
            "image_shape": list(images.shape[1:]), "npz_sha256": sha_file(p),
            "source": source, "split": make_split(labels, ids),
            "protocol": f"{len(labels)}-sample educational subset; {int(.8 * len(labels))} development / {len(labels)-int(.8 * len(labels))} internal holdout; "
                        "five-fold stratified CV only within development; not an official benchmark test score"}
    write_json(out / f"{domain}.json", meta)


def load_bundle(root: Path, domain: str, expected: int | None = None):
    p, m = root / f"{domain}.npz", root / f"{domain}.json"
    if not p.is_file() or not m.is_file():
        raise FileNotFoundError(f"Missing real dataset: {p}. This code does NOT download during training. "
                                "Prepare sources with 00_build_data_bundle.py first, or use a verified WITH_DATA ZIP.")
    meta = read_json(m)
    if sha_file(p) != meta["npz_sha256"]:
        raise ValueError(f"Data checksum mismatch: {p}. Restore the correct bundle.")
    with np.load(p, allow_pickle=False) as z:
        x, y, ids = z["images"], z["labels"], z["ids"]
    validate_arrays(x, y, ids, domain, expected=expected)
    s = meta["split"]
    train, test = np.asarray(s["train_pool"]), np.asarray(s["test"])
    if len(set(train) & set(test)) or sorted([*train, *test]) != list(range(len(y))):
        raise ValueError("Split is overlapping, missing, or out of range")
    if len(train) != int(.8 * len(y)) or len(test) != len(y) - len(train):
        raise ValueError("Require 80/20 split")
    canonical = make_split(y, ids, int(s["seed"]))
    if s != canonical:
        raise ValueError("Saved split was modified; the fixed train/test split must remain unchanged")
    return x, y, ids, meta


def nested_subset(pool, labels, count: int, seed: int = 421) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    classes = np.unique(labels[pool])
    if count > len(pool) or count < 5 * len(classes) or count % len(classes):
        raise ValueError(f"Training-pool size must be <={len(pool)}, >=5 per class, and a multiple of class count")
    rng = np.random.default_rng(seed)
    # Same per-class permutations for every size -> 200 is contained in 400, etc.
    chosen = [rng.permutation(pool[labels[pool] == c])[:count // len(classes)] for c in classes]
    out = np.sort(np.concatenate(chosen))
    if len(out) != count:
        raise ValueError("Not enough examples in one class")
    return out


def folds_for(pool, labels, seed: int):
    pool = np.asarray(pool)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return [(pool[a], pool[b]) for a, b in skf.split(pool, labels[pool])]


class ArrayImages(Dataset):
    def __init__(self, images, labels, indices, domain, augmentation=False):
        self.images, self.labels, self.indices = images, labels, np.asarray(indices)
        channels = images.shape[-1]
        shape = images.shape[1]
        steps = []
        if augmentation:
            if domain == "mnist":
                steps += [T.RandomAffine(degrees=8, translate=(.06, .06))]
            elif domain == "cifar10":
                steps += [T.RandomCrop(shape, padding=4), T.RandomHorizontalFlip()]
            else:
                steps += [T.RandomHorizontalFlip(), T.RandomAffine(degrees=8, translate=(.04, .04))]
        steps += [T.ToTensor(), T.Normalize([.5] * channels, [.5] * channels)]
        self.transform = T.Compose(steps)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        im = self.images[idx]
        pil = Image.fromarray(im[:, :, 0] if im.shape[-1] == 1 else im)
        return self.transform(pil), int(self.labels[idx])


def make_loader(x, y, idx, domain, batch, train, seed, augment=True, cuda=False):
    import torch
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(ArrayImages(x, y, idx, domain, augmentation=(train and augment)),
                      batch_size=int(batch), shuffle=bool(train), num_workers=0,
                      pin_memory=bool(cuda), generator=generator, drop_last=False)
