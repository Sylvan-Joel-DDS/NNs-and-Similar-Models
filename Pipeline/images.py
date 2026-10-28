"""RealWaste image data: discovery, stratified split, and a one-off cached uint8 array of every split.

Why: decoding ~4.7k 524x524 JPEGs takes minutes on a CPU and used to be repeated after every kernel restart.
Here every image is decoded + resized ONCE (multi-threaded), stored as .npy, and re-loaded in seconds.
The same `load_image` function is used for training data and for images uploaded to the assistant, so there is
no train/serve preprocessing mismatch.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from . import config
from .config import CATEGORIES, IMG_SIZE

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(root: Path | None = None) -> pd.DataFrame:
    root = Path(root) if root else config.find_image_root()
    rows = [(str(p), c, CATEGORIES.index(c)) for c in CATEGORIES
            for p in sorted((root / c).iterdir()) if p.suffix.lower() in IMG_EXT]
    return pd.DataFrame(rows, columns=["path", "category", "label"])


def load_image(src, size: int = IMG_SIZE) -> np.ndarray:
    """Path | PIL image | ndarray (gray/RGB/RGBA, uint8 or float 0-255) -> uint8 array (size, size, 3)."""
    if isinstance(src, (str, Path)):
        if not Path(src).exists():
            raise FileNotFoundError(f"Image not found: {src}")
        with Image.open(src) as im:
            im = im.convert("RGB")
            return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)
    if isinstance(src, Image.Image):
        return np.asarray(src.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
    if isinstance(src, np.ndarray):
        arr = src
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported array shape {src.shape}")
        im = Image.fromarray(np.clip(arr[..., :3], 0, 255).astype(np.uint8))
        return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.uint8)
    raise TypeError(f"Unsupported image input type: {type(src).__name__}")


def load_array(paths, workers: int = 8) -> np.ndarray:
    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(load_image, paths)))


def stratified_split(df: pd.DataFrame, val: float = 0.10, test: float = 0.10, seed: int = config.SEED) -> dict:
    """80/10/10, stratified by class: every class is in every split, no overlap, fixed order."""
    train_df, rest = train_test_split(df, test_size=val + test, stratify=df.label, random_state=seed)
    val_df, test_df = train_test_split(rest, test_size=test / (val + test), stratify=rest.label, random_state=seed)
    return {k: v.sort_values("path").reset_index(drop=True) for k, v in
            dict(train=train_df, val=val_df, test=test_df).items()}


@dataclass
class ImageData:
    fingerprint: str
    class_names: list
    X_train: np.ndarray; y_train: np.ndarray
    X_val: np.ndarray;   y_val: np.ndarray
    X_test: np.ndarray;  y_test: np.ndarray
    paths: dict                       # split -> list of file paths (aligned with X_*)

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({s: np.bincount(getattr(self, f"y_{s}"), minlength=len(self.class_names))
                             for s in ("train", "val", "test")}, index=self.class_names)


def get_image_data(force: bool = False, verbose: bool = True) -> ImageData:
    """Stratified split + cached arrays. First call decodes the images (~1-2 min); later calls take seconds."""
    config.ensure_dirs()
    df = list_images()
    root = config.find_image_root()
    fp = config.fingerprint([Path(p).relative_to(root).as_posix() for p in df.path], IMG_SIZE, config.SEED, "80/10/10")
    folder = config.CACHE_DIR / f"images_{fp}"
    splits = stratified_split(df)
    t0 = time.time()
    if folder.exists() and not force and all((folder / f"X_{s}.npy").exists() for s in splits):
        X = {s: np.load(folder / f"X_{s}.npy") for s in splits}
        src = "cache"
    else:
        folder.mkdir(parents=True, exist_ok=True)
        X = {}
        for s, sdf in splits.items():
            X[s] = load_array(sdf.path.tolist())
            np.save(folder / f"X_{s}.npy", X[s])
        src = "decoded from JPEG"
    if verbose:
        print(f"Image arrays ({src}) in {time.time() - t0:.1f}s | "
              + " / ".join(f"{s}: {len(X[s])}" for s in splits))
    return ImageData(fp, list(CATEGORIES),
                     X["train"], splits["train"].label.values, X["val"], splits["val"].label.values,
                     X["test"], splits["test"].label.values,
                     {s: splits[s].path.tolist() for s in splits})
