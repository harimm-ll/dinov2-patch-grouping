from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io import read_csv, write_csv, write_json


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(array, (2, 0, 1))


def _device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_features(
    image_csv: Path,
    output_dir: Path,
    image_root: Path | None = None,
    model_name: str = "dinov2_vits14",
    repo: str = "facebookresearch/dinov2",
    layer: int = 8,
    image_size: int = 518,
    patch_size: int = 14,
    batch_size: int = 4,
    device: str = "auto",
    torch_home: Path | None = None,
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Feature extraction requires PyTorch.") from exc

    rows = read_csv(image_csv)
    required = {"image_id", "image_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("images.csv must contain image_id and image_path columns")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("image_id values must be unique")
    if image_size % patch_size:
        raise ValueError("image_size must be divisible by patch_size")

    root = image_root or image_csv.parent
    resolved = []
    for index, row in enumerate(rows):
        path = Path(row["image_path"])
        full_path = path if path.is_absolute() else root / path
        if not full_path.exists():
            raise FileNotFoundError(full_path)
        resolved.append({"image_index": index, "image_id": row["image_id"], "image_path": row["image_path"], "full_path": full_path})

    class ImageDataset(Dataset):
        def __len__(self) -> int:
            return len(resolved)

        def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
            return _load_image(resolved[index]["full_path"], image_size), index

    output_dir.mkdir(parents=True, exist_ok=True)
    if torch_home is not None:
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(torch_home.resolve()))

    selected_device = _device(torch, device)
    model = torch.hub.load(repo, model_name, trust_repo=True).to(selected_device).eval()
    loader = DataLoader(ImageDataset(), batch_size=batch_size, shuffle=False, num_workers=0)
    feature_batches: list[np.ndarray] = []
    ordered_indices: list[int] = []
    for batch, indices in loader:
        batch = batch.to(device=selected_device, dtype=torch.float32)
        with torch.no_grad():
            output = model.get_intermediate_layers(
                batch,
                n=[layer],
                reshape=True,
                return_class_token=False,
                norm=True,
            )[0]
        feature_batches.append(output.permute(0, 2, 3, 1).cpu().numpy().astype(np.float16))
        ordered_indices.extend(int(value) for value in indices)

    features = np.concatenate(feature_batches, axis=0)
    if ordered_indices != list(range(len(resolved))):
        raise RuntimeError("The feature rows are not in manifest order")

    index_rows = [
        {"image_index": item["image_index"], "image_id": item["image_id"], "image_path": item["image_path"]}
        for item in resolved
    ]
    write_csv(output_dir / "image_index.csv", index_rows)
    metadata = {
        "model": model_name,
        "repo": repo,
        "layer": layer,
        "image_size": image_size,
        "patch_size": patch_size,
        "grid_height": int(features.shape[1]),
        "grid_width": int(features.shape[2]),
        "embedding_dim": int(features.shape[3]),
        "image_count": len(resolved),
        "preprocessing": "RGB, bicubic resize, ImageNet normalization",
    }
    np.savez_compressed(
        output_dir / f"patch_layer_{layer}.npz",
        features=features,
        metadata=np.asarray(json.dumps(metadata)),
    )
    write_json(output_dir / "manifest.json", metadata)
    return metadata

