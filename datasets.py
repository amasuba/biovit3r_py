"""Dataset definitions for BioViT3R.

This module contains simple dataset classes that can be used to train and
evaluate the different parts of the BioViT3R pipeline. The datasets are
designed to be flexible and extensible – you can subclass them to support
different data formats and modalities (e.g. RGB, depth, masks, camera
poses).

Note: These classes depend on PyTorch's ``torch.utils.data.Dataset`` class.
If PyTorch is not installed an ``ImportError`` will be raised. Install
PyTorch before using these datasets.

Example:

    >>> from biovit3r.datasets import MultiViewPlantDataset
    >>> dataset = MultiViewPlantDataset(root_dir="/path/to/data", num_views=12)
    >>> rgb, depth, pose, label = dataset[0]

The example above loads the first sample, returning the RGB image stack,
depth maps, camera poses and plant species label.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional


try:
    import torch
    from torch.utils.data import Dataset
    from PIL import Image
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "biovit3r.datasets requires PyTorch. Please install PyTorch and PIL before use."
    ) from e


class MultiViewPlantDataset(Dataset):
    """A dataset for multi-view plant images and associated metadata.

    This dataset assumes a directory structure like the following::

        root_dir/
            sample_000/
                meta.json
                rgb/
                    view_0.png
                    view_1.png
                    ...
                depth/
                    view_0.npy  # optional depth maps
                    view_1.npy
                    ...
                pose/
                    view_0.txt  # optional extrinsic pose matrices
                    view_1.txt
                    ...

    Each ``meta.json`` file contains metadata about the sample, such as
    species name or numeric label, and any other annotations. It should
    include at least a ``label`` field mapping to an integer index.

    Args:
        root_dir: Root directory containing one subdirectory per sample.
        num_views: Number of views to load per sample. If a sample has fewer
            views than this number, all available views are returned. If it
            has more, a subset is randomly selected during loading.
        transform: Optional transformation applied to each RGB image.
        preload: If True, load all data into memory at construction time.

    Returns:
        A tuple ``(rgb_views, depth_views, pose_views, label)`` where:
            - ``rgb_views`` is a list of ``num_views`` images (converted to
              tensors if a transform is provided).
            - ``depth_views`` is a list of depth arrays (or ``None`` if depth
              is not available).
            - ``pose_views`` is a list of 4×4 extrinsic matrices (or ``None``
              if pose is not available).
            - ``label`` is an integer species label.
    """

    def __init__(
        self,
        root_dir: str,
        num_views: int = 6,
        transform: Optional[object] = None,
        preload: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_views = num_views
        self.transform = transform
        self.samples = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])
        self.preload = preload
        self._cache: List[Tuple[List[object], List[Optional[object]], List[Optional[object]], int]] = []

        if preload:
            for idx in range(len(self.samples)):
                self._cache.append(self._load_sample(idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[object], List[Optional[object]], List[Optional[object]], int]:
        if self.preload:
            return self._cache[idx]
        return self._load_sample(idx)

    def _load_sample(
        self, idx: int
    ) -> Tuple[List[object], List[Optional[object]], List[Optional[object]], int]:
        sample_dir = self.samples[idx]
        meta_path = sample_dir / "meta.json"
        with meta_path.open("r") as f:
            meta = json.load(f)
        label = int(meta.get("label", -1))

        # list available views
        rgb_dir = sample_dir / "rgb"
        depth_dir = sample_dir / "depth"
        pose_dir = sample_dir / "pose"

        rgb_paths = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
        depth_paths = sorted(depth_dir.glob("*.npy")) if depth_dir.exists() else []
        pose_paths = sorted(pose_dir.glob("*.txt")) if pose_dir.exists() else []

        # select subset of views if necessary
        if self.num_views > 0 and len(rgb_paths) > self.num_views:
            # randomly sample without replacement
            indices = torch.randperm(len(rgb_paths))[: self.num_views].tolist()
            rgb_paths = [rgb_paths[i] for i in indices]
            if depth_paths:
                depth_paths = [depth_paths[i] for i in indices]
            if pose_paths:
                pose_paths = [pose_paths[i] for i in indices]
        # load images
        rgb_views = []
        depth_views = []
        pose_views = []
        for i, rgb_path in enumerate(rgb_paths):
            img = Image.open(rgb_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            rgb_views.append(img)
            if depth_paths:
                import numpy as np  # local import to avoid global dependency when depth not used
                depth = np.load(depth_paths[i])
                depth_views.append(depth)
            else:
                depth_views.append(None)
            if pose_paths:
                import numpy as np
                pose = np.loadtxt(pose_paths[i])
                if pose.shape == (4, 4):
                    pose_views.append(pose)
                else:
                    # if pose is 3x4, convert to 4x4 with last row [0,0,0,1]
                    M = np.eye(4)
                    M[: pose.shape[0], : pose.shape[1]] = pose
                    pose_views.append(M)
            else:
                pose_views.append(None)
        return rgb_views, depth_views, pose_views, label