"""3D reconstruction and pose estimation module.

This module implements the :class:`DepthPoseEstimator`, a neural network
designed to jointly predict per-pixel depth maps and camera pose from
multi-view RGB images. It utilises a Vision Transformer backbone to
extract rich feature representations and adds lightweight decoder heads
for depth and pose regression. A helper function, :func:`merge_point_clouds`,
provides a simple utility to fuse the predicted depth maps and poses into
a single point cloud.

While the architecture here is intentionally simplified relative to state-
of-the-art models like DUSt3R, it serves as a functional baseline and
illustrates how to connect a transformer to downstream geometry tasks.
Researchers are encouraged to adapt and expand upon this module to better
fit their needs (e.g. by adding convolutional refinement, attention-based
fusion or multi-scale processing).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import math
import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "biovit3r.models.reconstruction requires PyTorch. Install PyTorch to use this module."
    ) from e

from .backbone import VisionTransformerBackbone


def _normalize_rotation(R: torch.Tensor) -> torch.Tensor:
    """Normalize a 3x3 rotation matrix via SVD.

    Given a batch of approximate rotation matrices, this function projects
    them onto the nearest orthonormal matrices using singular value
    decomposition. This is helpful when converting from a continuous
    representation (e.g. 6D vectors) to SO(3).

    Args:
        R: Tensor of shape ``(B, 3, 3)`` representing unnormalized rotations.

    Returns:
        Tensor of shape ``(B, 3, 3)`` with orthonormal rows and det=1.
    """
    U, _, Vt = torch.linalg.svd(R)
    # Ensure right-handedness by flipping sign if determinant < 0
    det = torch.det(U @ Vt)
    sign = torch.sign(det).unsqueeze(-1).unsqueeze(-1)
    S = torch.eye(3, device=R.device).unsqueeze(0).repeat(R.shape[0], 1, 1)
    S[:, -1, -1] = sign.squeeze()
    R_ortho = U @ S @ Vt
    return R_ortho


class DepthPoseEstimator(nn.Module):
    """Predicts per-pixel depth maps and camera pose from RGB images.

    This model consists of a ViT backbone followed by separate heads for
    depth estimation and pose regression. The depth head is a simple
    convolutional decoder that upsamples the patch embeddings to the
    resolution of the patch grid; the pose head maps the pooled [CLS]
    features to a 6D vector parameterising rotation and translation.

    Parameters
    ----------
    backbone: Optional[VisionTransformerBackbone]
        ViT backbone used to extract feature maps and pooled features. If
        ``None``, a default ``vit_b_16`` backbone is created.
    decode_depth: bool
        Whether to include the depth head. Set to ``False`` if only pose
        estimation is required.
    decode_pose: bool
        Whether to include the pose head. Set to ``False`` if external
        pose information will be provided.
    upsample_factor: int
        Factor by which to upsample the depth prediction relative to the
        patch grid. For example, a factor of 16 will upsample a
        depth map of shape (H/patch, W/patch) to (H, W). The default is 16
        assuming a ViT with 16×16 patches.
    hidden_dim: int
        Number of hidden units in the pose head MLP.
    """

    def __init__(
        self,
        backbone: Optional[VisionTransformerBackbone] = None,
        decode_depth: bool = True,
        decode_pose: bool = True,
        upsample_factor: int = 16,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = VisionTransformerBackbone("vit_b_16", pretrained=True)
        self.backbone = backbone
        self.decode_depth = decode_depth
        self.decode_pose = decode_pose
        self.upsample_factor = upsample_factor
        self.patch_size = backbone.patch_size

        # Depth decoder: simple convolutional layers to map patch embeddings
        # to depth values. We take the transformer patch embeddings (excluding
        # the CLS token) of shape (B, N_patches, C) and reshape them into
        # (B, C, H_p, W_p). Then we apply a few 3×3 conv layers and upsample.
        if decode_depth:
            c_in = backbone.embed_dim
            self.depth_decoder = nn.Sequential(
                nn.Conv2d(c_in, c_in // 2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in // 2, c_in // 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in // 4, 1, kernel_size=3, padding=1),
            )

        # Pose head: MLP mapping pooled features to rotation+translation
        if decode_pose:
            c_in = backbone.embed_dim
            self.pose_head = nn.Sequential(
                nn.Linear(c_in, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 9),  # 6 for rotation repr, 3 for translation
            )

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Predict depth maps and camera poses for input images.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            A tuple ``(depth, pose)`` where ``depth`` is a tensor of shape
            ``(B, 1, H, W)`` or ``None`` if depth decoding is disabled, and
            ``pose`` is a tuple ``(R, t)`` containing rotation matrices
            ``(B, 3, 3)`` and translation vectors ``(B, 3)`` or ``None`` if
            pose decoding is disabled.
        """
        B, C, H, W = x.shape
        # Extract patch embeddings and pooled features
        patch_tokens = self.backbone(x, return_pooled=False)  # (B, N+1, C)
        cls_tokens = patch_tokens[:, 0]  # (B, C)
        patch_tokens = patch_tokens[:, 1:]  # (B, N, C)
        # Determine patch grid size
        H_p = H // self.patch_size
        W_p = W // self.patch_size
        patch_tokens = patch_tokens.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2)  # (B, C, H_p, W_p)
        depth_map: Optional[torch.Tensor] = None
        pose: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if self.decode_depth:
            depth_coarse = self.depth_decoder(patch_tokens)  # (B, 1, H_p, W_p)
            # Upsample to input resolution using bilinear interpolation
            depth_map = F.interpolate(
                depth_coarse,
                scale_factor=self.upsample_factor // self.patch_size,
                mode="bilinear",
                align_corners=False,
            )
            # Ensure correct size
            if depth_map.shape[-2] != H or depth_map.shape[-1] != W:
                depth_map = F.interpolate(depth_map, size=(H, W), mode="bilinear", align_corners=False)

        if self.decode_pose:
            pose_vec = self.pose_head(cls_tokens)  # (B, 9)
            rot_6d = pose_vec[:, :6]  # first 6 numbers for rotation rep
            trans = pose_vec[:, 6:]  # last 3 numbers for translation
            # Convert 6D rotation rep to 3x3 matrix via Gram-Schmidt
            r1 = rot_6d[:, 0:3]
            r2 = rot_6d[:, 3:6]
            # Orthonormalise
            b1 = F.normalize(r1, dim=-1)
            b2 = F.normalize(r2 - (b1 * r2).sum(dim=-1, keepdim=True) * b1, dim=-1)
            b3 = torch.cross(b1, b2)
            R = torch.stack([b1, b2, b3], dim=-1)  # (B, 3, 3)
            # Optionally normalise via SVD to ensure orthogonality
            R = _normalize_rotation(R)
            pose = (R, trans)

        return depth_map, pose

    @staticmethod
    def backproject_points(
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Back-project depth maps into 3D point clouds.

        Args:
            depth: Depth map tensor of shape ``(B, 1, H, W)``.
            intrinsics: Camera intrinsic matrix of shape ``(B, 3, 3)`` or ``(3, 3)``.
            device: Optional device on which to perform computations. If
                ``None``, uses the device of ``depth``.

        Returns:
            Point cloud tensor of shape ``(B, H*W, 3)`` with 3D coordinates in
            camera frame.
        """
        B, _, H, W = depth.shape
        if device is None:
            device = depth.device
        # Create pixel grid
        ys, xs = torch.meshgrid(
            torch.linspace(0, H - 1, H, device=device),
            torch.linspace(0, W - 1, W, device=device),
            indexing="ij",
        )
        # Flatten
        xs = xs.reshape(-1)
        ys = ys.reshape(-1)
        # Convert intrinsics to shape (B, 3, 3)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0).expand(B, -1, -1)
        inv_K = torch.linalg.inv(intrinsics)  # (B, 3, 3)
        points = []
        for b in range(B):
            d = depth[b, 0].reshape(-1)  # (H*W)
            pts = torch.stack([xs * d, ys * d, d], dim=1)  # (H*W, 3)
            pts = (inv_K[b] @ pts.T).T  # (H*W, 3)
            points.append(pts)
        return torch.stack(points, dim=0)


def merge_point_clouds(
    points_list: List[np.ndarray],
    poses: List[Tuple[np.ndarray, np.ndarray]],
    world_frame: Optional[str] = "first_view",
) -> np.ndarray:
    """Merge multiple point clouds into a single coordinate frame.

    Given a list of point clouds (each of shape ``(N_i, 3)``) and their
    corresponding camera poses (rotation matrices and translation vectors),
    this function transforms all points into a common world coordinate frame
    and concatenates them.

    Args:
        points_list: List of point clouds. Each entry is a ``(N, 3)`` array
            of points in the camera coordinate system.
        poses: List of tuples ``(R, t)`` where ``R`` is a 3×3 rotation
            matrix and ``t`` is a 3-element translation vector representing
            the pose of the camera relative to the world frame. The length
            of ``poses`` must match ``points_list``.
        world_frame: If ``"first_view"`` (default), the coordinate frame
            of the first view is used as the world frame, and all other
            point clouds are transformed into this frame. If ``"world"``,
            assumes the provided poses already express the camera poses in
            world coordinates.

    Returns:
        An ``(M, 3)`` array of points in the world frame where
        ``M = sum_i N_i``.
    """
    assert len(points_list) == len(poses), "Number of point clouds and poses must match."
    all_points = []
    # Determine world reference
    if world_frame == "first_view":
        R0, t0 = poses[0]
        R0_inv = R0.T
        t0_inv = -R0_inv @ t0
    for i, (pts, (R, t)) in enumerate(zip(points_list, poses)):
        if world_frame == "first_view":
            # Compute relative transformation: first view -> current view
            R_rel = R @ R0_inv
            t_rel = t + (-R @ R0_inv @ t0)
        else:
            R_rel, t_rel = R, t
        # Transform points from camera to world
        pts_world = (R_rel @ pts.T).T + t_rel
        all_points.append(pts_world)
    return np.concatenate(all_points, axis=0)