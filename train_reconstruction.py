"""Training script for the DepthPoseEstimator.

This script allows training the reconstruction subsystem of BioViT3R, which
predicts depth maps and camera poses from RGB images. It supports
supervised learning using ground-truth depth and pose annotations. The
training routine computes L1 loss on depth maps and rotational and
translational errors on poses. It can be extended to include self-
supervised losses (e.g. photometric consistency) if ground-truth depth
is not available.

Example usage:

    python -m biovit3r.train_reconstruction \
        --train-dir /path/to/train_data \
        --val-dir /path/to/val_data \
        --epochs 20 \
        --batch-size 2 \
        --lr 1e-4 \
        --output-model /path/to/save_recon.pth

The dataset used must contain depth maps and pose files for supervision.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import math
from typing import Tuple

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import transforms
except ImportError as e:
    raise ImportError(
        "PyTorch is required to run the training script. "
        "Please install torch and torchvision."
    ) from e

import numpy as np
from biovit3r.datasets import MultiViewPlantDataset
from biovit3r.models.reconstruction import DepthPoseEstimator
from biovit3r.utils.geometry import chamfer_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BioViT3R reconstruction model")
    parser.add_argument("--train-dir", type=str, required=True, help="Path to training data root")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to validation data root")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-model", type=str, default="reconstruction_model.pth", help="Output model file")
    parser.add_argument("--lambda-depth", type=float, default=1.0, help="Weight for depth loss")
    parser.add_argument("--lambda-pose", type=float, default=0.1, help="Weight for pose loss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Data transforms: ensure images are consistent size
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Load dataset with preload depth and pose
    train_ds = MultiViewPlantDataset(args.train_dir, num_views=1, transform=transform)
    val_ds = MultiViewPlantDataset(args.val_dir, num_views=1, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthPoseEstimator()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Loss functions
    depth_loss_fn = nn.L1Loss()

    def pose_loss_fn(pred_pose: Tuple[torch.Tensor, torch.Tensor], gt_pose: Tuple[np.ndarray, np.ndarray]) -> torch.Tensor:
        # pred_pose: (R_pred (B,3,3), t_pred (B,3))
        R_pred, t_pred = pred_pose
        R_gt, t_gt = gt_pose
        # convert gt to tensors on device
        R_gt = torch.tensor(R_gt, dtype=torch.float32, device=device)
        t_gt = torch.tensor(t_gt, dtype=torch.float32, device=device)
        # rotation error: geodesic distance on SO(3)
        Rt = R_gt @ R_pred.transpose(1, 2)
        trace = torch.clamp((Rt[:, 0, 0] + Rt[:, 1, 1] + Rt[:, 2, 2] - 1) / 2.0, -1.0, 1.0)
        ang = torch.arccos(trace)  # (B)
        rot_loss = ang.mean()
        trans_loss = (t_pred - t_gt).abs().mean()
        return rot_loss + trans_loss

    best_cd = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            # Each batch item: (rgb_views, depth_views, pose_views, label)
            rgb_views, depth_views, pose_views, labels = batch
            # Use only one view per sample
            rgb = torch.stack([img[0] for img in rgb_views]).to(device)
            has_depth = depth_views[0] is not None
            has_pose = pose_views[0] is not None
            optimizer.zero_grad()
            pred_depth, pred_pose = model(rgb)
            loss = 0.0
            if has_depth and pred_depth is not None:
                # gather gt depth
                gt_depth = torch.tensor(
                    np.stack([d[0] for d in depth_views]),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(1)  # (B,1,H,W)
                loss_d = depth_loss_fn(pred_depth, gt_depth)
                loss = loss + args.lambda_depth * loss_d
            if has_pose and pred_pose is not None:
                gt_R = np.stack([p[0] for p in pose_views])  # (B,3,3)
                gt_t = np.stack([p[1] for p in pose_views])  # (B,3)
                loss_p = pose_loss_fn(pred_pose, (gt_R, gt_t))
                loss = loss + args.lambda_pose * loss_p
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        # Evaluate on validation: compute Chamfer distance between predicted and gt point clouds
        model.eval()
        cd_scores = []
        with torch.no_grad():
            for rgb_views, depth_views, pose_views, labels in val_loader:
                rgb = rgb_views[0].to(device)
                pred_depth, pred_pose = model(rgb.unsqueeze(0))
                # Only evaluate if ground truth depth and pose exist
                if depth_views[0] is not None and pose_views[0] is not None and pred_depth is not None and pred_pose is not None:
                    # Back-project predicted depth to point cloud in camera frame
                    # Use simple pinhole intrinsics with focal length = 1, principal point at centre
                    B, _, H, W = pred_depth.shape
                    fx = fy = 1.0
                    cx = (W - 1) / 2.0
                    cy = (H - 1) / 2.0
                    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    points_pred = DepthPoseEstimator.backproject_points(pred_depth.cpu(), torch.tensor(intr, dtype=torch.float32)).numpy()[0]
                    # Transform to world frame using predicted pose
                    R_pred, t_pred = pred_pose
                    R_pred = R_pred[0].cpu().numpy()
                    t_pred = t_pred[0].cpu().numpy()
                    points_pred_world = (R_pred @ points_pred.T).T + t_pred
                    # Process ground truth
                    gt_depth = depth_views[0][0]
                    # Back-project using same intrinsics
                    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
                    pts_gt = np.stack([(xs * gt_depth).ravel(), (ys * gt_depth).ravel(), gt_depth.ravel()], axis=1)
                    invK = np.linalg.inv(intr)
                    pts_gt_cam = (invK @ pts_gt.T).T
                    R_gt, t_gt = pose_views[0]
                    points_gt_world = (R_gt @ pts_gt_cam.T).T + t_gt
                    # Subsample for efficiency
                    idx_pred = np.random.choice(points_pred_world.shape[0], size=min(2000, points_pred_world.shape[0]), replace=False)
                    idx_gt = np.random.choice(points_gt_world.shape[0], size=min(2000, points_gt_world.shape[0]), replace=False)
                    cd = chamfer_distance(points_pred_world[idx_pred], points_gt_world[idx_gt])
                    cd_scores.append(cd)
        avg_cd = float(np.mean(cd_scores)) if cd_scores else float('inf')
        print(f"Epoch {epoch}/{args.epochs} | Train loss {avg_loss:.4f} | Val Chamfer {avg_cd:.4f}")
        # Save best model
        if avg_cd < best_cd:
            best_cd = avg_cd
            torch.save(model.state_dict(), args.output_model)
    print(f"Training complete. Best validation Chamfer distance: {best_cd:.4f}")


if __name__ == "__main__":
    main()