# BioViT3R

BioViT3R is a modular research codebase for self‐supervised 3D plant
reconstruction and biomass estimation. It implements the three key
subsystems described in the BioViT3R pipeline: **recognition** (plant
species classification), **reconstruction** (multi‐view depth and pose
estimation) and **reorganization** (biomass calculation from 3D geometry).

The code is intentionally lightweight and serves as a starting point for
developing and experimenting with transformer‐based approaches to plant
phenotyping. Many design choices, such as the specific network
architectures and loss functions, are simplified relative to state‐of‐the
art models. Users are encouraged to extend and replace components to meet
their research needs.

## Installation

BioViT3R relies on Python 3.10+ and several third–party packages. At
minimum you will need:

* [PyTorch](https://pytorch.org) (1.13 or newer) and
  [torchvision](https://pytorch.org/vision) for deep learning modules.
* [scipy](https://scipy.org) for convex hull volume computation.
* [scikit‐learn](https://scikit-learn.org) for evaluation metrics.

You can install these dependencies with pip:

```bash
pip install torch torchvision scipy scikit-learn
```

Optionally, install [timm](https://github.com/huggingface/pytorch-image-models)
to use a wider range of Vision Transformer backbones including DINO
pretrained models:

```bash
pip install timm
```

## Code Structure

* ``biovit3r/datasets.py`` – dataset definitions for loading multi–view
  plant images, optional depth maps and camera poses.
* ``biovit3r/models/backbone.py`` – wrappers around Vision Transformer
  architectures from torchvision or timm.
* ``biovit3r/models/recognition.py`` – plant species classifier using a
  ViT backbone and a linear head.
* ``biovit3r/models/reconstruction.py`` – a simple depth and pose
  estimator built on top of a ViT backbone, plus a utility for fusing
  point clouds across views.
* ``biovit3r/models/reorganization.py`` – biomass estimator computing
  allometric volume from point clouds and applying species–specific
  coefficients.
* ``biovit3r/utils`` – helper functions for geometry (Chamfer distance)
  and evaluation metrics (MAE, MAPE, R²).
* ``biovit3r/train_recognition.py`` – example training script for the
  recognition subsystem.
* ``biovit3r/train_reconstruction.py`` – example training script for the
  reconstruction subsystem.

## Getting Started

1. **Prepare your dataset.** Organise your multi–view plant dataset
   following the directory structure described in
   ``biovit3r/datasets.py``. For each sample create a folder containing
   a ``meta.json`` file, an ``rgb/`` directory with one or more RGB
   images, and optionally ``depth/`` and ``pose/`` subdirectories with
   NumPy depth maps and camera pose matrices.

2. **Train the recognition model.** Use the provided training script
   to fine–tune a ViT classifier on your species labels:

   ```bash
   python -m biovit3r.train_recognition \
       --train-dir path/to/train \
       --val-dir path/to/val \
       --num-classes NUM_CLASSES \
       --epochs 10 \
       --batch-size 4 \
       --lr 1e-4 \
       --output-model model_recognition.pth
   ```

3. **Train the reconstruction model.** With depth and pose annotations
   available, train the depth and pose estimator:

   ```bash
   python -m biovit3r.train_reconstruction \
       --train-dir path/to/train \
       --val-dir path/to/val \
       --epochs 20 \
       --batch-size 2 \
       --lr 1e-4 \
       --output-model model_reconstruction.pth
   ```

4. **Run inference.** Load the trained models and process new image
   sequences through the pipeline. An example skeleton is provided below:

   ```python
   import torch
   from biovit3r.models import PlantRecognitionModel, DepthPoseEstimator, BiomassEstimator
   from biovit3r.models.reconstruction import merge_point_clouds
   from biovit3r.datasets import MultiViewPlantDataset

   # Load models
   rec_model = PlantRecognitionModel(num_classes=NUM_CLASSES)
   rec_model.load_state_dict(torch.load("model_recognition.pth"))
   rec_model.eval()

   recon_model = DepthPoseEstimator()
   recon_model.load_state_dict(torch.load("model_reconstruction.pth"))
   recon_model.eval()

   biomass_estimator = BiomassEstimator(allometric_params={...})

   # Process a sample
   dataset = MultiViewPlantDataset("/path/to/sample", num_views=6)
   rgb_views, depth_views, pose_views, _ = dataset[0]
   # Convert images to tensor and normalise
   transform = ...  # same as in training
   imgs = torch.stack([transform(img) for img in rgb_views])
   # Recognise species
   species = rec_model.predict(imgs)
   # Reconstruct depth and pose for each view
   pred_depths = []
   pred_poses = []
   for img in imgs:
       d, (R, t) = recon_model(img.unsqueeze(0))
       pred_depths.append(d[0].cpu())
       pred_poses.append((R[0].cpu().numpy(), t[0].cpu().numpy()))
   # Convert predicted depths to point clouds
   # ... create intrinsics
   intr = torch.eye(3)
   points_list = []
   for d in pred_depths:
       pts = DepthPoseEstimator.backproject_points(d.unsqueeze(0), intr).squeeze(0).cpu().numpy()
       points_list.append(pts)
   # Merge point clouds into world frame
   merged = merge_point_clouds(points_list, pred_poses)
   # Estimate biomass
   biomass = biomass_estimator.estimate_biomass(merged, int(species))
   print("Biomass estimate:", biomass)
   ```

The example above illustrates how to run each subsystem sequentially.

## Contributing

Contributions are welcome! If you find bugs or have ideas for improving the
architecture (e.g. integrating self–supervised losses, supporting video
input or implementing more sophisticated reconstruction backbones), feel
free to open an issue or submit a pull request.

## License

This project is distributed under the MIT License. See the
``LICENSE`` file for details.