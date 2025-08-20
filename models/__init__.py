"""Model components for BioViT3R.

This subpackage bundles together the various neural network modules used in
the BioViT3R pipeline. The modules are organised by subsystem:

* ``backbone`` provides wrappers around Vision Transformer architectures and
  associated utilities for feature extraction.
* ``recognition`` defines classifiers for plant species recognition using
  transformer features.
* ``reconstruction`` contains models for predicting depth maps and camera
  poses from images and fusing them into 3D point clouds.
* ``reorganization`` includes functions and small modules used to convert
  point clouds into biomass estimates via allometric relationships.

Most modules are implemented with PyTorch. To avoid import errors when
PyTorch is unavailable, each module checks for PyTorch at import time and
provides informative error messages. When writing new models, consider
adding similar guards to ease debugging on systems without the requisite
libraries installed.
"""

from .backbone import VisionTransformerBackbone
from .recognition import PlantRecognitionModel
from .reconstruction import DepthPoseEstimator, merge_point_clouds
from .reorganization import BiomassEstimator

__all__ = [
    "VisionTransformerBackbone",
    "PlantRecognitionModel",
    "DepthPoseEstimator",
    "BiomassEstimator",
    "merge_point_clouds",
]