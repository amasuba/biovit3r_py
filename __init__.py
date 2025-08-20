"""Top-level package for BioViT3R.

This package provides modules to build and train the BioViT3R system for
automated plant biomass estimation. The system is organised into three
primary subsystems: recognition, reconstruction and reorganization. Each
submodule encapsulates a specific stage of the pipeline.

.. code-block:: python

   from biovit3r.models.recognition import PlantRecognitionModel
   from biovit3r.models.reconstruction import DepthPoseEstimator
   from biovit3r.models.reorganization import BiomassEstimator

   # Instantiate subsystems
   recogniser = PlantRecognitionModel(num_classes=5)
   reconstructor = DepthPoseEstimator()
   reorganiser = BiomassEstimator()

The intent of this package is to provide a coherent code structure that
researchers can extend. Many of the models defined here depend on
PyTorch and optionally `timm` for pre-trained Vision Transformer backbones.
If those dependencies are not available the modules will raise an
``ImportError`` at import time. See the README for further guidance on
setting up the environment.
"""

__all__ = [
    "datasets",
    "models",
    "utils",
]