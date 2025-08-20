"""Plant species recognition module.

This module defines the :class:`PlantRecognitionModel` class, which wraps a
Vision Transformer backbone for feature extraction and adds a classification
head to predict plant species. The model can be trained end-to-end or
fine-tuned by freezing the backbone and training only the head. The number
of classes must be specified at construction time.

Example usage::

    from biovit3r.models.backbone import VisionTransformerBackbone
    from biovit3r.models.recognition import PlantRecognitionModel
    import torch

    backbone = VisionTransformerBackbone("vit_b_16", pretrained=True)
    model = PlantRecognitionModel(num_classes=10, backbone=backbone)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    probs = logits.softmax(dim=-1)
    predicted = probs.argmax(dim=-1)

"""
from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "biovit3r.models.recognition requires PyTorch. Install PyTorch before use."
    ) from e

from .backbone import VisionTransformerBackbone


class PlantRecognitionModel(nn.Module):
    """Plant species classifier using a ViT backbone.

    Parameters
    ----------
    num_classes: int
        Number of plant species (classes) the model should predict.
    backbone: Optional[VisionTransformerBackbone]
        A ViT backbone instance. If ``None``, a default ``vit_b_16`` backbone
        is created. Passing a backbone allows reusing a shared backbone
        across different tasks.
    dropout: float
        Dropout probability applied to the pooled features before the
        classification head. Set to 0.0 to disable dropout.

    Attributes
    ----------
    backbone: VisionTransformerBackbone
        The underlying ViT used for feature extraction.
    classifier: nn.Linear
        Linear layer mapping the pooled features to class logits.
    """

    def __init__(
        self,
        num_classes: int,
        backbone: Optional[VisionTransformerBackbone] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if backbone is None:
            backbone = VisionTransformerBackbone("vit_b_16", pretrained=True)
        self.backbone = backbone
        embed_dim = backbone.embed_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classifier.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        # Extract pooled [CLS] token features
        pooled = self.backbone(x, return_pooled=True)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze the backbone parameters."""
        self.backbone.freeze()

    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone parameters."""
        self.backbone.unfreeze()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted class indices for input batch ``x``.

        This convenience method applies the softmax to the logits and returns
        the index of the maximum probability for each sample.
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        return probs.argmax(dim=-1)