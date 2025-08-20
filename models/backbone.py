"""Vision Transformer backbone wrappers.

This module defines wrapper classes around Vision Transformer (ViT) models.
The primary class is :class:`VisionTransformerBackbone`, which instantiates
a ViT model from either ``torchvision`` or ``timm`` and exposes a unified
interface for feature extraction. The wrapper additionally allows for
freezing and unfreezing of backbone layers and accommodates different
patch sizes.

The code relies on PyTorch for both model definitions and tensor
operations. If PyTorch is not installed, importing this module will raise
an :class:`ImportError`. To enable ViT support, install PyTorch and
optionally the ``timm`` library if you wish to use non-``torchvision``
pretrained models (e.g. DINO weights).

Example:

    >>> from biovit3r.models.backbone import VisionTransformerBackbone
    >>> backbone = VisionTransformerBackbone(model_name="vit_b_16", pretrained=True)
    >>> x = torch.randn(2, 3, 224, 224)
    >>> features = backbone(x)
    >>> features.shape  # (B, num_patches, hidden_dim)

By default the backbone returns the patch embeddings and the [CLS] token.
Call ``backbone(x, return_pooled=True)`` to get just the pooled [CLS]
representation.
"""
from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "biovit3r.models.backbone requires PyTorch and torchvision. "
        "Please install them before using the VisionTransformerBackbone."
    ) from e


class VisionTransformerBackbone(nn.Module):
    """A wrapper around various Vision Transformer models.

    Parameters
    ----------
    model_name: str
        Name of the ViT architecture to load. Supported values include
        ``"vit_b_16"``, ``"vit_b_32"``, ``"vit_l_16"``, ``"vit_l_32"``. These map
        to the corresponding models in ``torchvision.models``. You may
        optionally specify a model from the ``timm`` library by prefixing the
        name with ``timm:`` (e.g. ``"timm:vit_base_patch16_224"``). When
        using ``timm``, ensure the library is installed.
    pretrained: bool
        Whether to load pretrained weights. For ``torchvision`` models this
        corresponds to ImageNet-22k or ImageNet1k pretrained weights. When
        specifying a ``timm`` model the default pretrained weights will be
        loaded if available.
    return_cls_token: bool
        If True, return the pooled [CLS] token in addition to the patch
        embeddings. Useful for classification heads.

    Notes
    -----
    This class does not implement DINO-specific feature extraction by
    itself. To use DINO-pretrained weights you must load a ``timm`` model
    that has been trained with DINO, or manually load weights into the
    transformer after construction. See the project documentation for
    instructions on integrating DINO models.
    """

    def __init__(
        self,
        model_name: str = "vit_b_16",
        pretrained: bool = True,
        return_cls_token: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.return_cls_token = return_cls_token

        # Determine whether to use torchvision or timm
        if model_name.startswith("timm:"):
            # Lazy import timm if available
            try:
                import timm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "timm library is required for timm models. "
                    "Install it via 'pip install timm'."
                ) from e
            timm_name = model_name.split(":", 1)[1]
            self.vit = timm.create_model(timm_name, pretrained=pretrained)
        else:
            if model_name == "vit_b_16":
                self.vit = vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
            elif model_name == "vit_b_32":
                self.vit = vit_b_32(weights="IMAGENET1K_V1" if pretrained else None)
            elif model_name == "vit_l_16":
                self.vit = vit_l_16(weights="IMAGENET1K_V1" if pretrained else None)
            elif model_name == "vit_l_32":
                self.vit = vit_l_32(weights="IMAGENET1K_V1" if pretrained else None)
            else:
                raise ValueError(f"Unsupported ViT model name: {model_name}")

        # Expose useful attributes
        self.embed_dim: int = getattr(self.vit, "hidden_dim", 768)
        # The patch size can be inferred from the convolutional projection
        self.patch_size: int = getattr(self.vit, "patch_size", 16)

    def forward(self, x: torch.Tensor, return_pooled: bool = False) -> torch.Tensor:
        """Forward pass through the ViT.

        If ``return_pooled`` is True and ``self.return_cls_token`` is True,
        returns the pooled [CLS] token of shape ``(batch_size, embed_dim)``. If
        ``return_pooled`` is False, returns the patch embeddings including
        the [CLS] token of shape ``(batch_size, num_patches+1, embed_dim)``.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            return_pooled: Whether to return the pooled embedding.

        Returns:
            Tensor representing patch embeddings or pooled representation.
        """
        # ensure input is float32
        x = x.to(dtype=torch.float32)
        out = self.vit._process_input(x)  # embed patches and add cls token
        # run through transformer encoder layers
        # The torchvision ViT exposes the encoder as a module
        out = self.vit.encoder(out)
        if return_pooled or self.return_cls_token:
            # The first token corresponds to [CLS]
            pooled = out[:, 0]
            return pooled
        return out

    def freeze(self) -> None:
        """Freeze all the backbone parameters (no gradient updates)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all the backbone parameters (enable gradient updates)."""
        for param in self.parameters():
            param.requires_grad = True