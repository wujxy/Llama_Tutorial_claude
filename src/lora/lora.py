"""
LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning.

LoRA freezes the pre-trained model weights and injects trainable rank decomposition
matrices into each layer. This greatly reduces the number of trainable parameters
while maintaining model performance.

The key idea is:
    output = W @ x + (scale * B @ A @ x)

Where:
    - W is the frozen weight matrix
    - A and B are low-rank matrices (rank r << d)
    - scale = alpha / r

Usage:
    from src.lora.lora import LoRALinear, apply_lora

    # Replace a linear layer with LoRA
    lora_layer = LoRALinear(original_layer, r=8, alpha=16)

    # Apply LoRA to a model
    apply_lora(model, r=8, alpha=16, target_modules=['wq', 'wk', 'wv', 'wo'])
"""

from typing import List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA-enhanced linear layer.

    This wraps a frozen linear layer and adds low-rank adaptation:
        output = W @ x + scale * B @ A @ x

    Where:
        - W: Frozen weight matrix
        - A: Down-projection (d_in -> r), initialized with Kaiming
        - B: Up-projection (r -> d_out), initialized to zeros
        - scale: Scaling factor (alpha / r)

    Args:
        original_layer: The frozen nn.Linear layer
        r: LoRA rank (dimension of the low-rank matrices)
        alpha: LoRA scaling parameter
        dropout: Dropout rate for LoRA path
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Store original frozen layer
        self.original_layer = original_layer

        # LoRA parameters
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Get dimensions from original layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        # LoRA matrices
        # A: Down-projection (in_features -> r), Kaiming initialization
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # B: Up-projection (r -> out_features), initialized to zeros
        # This ensures the initial perturbation is zero
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Freeze original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Original frozen path: W @ x
        original_output = self.original_layer(x)

        # LoRA path: scale * B @ A @ x
        lora_output = self.dropout(x)

        # Ensure LoRA parameters are on the same device as input
        lora_A = self.lora_A.to(x.device)
        lora_B = self.lora_B.to(x.device)

        lora_output = lora_output @ lora_A.T  # (..., r)
        lora_output = lora_output @ lora_B.T  # (..., out_features)
        lora_output = lora_output * self.scaling

        return original_output + lora_output

    def merge_weights(self) -> None:
        """
        Merge LoRA weights into the original layer.

        This updates W <- W + scale * B @ A and removes the LoRA matrices.
        Useful for deployment when you want a single weight matrix.
        """
        # Compute merged weight
        delta_w = self.lora_B @ self.lora_A * self.scaling
        merged_weight = self.original_layer.weight.data + delta_w

        # Update original layer weight
        self.original_layer.weight.data = merged_weight

        # Remove LoRA parameters (set to None to indicate merged)
        self.lora_A = None
        self.lora_B = None

    def get_lora_parameters(self) -> List[nn.Parameter]:
        """
        Get trainable LoRA parameters.

        Returns:
            List of LoRA parameters (lora_A and lora_B)
        """
        if self.lora_A is None or self.lora_B is None:
            return []
        return [self.lora_A, self.lora_B]


import math


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply LoRA to specific linear layers in a model.

    This function recursively searches through the model and replaces
    matching linear layers with LoRA-enhanced versions.

    Args:
        model: The model to apply LoRA to
        r: LoRA rank
        alpha: LoRA scaling parameter
        dropout: Dropout rate
        target_modules: List of module names to apply LoRA to
                       (e.g., ['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'])
                       If None, applies to all Linear layers

    Returns:
        The modified model with LoRA applied

    Example:
        # Apply LoRA to attention and MLP layers
        model = apply_lora(
            model,
            r=8,
            alpha=16,
            target_modules=['wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']
        )
    """
    if target_modules is None:
        # Default: apply to all Linear layers
        target_modules = []

    # Track which modules were replaced
    replaced_count = 0

    # Recursively apply LoRA to matching modules
    for name, module in list(model.named_children()):
        # Check if this module matches a target
        is_target = (
            len(target_modules) == 0 or  # Apply to all if no targets specified
            any(name.endswith(t) or t in name for t in target_modules)
        )

        if isinstance(module, nn.Linear) and is_target:
            # Replace with LoRA version
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(model, name, lora_layer)
            replaced_count += 1
        else:
            # Recursively process child modules
            replaced = apply_lora(
                module,
                r=r,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            )
            replaced_count += replaced

    return replaced_count


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all trainable LoRA parameters from a model.

    This is useful for creating an optimizer that only updates LoRA weights:
        optimizer = torch.optim.AdamW(get_lora_parameters(model), lr=1e-3)

    Args:
        model: The model to extract LoRA parameters from

    Returns:
        List of LoRA parameters
    """
    lora_params = []

    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend(module.get_lora_parameters())

    return lora_params


def freeze_base_model(model: nn.Module) -> None:
    """
    Freeze all non-LoRA parameters in the model.

    This sets requires_grad=False for all parameters except LoRA ones.

    Args:
        model: The model to freeze
    """
    for name, param in model.named_parameters():
        # Only LoRA parameters should be trainable
        # (lora_A and lora_B are in LoRALinear modules)
        is_lora_param = 'lora_A' in name or 'lora_B' in name
        param.requires_grad = is_lora_param


def save_lora_weights(model: nn.Module, path: str) -> None:
    """
    Save only the LoRA weights from a model.

    Args:
        model: The model with LoRA
        path: Path to save the weights
    """
    lora_state_dict = {}

    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            lora_state_dict[name] = param.data.clone()

    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str) -> None:
    """
    Load LoRA weights into a model.

    Args:
        model: The model to load weights into
        path: Path to the saved LoRA weights
    """
    lora_state_dict = torch.load(path, map_location='cpu', weights_only=False)

    model_state_dict = model.state_dict()
    for name, param in lora_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
        else:
            print(f"Warning: {name} not found in model")
