"""H-optimus-0 encoder with LoRA adaptation and convolutional decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):
    """Bilinear ×2 upsample followed by two 3×3 convolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))


class ConvDecoder(nn.Module):
    """Progressive upsampling decoder: patch embeddings → RGB image.

    Applies a 1×1 projection then 4 × bilinear-upsample blocks
    (each doubling spatial resolution), finishing with a bilinear
    interpolation to *target_size* and a 1×1 head with sigmoid output.

    Channel progression: embed_dim → 512 → 256 → 128 → 64 → 32 → 3
    """

    def __init__(self, embed_dim: int = 1024, target_size: int = 224):
        super().__init__()
        self.target_size = target_size

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        self.blocks = nn.ModuleList(
            [
                UpsampleBlock(512, 256),
                UpsampleBlock(256, 128),
                UpsampleBlock(128, 64),
                UpsampleBlock(64, 32),
            ]
        )

        self.head = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim, H_grid, W_grid) spatial patch embeddings.
        Returns:
            (B, 3, target_size, target_size) predicted image in [0, 1].
        """
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = F.interpolate(
            x,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        return self.head(x)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class HOptimusLoRA(nn.Module):
    """H-optimus-0 ViT encoder + LoRA adaptation + convolutional decoder.

    1. Loads the pre-trained bioptimus/H-optimus-0 backbone.
    2. Freezes all encoder weights.
    3. Injects LoRA adapters on attention query/value projections.
    4. Attaches a lightweight ``ConvDecoder`` to reconstruct an RGB image
       from the encoder's patch embeddings.

    Only LoRA adapters and decoder parameters are trainable.

    Args:
        model_name: HuggingFace model identifier.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout applied inside LoRA layers.
        target_size: Output image spatial resolution.
    """

    def __init__(
        self,
        model_name: str = "bioptimus/H-optimus-0",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_size: int = 224,
    ):
        super().__init__()

        # Load pre-trained encoder (requires HF token & model access)
        encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Store architecture constants before PEFT wrapping
        self.embed_dim = getattr(encoder.config, "hidden_size", 1024)
        self.patch_size = getattr(encoder.config, "patch_size", 14)
        self.target_size = target_size

        # Freeze all encoder weights
        for param in encoder.parameters():
            param.requires_grad = False

        # Apply LoRA on attention projections
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            bias="none",
        )
        self.encoder = get_peft_model(encoder, lora_config)
        self.encoder.print_trainable_parameters()

        # Decoder
        self.decoder = ConvDecoder(embed_dim=self.embed_dim, target_size=target_size)

    # ------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) normalised input images.
        Returns:
            (B, 3, target_size, target_size) predicted image in [0, 1].
        """
        # Encode
        outputs = self.encoder(pixel_values=pixel_values)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, D)

        # Number of spatial patch tokens (handles CLS + optional register tokens)
        grid_size = pixel_values.shape[-1] // self.patch_size
        num_patches = grid_size * grid_size
        patch_tokens = hidden_states[:, -num_patches:, :]  # (B, N, D)

        # Reshape to spatial feature map
        B = patch_tokens.shape[0]
        spatial = (
            patch_tokens
            .transpose(1, 2)
            .reshape(B, self.embed_dim, grid_size, grid_size)
        )

        return self.decoder(spatial)

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def get_trainable_state_dict(self) -> dict:
        """Return state dict with only trainable params + decoder buffers."""
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.data
        # Include decoder BatchNorm running stats
        for name, buf in self.decoder.named_buffers():
            state[f"decoder.{name}"] = buf
        return state

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        """Load a checkpoint produced by ``get_trainable_state_dict``."""
        self.load_state_dict(state_dict, strict=False)
