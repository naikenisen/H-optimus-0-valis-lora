import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model


def _infer_lora_targets(model: nn.Module) -> list[str]:
    """Find attention projection module suffixes compatible with PEFT LoRA."""
    linear_suffixes = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    preferred = ["q_proj", "v_proj", "query", "value", "qkv"]
    targets = [name for name in preferred if name in linear_suffixes]
    if targets:
        return targets

    # Fallback for timm-style ViT blocks where qkv/proj are common names.
    fallback = [name for name in ("qkv", "proj") if name in linear_suffixes]
    if fallback:
        return fallback

    raise ValueError(
        "Could not infer LoRA target modules from encoder. "
        f"Detected linear suffixes: {sorted(linear_suffixes)}"
    )


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

class UpsampleBlock(nn.Module):

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

    def __init__(self, embed_dim: int = 1536, target_size: int = 224):
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

    def __init__(
        self,
        model_name: str = "hf-hub:bioptimus/H-optimus-0",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_size: int = 224,
    ):
        super().__init__()

        # Load encoder via timm — the official method for H-optimus-0.
        # dynamic_img_size=False keeps pos_embed at native 224 (16x16 grid).
        encoder = timm.create_model(
            model_name,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )

        self.embed_dim = encoder.num_features  # 1536 for H-optimus-0
        self.patch_size = getattr(encoder.patch_embed, "patch_size", (14, 14))
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]
        self.encoder_input_size = 224
        self.target_size = target_size

        print(f"[model] Encoder loaded: embed_dim={self.embed_dim}, "
              f"patch_size={self.patch_size}, "
              f"pos_embed shape={encoder.pos_embed.shape}")

        # Freeze all encoder weights
        for param in encoder.parameters():
            param.requires_grad = False

        # Apply LoRA on attention projections
        target_modules = _infer_lora_targets(encoder)
        print(f"[model] LoRA target modules: {target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.encoder = get_peft_model(encoder, lora_config)
        self.encoder.print_trainable_parameters()

        # Decoder
        self.decoder = ConvDecoder(embed_dim=self.embed_dim, target_size=target_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # timm ViT forward_features returns (B, num_tokens, D) including
        # CLS / register prefix tokens followed by spatial patch tokens.
        features = self.encoder.model.forward_features(pixel_values)

        # Strip CLS / register prefix tokens to keep only spatial patches.
        grid_size = self.encoder_input_size // self.patch_size  # 224/14 = 16
        num_patches = grid_size * grid_size  # 256
        patch_tokens = features[:, -num_patches:, :]  # (B, 256, 1536)

        # Reshape to spatial feature map
        B = patch_tokens.shape[0]
        spatial = (
            patch_tokens
            .transpose(1, 2)
            .reshape(B, self.embed_dim, grid_size, grid_size)
        )

        return self.decoder(spatial)

    def get_trainable_state_dict(self) -> dict:
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                state[name] = param.data
        # Include decoder BatchNorm running stats
        for name, buf in self.decoder.named_buffers():
            state[f"decoder.{name}"] = buf
        return state

    def load_trainable_state_dict(self, state_dict: dict) -> None:
        self.load_state_dict(state_dict, strict=False)
