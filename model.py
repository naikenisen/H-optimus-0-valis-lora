import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel


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


def _resize_timm_pos_embed(timm_model: nn.Module, target_size: int, patch_size: int) -> None:
    """Resize timm positional embeddings to match a target image grid."""
    pos_embed = getattr(timm_model, "pos_embed", None)
    if pos_embed is None or pos_embed.ndim != 3:
        return

    target_grid = target_size // patch_size
    target_tokens = target_grid * target_grid
    cur_tokens = pos_embed.shape[1]
    embed_dim = pos_embed.shape[-1]

    # Handle both formats: [N] (no prefix token) and [1 + N] (CLS prefix).
    prefix_tokens = 0
    spatial_tokens = cur_tokens
    old_grid = int(spatial_tokens ** 0.5)
    if old_grid * old_grid != spatial_tokens and cur_tokens > 1:
        candidate = cur_tokens - 1
        old_grid = int(candidate ** 0.5)
        if old_grid * old_grid == candidate:
            prefix_tokens = 1
            spatial_tokens = candidate

    if old_grid * old_grid != spatial_tokens:
        return

    if spatial_tokens == target_tokens:
        return

    prefix = pos_embed[:, :prefix_tokens, :] if prefix_tokens else None
    spatial = pos_embed[:, prefix_tokens:, :]
    pos_2d = spatial.reshape(1, old_grid, old_grid, embed_dim).permute(0, 3, 1, 2)
    resized_2d = F.interpolate(
        pos_2d,
        size=(target_grid, target_grid),
        mode="bicubic",
        align_corners=False,
    )
    resized = resized_2d.permute(0, 2, 3, 1).reshape(1, target_tokens, embed_dim)
    if prefix is not None:
        resized = torch.cat([prefix, resized], dim=1)

    timm_model.pos_embed = nn.Parameter(resized)

    patch_embed = getattr(timm_model, "patch_embed", None)
    if patch_embed is not None:
        if hasattr(patch_embed, "img_size"):
            patch_embed.img_size = (target_size, target_size)
        if hasattr(patch_embed, "grid_size"):
            patch_embed.grid_size = (target_grid, target_grid)
        if hasattr(patch_embed, "num_patches"):
            patch_embed.num_patches = target_tokens


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
        model_name: str = "bioptimus/H-optimus-0",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_size: int = 224,
    ):
        super().__init__()

        # Force 224x224 encoder grid so pretrained pos_embed (16x16=256 tokens)
        # matches the checkpoint exactly during from_pretrained loading.
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        for size_attr in ("image_size", "img_size"):
            if hasattr(config, size_attr):
                setattr(config, size_attr, 224)
        # TimmWrapperModel forwards timm_model_kwargs to timm.create_model().
        # The dict may not exist on the config — create it if needed.
        if not hasattr(config, "timm_model_kwargs") or not isinstance(
            getattr(config, "timm_model_kwargs", None), dict
        ):
            config.timm_model_kwargs = {}
        config.timm_model_kwargs["img_size"] = 224

        encoder = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True,
        )

        self.encoder_input_size = 224
        timm_model = getattr(encoder, "timm_model", None)
        # Safety net: resize pos_embed if it still doesn't match 224 grid.
        if timm_model is not None:
            _resize_timm_pos_embed(timm_model, target_size=224, patch_size=getattr(config, "patch_size", 14))

        # Store architecture constants before PEFT wrapping
        timm_model = getattr(encoder, "timm_model", None)
        self.embed_dim = (
            getattr(encoder.config, "hidden_size", None)
            or getattr(encoder.config, "embed_dim", None)
            or getattr(timm_model, "num_features", None)
            or getattr(timm_model, "embed_dim", None)
            or 1024
        )
        self.patch_size = getattr(encoder.config, "patch_size", 14)
        self.target_size = target_size

        # Freeze all encoder weights
        for param in encoder.parameters():
            param.requires_grad = False

        # Apply LoRA on attention projections
        target_modules = _infer_lora_targets(encoder)
        print(f"LoRA target modules: {target_modules}")
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
        if self.encoder_input_size is not None and pixel_values.shape[-1] != self.encoder_input_size:
            pixel_values = F.interpolate(
                pixel_values,
                size=(self.encoder_input_size, self.encoder_input_size),
                mode="bilinear",
                align_corners=False,
            )

        # Encode
        outputs = self.encoder(pixel_values=pixel_values)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, D)

        # Number of spatial patch tokens (handles CLS + optional register tokens)
        num_patches = hidden_states.shape[1]
        grid_size = int(num_patches ** 0.5)
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
