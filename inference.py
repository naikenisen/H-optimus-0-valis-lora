"""Inference script — predict CD30 from HES patches using a trained checkpoint."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor

from model import HOptimusLoRA


def preprocess(image_bgr: np.ndarray, image_size: int, mean, std) -> torch.Tensor:
    """Resize, normalise and convert a single BGR image to a model-ready tensor."""
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)


def postprocess(tensor: torch.Tensor, original_size: tuple) -> np.ndarray:
    """Convert model output tensor back to a BGR uint8 image at *original_size*."""
    img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()  # (H, W, 3) in [0, 1]
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    h, w = original_size
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def parse_args():
    p = argparse.ArgumentParser(description="HES → CD30 inference")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument(
        "--input",
        required=True,
        help="Path to a single HES image or a directory of images",
    )
    p.add_argument("--output_dir", default="predictions")
    p.add_argument("--model_name", default="bioptimus/H-optimus-0")
    p.add_argument("--image_size", type=int, default=None, help="Override image size")
    p.add_argument("--lora_r", type=int, default=None, help="Override LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=None, help="Override LoRA alpha")
    p.add_argument("--device", default=None, help="Force device (cuda / cpu)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- Load checkpoint config -------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ckpt.get("config", {})

    model_name = config.get("model_name", args.model_name)
    image_size = args.image_size or config.get("image_size", 224)
    lora_r = args.lora_r or config.get("lora_r", 16)
    lora_alpha = args.lora_alpha or config.get("lora_alpha", 32)
    lora_dropout = config.get("lora_dropout", 0.05)

    # --- Normalisation values ---------------------------------------------
    if "image_mean" in config and "image_std" in config:
        mean = np.array(config["image_mean"], dtype=np.float32)
        std = np.array(config["image_std"], dtype=np.float32)
    else:
        try:
            processor = AutoImageProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            mean = np.array(processor.image_mean, dtype=np.float32)
            std = np.array(processor.image_std, dtype=np.float32)
        except Exception:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # --- Build and load model ---------------------------------------------
    print("Loading model …")
    model = HOptimusLoRA(
        model_name=model_name,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_size=image_size,
    )
    model.load_trainable_state_dict(ckpt["trainable_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"Checkpoint epoch {ckpt.get('epoch', '?')} | val_loss {ckpt.get('val_loss', '?')}")

    # --- Collect input images ---------------------------------------------
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        )
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Run inference ----------------------------------------------------
    print(f"Processing {len(image_paths)} image(s) → {output_dir}/")
    for img_path in tqdm(image_paths):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [skip] Cannot read {img_path}")
            continue

        original_size = (img_bgr.shape[0], img_bgr.shape[1])
        tensor = preprocess(img_bgr, image_size, mean, std).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(tensor)

        result = postprocess(pred, original_size)
        cv2.imwrite(str(output_dir / img_path.name), result)

    print("Done.")


if __name__ == "__main__":
    main()
