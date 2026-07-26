import os
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from huggingface_hub import hf_hub_download, snapshot_download
from pipeline import Zero123PlusPipeline
from PIL import Image

ZERO123PLUS_REPO_ID = "sudo-ai/zero123plus-v1.2"
INSTANTMESH_REPO_ID = "TencentARC/InstantMesh"
INSTANTMESH_UNET_FILENAME = "diffusion_pytorch_model.bin"


def _existing_path(path_like):
    if not path_like:
        return None
    path = Path(path_like).expanduser()
    if path.exists():
        return str(path)
    return None


def _resolve_zero123plus_source():
    candidate_keys = ["ZERO123PLUS_MODEL_DIR", "ZERO123PLUS_LOCAL_DIR"]
    for env_key in candidate_keys:
        resolved = _existing_path(os.environ.get(env_key))
        if resolved:
            print(f"[load_z123_pipe] Use ZERO123++ from ${env_key}: {resolved}")
            return resolved

    try:
        cached_dir = snapshot_download(ZERO123PLUS_REPO_ID, local_files_only=True)
        print(f"[load_z123_pipe] Use cached ZERO123++ snapshot: {cached_dir}")
        return cached_dir
    except Exception as exc:
        print(f"[load_z123_pipe] Cached ZERO123++ snapshot unavailable, fallback to hub download: {exc}")
        return ZERO123PLUS_REPO_ID


def _resolve_instantmesh_unet_path():
    candidates = [
        os.environ.get("INSTANTMESH_UNET_PATH"),
        os.environ.get("ZERO123PLUS_UNET_PATH"),
        "ckpts/diffusion_pytorch_model.bin",
        "checkpoints/diffusion_pytorch_model.bin",
    ]
    for candidate in candidates:
        resolved = _existing_path(candidate)
        if resolved:
            print(f"[load_z123_pipe] Use local InstantMesh UNet: {resolved}")
            return resolved

    try:
        cached_path = hf_hub_download(
            repo_id=INSTANTMESH_REPO_ID,
            filename=INSTANTMESH_UNET_FILENAME,
            repo_type="model",
            local_files_only=True,
        )
        print(f"[load_z123_pipe] Use cached InstantMesh UNet: {cached_path}")
        return cached_path
    except Exception as exc:
        print(f"[load_z123_pipe] Cached InstantMesh UNet unavailable, fallback to hub download: {exc}")
        return hf_hub_download(
            repo_id=INSTANTMESH_REPO_ID,
            filename=INSTANTMESH_UNET_FILENAME,
            repo_type="model",
        )


def _parse_visible_cuda_devices():
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw_value:
        return raw_value, None

    tokens = [token.strip() for token in raw_value.split(",") if token.strip()]
    if tokens and all(token.isdigit() for token in tokens):
        return raw_value, [int(token) for token in tokens]
    return raw_value, None


def resolve_torch_device(device_number):
    if not torch.cuda.is_available():
        visible_raw, _ = _parse_visible_cuda_devices()
        message = (
            "CUDA is not available in this Python process. "
            "This SV3D runner requires a CUDA GPU because the pipeline is loaded in float16."
        )
        if visible_raw:
            message += f" CUDA_VISIBLE_DEVICES={visible_raw}."
        message += " Choose a usable GPU, for example CUDA_VISIBLE_DEVICES=3 DEVICE_NUMBER=0 on this machine right now."
        raise RuntimeError(message)

    requested_index = int(device_number)
    visible_count = torch.cuda.device_count()
    if 0 <= requested_index < visible_count:
        return torch.device(f"cuda:{requested_index}")

    visible_raw, visible_physical_ids = _parse_visible_cuda_devices()
    if visible_physical_ids and requested_index in visible_physical_ids:
        local_index = visible_physical_ids.index(requested_index)
        print(
            "[resolve_torch_device] "
            f"Remap physical GPU {requested_index} to local cuda:{local_index} "
            f"under CUDA_VISIBLE_DEVICES={visible_raw}"
        )
        return torch.device(f"cuda:{local_index}")

    message = (
        f"Invalid CUDA device index {requested_index}: "
        f"PyTorch currently sees {visible_count} visible device(s)."
    )
    if visible_raw:
        message += f" CUDA_VISIBLE_DEVICES={visible_raw}."
        if visible_physical_ids:
            local_map = ", ".join(
                f"cuda:{local_idx}->GPU {physical_id}"
                for local_idx, physical_id in enumerate(visible_physical_ids)
            )
            message += f" Local mapping is [{local_map}]."
        message += " When CUDA_VISIBLE_DEVICES is set, DEVICE_NUMBER must use the local cuda index."
    else:
        message += " DEVICE_NUMBER must be in [0, visible_count - 1]."
    raise ValueError(message)


def load_z123_pipe(device_number):
    device = resolve_torch_device(device_number)

    pipeline_source = _resolve_zero123plus_source()
    pipeline = Zero123PlusPipeline.from_pretrained(
        pipeline_source, torch_dtype=torch.float16
    )
    # DDPM supports custom timesteps
    pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

    unet_ckpt_path = _resolve_instantmesh_unet_path()
    state_dict = torch.load(unet_ckpt_path, map_location="cpu")
    pipeline.unet.load_state_dict(state_dict, strict=True)

    pipeline.to(device)
    return pipeline


def add_white_bg(image):
    # Check if image has transparency (RGBA or LA mode)
    if image.mode in ("RGBA", "LA"):
        # Create a white background image of the same size
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        # Paste original image onto white background using alpha channel as mask
        white_bg.paste(image, mask=image.split()[-1])
        return white_bg
    # If no transparency, return the original image
    return image
