import gc
import os
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = os.path.abspath(os.path.join(repo_root, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from utils import load_z123_pipe, resolve_torch_device
from scripts.sampling.simple_video_sample import (
    get_batch,
    get_unique_embedder_keys_from_conditioner,
    load_model,
)

NUM_FRAMES = 21
TARGET_SIZE = (576, 576)
DEFAULT_LOCAL_AUX_LAYOUTS = {
    6: {
        "target_indices": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "blend_weights": [0.55, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.72, 0.45],
    },
    10: {
        "target_indices": [6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "blend_weights": [0.45, 0.68, 0.76, 0.78, 0.78, 0.78, 0.76, 0.72, 0.58, 0.45],
    },
    13: {
        "target_indices": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "blend_weights": [0.45, 0.72, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.62, 0.50],
    },
}


def flush():
    gc.collect()
    torch.cuda.empty_cache()


class VAEProcessor:
    def __init__(self, vae, device=None):
        self.vae = vae
        if device is None:
            try:
                inferred_device = next(self.vae.parameters()).device
            except StopIteration:
                inferred_device = torch.device("cpu")
            if inferred_device.type == "cpu" and torch.cuda.is_available():
                inferred_device = torch.device("cuda")
            self.device = torch.device(inferred_device)
        else:
            self.device = torch.device(device)
        self.scale_factor = 0.18215

    def encode(self, images, batch_size=1):
        if next(self.vae.parameters()).device != self.device:
            self.vae.to(self.device)
        all_latents = []
        for i in tqdm(range(0, len(images), batch_size), desc="VAE Encoding"):
            batch_imgs = images[i:i + batch_size]
            batch_tensor = torch.stack(
                [ToTensor()(img).to(self.device) * 2.0 - 1.0 for img in batch_imgs]
            )
            with torch.no_grad():
                latents = self.vae.encode(batch_tensor).latent_dist.sample() * self.scale_factor
                all_latents.append(latents.cpu())
        self.vae.to("cpu")
        return torch.cat(all_latents, dim=0)

    def decode(self, latents, batch_size=1):
        if next(self.vae.parameters()).device != self.device:
            self.vae.to(self.device)
        latents = latents.to(dtype=torch.float32) / self.scale_factor
        all_images = []
        for i in tqdm(range(0, latents.shape[0], batch_size), desc="VAE Decoding"):
            batch_latents = latents[i:i + batch_size].to(self.device)
            with torch.no_grad():
                decoded = self.vae.decode(batch_latents).sample
                images = (decoded * 0.5 + 0.5).clamp(0, 1)
                for img_tensor in images:
                    all_images.append(ToPILImage()(img_tensor.cpu()))
        self.vae.to("cpu")
        return all_images


def process_image_smart(img, target_size=TARGET_SIZE):
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def load_raw_frames(mv_dir, target_size=TARGET_SIZE):
    paths = sorted([path for path in Path(mv_dir).iterdir() if path.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    if len(paths) < NUM_FRAMES:
        raise ValueError(f"Need at least {NUM_FRAMES} frames, got {len(paths)} in {mv_dir}")
    return [process_image_smart(Image.open(path), target_size) for path in paths[:NUM_FRAMES]]


def reorder_frames(frames, shift_index):
    if shift_index == NUM_FRAMES - 1:
        return frames
    return frames[shift_index + 1:] + frames[:shift_index + 1]


def build_stream_order(cond_index, num_frames=NUM_FRAMES):
    if cond_index == num_frames - 1:
        return list(range(num_frames))
    return list(range(cond_index + 1, num_frames)) + list(range(0, cond_index + 1))


def build_source_indices(cond_index, target_original_indices, num_frames=NUM_FRAMES):
    return [int((idx - cond_index - 1) % num_frames) for idx in target_original_indices]


def assign_frames_to_guides(cond_indices, num_frames=NUM_FRAMES):
    def circular_distance(a, b):
        diff = abs(a - b)
        return min(diff, num_frames - diff)

    ownership = {cond_index: [] for cond_index in cond_indices}
    for frame_idx in range(num_frames):
        owner = min(
            cond_indices,
            key=lambda cond_index: (circular_distance(frame_idx, cond_index), cond_indices.index(cond_index)),
        )
        ownership[owner].append(frame_idx)
    return ownership


def build_explicit_ownership(guide_specs, num_frames=NUM_FRAMES):
    ownership = {}
    assigned = set()
    specs_with_explicit = 0
    for spec in guide_specs:
        cond_index = int(spec["cond_index"])
        explicit = spec.get("owned_original_indices")
        if explicit is None:
            continue
        specs_with_explicit += 1
        normalized = []
        for frame_idx in explicit:
            frame_idx = int(frame_idx)
            if not 0 <= frame_idx < num_frames:
                raise ValueError(f"owned_original_indices out of range: {frame_idx}")
            if frame_idx in assigned:
                raise ValueError(f"Duplicate explicit owned frame index: {frame_idx}")
            assigned.add(frame_idx)
            normalized.append(frame_idx)
        ownership[cond_index] = normalized

    if specs_with_explicit == 0:
        return None
    if specs_with_explicit != len(guide_specs):
        raise ValueError("If one guide uses owned_original_indices, all guides in the case must provide it")

    missing = [idx for idx in range(num_frames) if idx not in assigned]
    if missing:
        missing_text = ",".join(f"{idx:02d}" for idx in missing)
        raise ValueError(f"Explicit owned_original_indices do not cover all frames: {missing_text}")
    return ownership


def reorder_tensor_to_stream_order(tensor, cond_index):
    order = build_stream_order(cond_index)
    index = torch.tensor(order, device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, index)


def build_generic_local_aux_layout(cond_index, num_frames=NUM_FRAMES):
    start = max(0, cond_index - 4)
    end = min(num_frames - 1, cond_index + 4)
    target_indices = list(range(start, end + 1))
    max_dist = max(abs(target_indices[0] - cond_index), abs(target_indices[-1] - cond_index), 1)
    blend_weights = []
    for target_idx in target_indices:
        dist = abs(target_idx - cond_index)
        weight = 0.78 - 0.33 * (dist / max_dist)
        blend_weights.append(round(max(0.45, min(0.78, weight)), 2))
    return {"target_indices": target_indices, "blend_weights": blend_weights}


def resolve_local_aux_layout(cond_index):
    layout = DEFAULT_LOCAL_AUX_LAYOUTS.get(cond_index)
    if layout is None:
        layout = build_generic_local_aux_layout(cond_index)
    return {
        "target_indices": list(layout["target_indices"]),
        "blend_weights": list(layout["blend_weights"]),
    }


def get_conditions(model, cond_tensor, version, motion_bucket_id, fps_id, elev, az_rad, device, aug=0.02):
    cond_tensor = cond_tensor.to(dtype=model.dtype)
    value_dict = {
        "cond_frames_without_noise": cond_tensor,
        "motion_bucket_id": torch.tensor([motion_bucket_id], device=device, dtype=torch.long),
        "fps_id": torch.tensor([fps_id], device=device, dtype=torch.long),
        "cond_aug": torch.tensor([aug], device=device, dtype=model.dtype),
        "cond_frames": cond_tensor + aug * torch.randn_like(cond_tensor),
    }
    if version == "sv3d_p":
        value_dict["polars_rad"] = torch.tensor([np.deg2rad(90 - elev)] * NUM_FRAMES, device=device, dtype=model.dtype)
        value_dict["azimuths_rad"] = torch.tensor(az_rad, device=device, dtype=model.dtype)

    with torch.no_grad():
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(model.conditioner),
            value_dict,
            [1, NUM_FRAMES],
            T=NUM_FRAMES,
            device=device,
        )
        c, uc = model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=["cond_frames", "cond_frames_without_noise"],
        )
    return c, uc


def calculate_adaptive_guidance(azimuths_deg, min_scale, max_scale, power=1.0, ref_index=0, device="cuda"):
    num_frames = len(azimuths_deg)
    indices = np.arange(num_frames)
    distance = np.abs(indices - int(ref_index))
    distance = np.minimum(distance, num_frames - distance)
    max_distance = max(num_frames // 2, 1)
    factor = np.power(distance / max_distance, power)
    scales = min_scale + (max_scale - min_scale) * factor
    return torch.from_numpy(scales).to(device, dtype=torch.float32).view(-1, 1, 1, 1)


def calculate_owned_adaptive_guidance(
    azimuths_deg,
    cond_index,
    owned_original_indices,
    min_scale,
    max_scale,
    power=1.0,
    device="cuda",
):
    if not 0 <= int(cond_index) < len(azimuths_deg):
        raise ValueError(f"cond_index out of range: {cond_index}")
    if owned_original_indices is not None:
        owned = [int(idx) for idx in owned_original_indices]
        if int(cond_index) not in owned:
            raise ValueError(
                f"cond_index {int(cond_index):02d} must be included in owned_original_indices "
                "so the guide reference frame is the minimum tar scale"
            )

    return calculate_adaptive_guidance(
        azimuths_deg,
        min_scale,
        max_scale,
        power=power,
        ref_index=cond_index,
        device=device,
    )


def format_frame_list(indices):
    return ",".join(f"{int(idx):02d}" for idx in indices)


def summarize_owned_tar_scales(scale_tensor, owned_original_indices):
    if scale_tensor is None or owned_original_indices is None:
        return ""
    parts = []
    for idx in owned_original_indices:
        scale = float(scale_tensor[int(idx)].item())
        parts.append(f"{int(idx):02d}:{scale:.6f}")
    return ", ".join(parts)


def summarize_global_tar_assignment(streams, use_adaptive_tgs, fallback_scale):
    assignment = {}
    duplicates = []
    for stream in streams:
        edit_idx = int(stream.get("edit_idx", -1))
        cond_index = int(stream["cond_index"])
        scale_tensor = stream.get("tar_scale_orig")
        for frame_idx in stream["owned_original_indices"]:
            frame_idx = int(frame_idx)
            if use_adaptive_tgs and scale_tensor is not None:
                scale = float(scale_tensor[frame_idx].item())
            else:
                scale = float(fallback_scale)
            if frame_idx in assignment:
                duplicates.append(frame_idx)
            assignment[frame_idx] = (edit_idx, cond_index, scale)

    lines = []
    for start in range(0, NUM_FRAMES, 7):
        parts = []
        for frame_idx in range(start, min(start + 7, NUM_FRAMES)):
            if frame_idx not in assignment:
                parts.append(f"{frame_idx:02d}:MISSING")
                continue
            edit_idx, cond_index, scale = assignment[frame_idx]
            parts.append(f"{frame_idx:02d}:edit{edit_idx}/src{cond_index:02d}={scale:.6f}")
        lines.append(" | ".join(parts))
    if duplicates:
        dup_text = ",".join(f"{idx:02d}" for idx in sorted(set(duplicates)))
        lines.append(f"WARNING duplicate owner indices: {dup_text}")
    return lines


def scheduler_sigmas(def_sigmas, scheduler_type, device, num_steps, use_custom_scheduler):
    if not use_custom_scheduler:
        return def_sigmas

    start_sigma = def_sigmas[0].item()
    end_sigma = def_sigmas[-1].item()
    t_space = torch.arange(num_steps + 1, device=device).float() / num_steps

    if scheduler_type == "default_slower":
        return (start_sigma ** (1 / 3) + t_space * (end_sigma ** (1 / 3) - start_sigma ** (1 / 3))) ** 3
    if scheduler_type == "slower":
        return start_sigma + (t_space ** 2) * (end_sigma - start_sigma)
    if scheduler_type == "cubic_fidelity":
        return start_sigma * (1 - t_space ** 1.5) ** 3
    if scheduler_type == "medium":
        return (start_sigma ** (1 / 3) + t_space * (end_sigma ** (1 / 3) - start_sigma ** (1 / 3))) ** 3
    if scheduler_type == "linear":
        return start_sigma + t_space * (end_sigma - start_sigma)
    return def_sigmas


def mix_cfg(v_uncond, v_cond, cfg_scale):
    return v_uncond + cfg_scale * (v_cond - v_uncond)


def sigma_to_unit(sigma, start_sigma, end_sigma):
    return ((sigma - end_sigma) / (start_sigma - end_sigma + 1e-8)).clamp(0.0, 1.0)


def get_latent_anchor(sample, velocity, sigma_unit):
    return sample + (1.0 - sigma_unit) * velocity


def build_soft_mask(reference_src, reference_tgt, threshold_scale=1.5, sharpness=12.0, min_value=0.10):
    latent_diff = torch.abs(reference_tgt - reference_src).mean(dim=1, keepdim=True)
    dynamic_thresh = latent_diff.mean() * threshold_scale
    soft_mask = torch.sigmoid((latent_diff - dynamic_thresh) * sharpness)
    return min_value + (1.0 - min_value) * soft_mask


def apply_latent_blend(zt, src_latents, strength=0.9, sharpness=18.0, percentile=0.96):
    if strength <= 0.0:
        return zt

    percentile = float(np.clip(percentile, 0.0, 1.0))
    zt_f = zt.to(dtype=torch.float32)
    src_f = src_latents.to(device=zt.device, dtype=torch.float32)
    edit_diff = torch.abs(zt_f - src_f).mean(dim=1, keepdim=True)
    global_thresh = torch.quantile(edit_diff.flatten(), percentile)
    edit_mask = (edit_diff / (global_thresh + 1e-6)).clamp(0.0, 1.0)
    edit_mask = torch.sigmoid((edit_mask - 0.5) * sharpness)
    correction = (1.0 - edit_mask) * strength * (src_f - zt_f)
    blended = (zt_f + correction).to(zt.dtype)
    print(
        f"\n[ColorC1 Latent Blend] strength={strength} | sharpness={sharpness} | "
        f"percentile={percentile} | global_thresh={global_thresh:.4f} | "
        f"edit_mask mean={edit_mask.mean():.3f} max={edit_mask.max():.3f}"
    )
    return blended


def compute_edit_direction(
    model,
    denoiser,
    zt_src,
    zt_tar,
    s_in_curr,
    s_in_next,
    c_src,
    uc_src,
    c_tgt,
    uc_tgt,
    src_guidance_scale,
    tar_guidance_scale,
    tar_scale_tensor,
    use_adaptive_tgs,
    use_anchorflow,
    sigma_unit,
    use_soft_mask=False,
    soft_mask_threshold_scale=1.5,
    soft_mask_sharpness=12.0,
    soft_mask_min=0.10,
    use_late_boost=False,
    late_boost_gain=0.30,
    use_source_velocity=False,
    source_velocity_scale=1.0,
    active_source_indices=None,
):
    _, vu_src, vc_src, _ = model.sampler.sampler_step(s_in_curr, s_in_next, denoiser, zt_src, c_src, uc_src, 0.0)
    _, vu_tgt, vc_tgt, _ = model.sampler.sampler_step(s_in_curr, s_in_next, denoiser, zt_tar, c_tgt, uc_tgt, 0.0)

    v_src = mix_cfg(vu_src, vc_src, src_guidance_scale)
    if use_adaptive_tgs:
        v_tgt = vu_tgt + tar_scale_tensor * (vc_tgt - vu_tgt)
    else:
        v_tgt = mix_cfg(vu_tgt, vc_tgt, tar_guidance_scale)

    if use_anchorflow:
        ft_src = get_latent_anchor(zt_src, v_src, sigma_unit)
        ft_tgt = get_latent_anchor(zt_tar, v_tgt, sigma_unit)
        delta = (2.0 - sigma_unit) * (ft_tgt - ft_src)
        mask_ref_src, mask_ref_tgt = ft_src, ft_tgt
    else:
        delta = v_tgt - v_src
        mask_ref_src, mask_ref_tgt = v_src, v_tgt

    if use_soft_mask:
        delta = build_soft_mask(
            mask_ref_src,
            mask_ref_tgt,
            threshold_scale=soft_mask_threshold_scale,
            sharpness=soft_mask_sharpness,
            min_value=soft_mask_min,
        ) * delta

    if use_late_boost:
        delta = (1.0 + late_boost_gain * (1.0 - sigma_unit)) * delta

    if use_source_velocity:
        delta = source_velocity_scale * v_src + delta

    if active_source_indices is not None:
        active = torch.as_tensor(active_source_indices, device=delta.device, dtype=torch.long)
        mask = torch.zeros((delta.shape[0], 1, 1, 1), device=delta.device, dtype=delta.dtype)
        if active.numel() > 0:
            mask.index_fill_(0, active, 1.0)
        delta = delta * mask

    return delta


@torch.no_grad()
def run_sampler(
    model,
    denoiser,
    zt_global,
    main_stream,
    aux_streams,
    num_steps,
    start_step,
    src_guidance_scale,
    tar_guidance_scale,
    scheduler_type,
    use_custom_scheduler,
    use_adaptive_tgs,
    use_anchorflow,
    use_soft_mask=False,
    soft_mask_threshold_scale=1.5,
    soft_mask_sharpness=12.0,
    soft_mask_min=0.10,
    use_late_boost=False,
    late_boost_gain=0.30,
    use_source_velocity=False,
    source_velocity_scale=1.0,
    use_local_aux_blend=False,
    n_avg=2,
):
    device = zt_global.device
    dummy_noise = torch.randn_like(zt_global).to(device)
    _, s_in, def_sigmas, _, _, _ = model.sampler.prepare_sampling_loop(dummy_noise, main_stream["c_src"], main_stream["uc_src"], num_steps)
    sigmas = scheduler_sigmas(def_sigmas, scheduler_type, device, num_steps, use_custom_scheduler)
    start_sigma = sigmas[0]
    end_sigma = sigmas[-1]
    to_time = lambda s: (s - sigmas[-1]) / (sigmas[0] - sigmas[-1] + 1e-8)

    print(
        f"\n[UnifiedSV3DEdit] guides={1 + len(aux_streams)} | n_avg={n_avg} | start_step={start_step}/{num_steps} | "
        f"custom_scheduler={use_custom_scheduler}({scheduler_type}) | adaptive_tgs={use_adaptive_tgs} | anchor={use_anchorflow}"
    )
    print(
        "  Extra features: "
        f"soft_mask={use_soft_mask} | late_boost={use_late_boost} | "
        f"source_vel={use_source_velocity} | local_aux_blend={use_local_aux_blend}"
    )
    print(f"  Main guide ({main_stream['name']}) owns global indices: {main_stream['target_indices_global']}")
    if main_stream.get("tar_scale_summary"):
        print(f"    Adaptive tar scales: {main_stream['tar_scale_summary']}")
    for aux in aux_streams:
        print(f"  Aux guide ({aux['name']}) owns global indices: {aux['target_indices_global']}")
        if aux.get("tar_scale_summary"):
            print(f"    Adaptive tar scales: {aux['tar_scale_summary']}")
    if not use_local_aux_blend:
        print("  Final per-frame tar assignment (original indices):")
        for line in summarize_global_tar_assignment([*aux_streams, main_stream], use_adaptive_tgs, tar_guidance_scale):
            print(f"    {line}")

    for aux in aux_streams:
        weights = aux.get("blend_weights")
        if weights is None:
            continue
        if torch.is_tensor(weights):
            aux["blend_weights"] = weights.to(device=device, dtype=zt_global.dtype).view(-1, 1, 1, 1)
        else:
            aux["blend_weights"] = torch.tensor(weights, device=device, dtype=zt_global.dtype).view(-1, 1, 1, 1)

    with torch.autocast(device.type):
        for i in tqdm(range(len(sigmas) - 1), desc="Sampling"):
            if i < start_step:
                continue

            sigma_i, sigma_next = sigmas[i], sigmas[i + 1]
            sigma_unit = sigma_to_unit(sigma_i, start_sigma, end_sigma)
            s_in_curr, s_in_next = s_in * sigma_i, s_in * sigma_next
            dt = to_time(sigma_next) - to_time(sigma_i)

            main_diff_acc = 0
            aux_diff_acc = [0 for _ in aux_streams]
            for _ in range(n_avg):
                noise_i = torch.randn_like(zt_global)

                zt_src_main = main_stream["x0_src"] + noise_i * sigma_i
                zt_tar_main = zt_global + noise_i * sigma_i
                main_diff_acc = main_diff_acc + compute_edit_direction(
                    model,
                    denoiser,
                    zt_src_main,
                    zt_tar_main,
                    s_in_curr,
                    s_in_next,
                    main_stream["c_src"],
                    main_stream["uc_src"],
                    main_stream["c_tgt"],
                    main_stream["uc_tgt"],
                    src_guidance_scale,
                    tar_guidance_scale,
                    main_stream["tar_scale_tensor"],
                    use_adaptive_tgs,
                    use_anchorflow,
                    sigma_unit,
                    use_soft_mask,
                    soft_mask_threshold_scale,
                    soft_mask_sharpness,
                    soft_mask_min,
                    use_late_boost,
                    late_boost_gain,
                    use_source_velocity,
                    source_velocity_scale,
                    main_stream["source_indices_stream"],
                )

                for stream_idx, aux in enumerate(aux_streams):
                    zt_src_aux = aux["x0_src"] + noise_i * sigma_i
                    zt_tar_aux = aux["zt"] + noise_i * sigma_i
                    aux_diff_acc[stream_idx] = aux_diff_acc[stream_idx] + compute_edit_direction(
                        model,
                        denoiser,
                        zt_src_aux,
                        zt_tar_aux,
                        s_in_curr,
                        s_in_next,
                        aux["c_src"],
                        aux["uc_src"],
                        aux["c_tgt"],
                        aux["uc_tgt"],
                        src_guidance_scale,
                        tar_guidance_scale,
                        aux["tar_scale_tensor"],
                        use_adaptive_tgs,
                        use_anchorflow,
                        sigma_unit,
                        use_soft_mask,
                        soft_mask_threshold_scale,
                        soft_mask_sharpness,
                        soft_mask_min,
                        use_late_boost,
                        late_boost_gain,
                        use_source_velocity,
                        source_velocity_scale,
                        aux["source_indices_stream"],
                    )

            main_diff = main_diff_acc / n_avg
            if use_local_aux_blend:
                global_diff = main_diff.clone()
                for stream_idx, aux in enumerate(aux_streams):
                    aux_diff = aux_diff_acc[stream_idx] / n_avg
                    aux["zt"] = aux["zt"] + dt * aux_diff
                    target_indices = aux["target_indices_global"]
                    source_indices = aux["source_indices_stream"]
                    weights = aux["blend_weights"]
                    global_diff[target_indices] = weights * aux_diff[source_indices] + (1.0 - weights) * global_diff[target_indices]
            else:
                global_diff = torch.zeros_like(main_diff)
                global_diff[main_stream["target_indices_global"]] = main_diff[main_stream["source_indices_stream"]]
                for stream_idx, aux in enumerate(aux_streams):
                    aux_diff = aux_diff_acc[stream_idx] / n_avg
                    aux["zt"] = aux["zt"] + dt * aux_diff
                    global_diff[aux["target_indices_global"]] = aux_diff[aux["source_indices_stream"]]

            zt_global = zt_global + dt * global_diff

    return zt_global


def build_runtime_streams(
    raw_frames,
    guide_specs,
    model,
    device,
    sv3d_version,
    motion_bucket_id,
    fps_id,
    elevations_deg,
    azimuths_rad,
    azimuths_deg,
    vae_processor,
    encode_batch_size,
    tar_guidance_scale,
    tar_guidance_max,
    guidance_power,
    use_adaptive_tgs,
    use_local_aux_blend=False,
):
    cond_indices = [int(spec["cond_index"]) for spec in guide_specs]
    main_cond_index = cond_indices[-1]
    ownership = None
    if not use_local_aux_blend:
        ownership = build_explicit_ownership(guide_specs)
        if ownership is None:
            ownership = assign_frames_to_guides(cond_indices)

    def prep_img(path):
        return (ToTensor()(process_image_smart(Image.open(path))) * 2 - 1).unsqueeze(0).to(device)

    streams = []
    for spec_idx, spec in enumerate(guide_specs):
        cond_index = int(spec["cond_index"])
        if use_local_aux_blend:
            if spec_idx == len(guide_specs) - 1:
                owned_original_indices = list(range(NUM_FRAMES))
                source_indices_stream = list(range(NUM_FRAMES))
                target_indices_global = list(range(NUM_FRAMES))
                blend_weights = None
            else:
                layout = resolve_local_aux_layout(cond_index)
                owned_original_indices = list(layout["target_indices"])
                source_indices_stream = build_source_indices(cond_index, owned_original_indices)
                target_indices_global = build_source_indices(main_cond_index, owned_original_indices)
                blend_weights = list(layout["blend_weights"])
        else:
            owned_original_indices = ownership[cond_index]
            source_indices_stream = build_source_indices(cond_index, owned_original_indices)
            target_indices_global = build_source_indices(main_cond_index, owned_original_indices)
            blend_weights = None

        x0_src = vae_processor.encode(reorder_frames(raw_frames, cond_index), batch_size=encode_batch_size).to(dtype=torch.float16)
        c_src, uc_src = get_conditions(
            model,
            prep_img(spec["src_condition_path"]),
            sv3d_version,
            motion_bucket_id,
            fps_id,
            elevations_deg,
            azimuths_rad,
            device,
        )
        c_tgt, uc_tgt = get_conditions(
            model,
            prep_img(spec["tgt_condition_path"]),
            sv3d_version,
            motion_bucket_id,
            fps_id,
            elevations_deg,
            azimuths_rad,
            device,
        )

        tar_scale_tensor = None
        tar_scale_orig = None
        tar_scale_summary = ""
        if use_adaptive_tgs:
            tar_scale_orig = calculate_owned_adaptive_guidance(
                azimuths_deg,
                cond_index,
                owned_original_indices,
                tar_guidance_scale,
                tar_guidance_max,
                power=guidance_power,
                device=device,
            )
            tar_scale_tensor = reorder_tensor_to_stream_order(tar_scale_orig, cond_index)
            tar_scale_summary = summarize_owned_tar_scales(tar_scale_orig, owned_original_indices)

        streams.append(
            {
                **spec,
                "cond_index": cond_index,
                "owned_original_indices": owned_original_indices,
                "source_indices_stream": source_indices_stream,
                "target_indices_global": target_indices_global,
                "x0_src": x0_src.to(device),
                "c_src": c_src,
                "uc_src": uc_src,
                "c_tgt": c_tgt,
                "uc_tgt": uc_tgt,
                "tar_scale_tensor": tar_scale_tensor,
                "tar_scale_orig": tar_scale_orig,
                "tar_scale_summary": tar_scale_summary,
                "blend_weights": blend_weights,
            }
        )

    return streams


def run_sv3d_edit(
    guide_specs,
    original_mv,
    save_dir,
    device_number=0,
    seed=18,
    T_steps=50,
    n_max=31,
    src_guidance_scale=3.5,
    tar_guidance_scale=5.0,
    tar_guidance_max=10.0,
    guidance_power=1.0,
    scheduler_type="default",
    use_custom_scheduler=False,
    use_adaptive_tgs=False,
    use_anchorflow=False,
    use_soft_mask=False,
    soft_mask_threshold_scale=1.5,
    soft_mask_sharpness=12.0,
    soft_mask_min=0.10,
    use_late_boost=False,
    late_boost_gain=0.30,
    use_source_velocity=False,
    source_velocity_scale=1.0,
    use_latent_blend=False,
    latent_blend_strength=0.9,
    latent_mask_sharpness=18.0,
    latent_mask_percentile=0.96,
    use_local_aux_blend=False,
    sv3d_version="sv3d_p",
    motion_bucket_id=127,
    fps_id=6,
    elevations_deg=10.0,
    azimuths_deg=None,
    encode_batch_size=1,
    pipeline=None,
):
    if not guide_specs:
        raise ValueError("guide_specs cannot be empty")

    device = resolve_torch_device(device_number)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if azimuths_deg is None:
        azimuths_deg = np.linspace(0, 360, NUM_FRAMES + 1)[1:] % 360
    azimuths_rad = [np.deg2rad(value) for value in azimuths_deg]

    if pipeline is None:
        pipeline = load_z123_pipe(device_number)

    model = None
    streams = None
    main_stream = None
    aux_streams = None
    zt_global = None
    frames = None
    try:
        vae_processor = VAEProcessor(pipeline.vae.to(torch.float32), device=device)
        raw_frames = load_raw_frames(original_mv)
        pipeline.vae.to("cpu")
        flush()

        model, _ = load_model(
            f"scripts/sampling/configs/{sv3d_version}.yaml",
            str(device),
            NUM_FRAMES,
            T_steps,
            False,
            build_filter=False,
        )
        model = model.to(device).eval().to(dtype=torch.float16)

        streams = build_runtime_streams(
            raw_frames,
            guide_specs,
            model,
            device,
            sv3d_version,
            motion_bucket_id,
            fps_id,
            elevations_deg,
            azimuths_rad,
            azimuths_deg,
            vae_processor,
            encode_batch_size,
            tar_guidance_scale,
            tar_guidance_max,
            guidance_power,
            use_adaptive_tgs,
            use_local_aux_blend,
        )

        main_stream = streams[-1]
        aux_streams = []
        for stream in streams[:-1]:
            aux_streams.append({**stream, "zt": stream["x0_src"].clone()})
        zt_global = main_stream["x0_src"].clone()
        main_cond_index = int(main_stream["cond_index"])

        def denoiser(input_tensor, sigma, cond):
            return model.denoiser(
                model.model,
                input_tensor,
                sigma,
                cond,
                image_only_indicator=torch.zeros(2, NUM_FRAMES, device=device, dtype=model.dtype),
                num_video_frames=NUM_FRAMES,
            )

        zt_global = run_sampler(
            model,
            denoiser,
            zt_global,
            main_stream,
            aux_streams,
            T_steps,
            max(0, T_steps - n_max),
            src_guidance_scale,
            tar_guidance_scale,
            scheduler_type,
            use_custom_scheduler,
            use_adaptive_tgs,
            use_anchorflow,
            use_soft_mask,
            soft_mask_threshold_scale,
            soft_mask_sharpness,
            soft_mask_min,
            use_late_boost,
            late_boost_gain,
            use_source_velocity,
            source_velocity_scale,
            use_local_aux_blend,
        )

        del model
        model = None
        flush()

        if use_latent_blend:
            zt_global = apply_latent_blend(
                zt_global,
                main_stream["x0_src"],
                strength=latent_blend_strength,
                sharpness=latent_mask_sharpness,
                percentile=latent_mask_percentile,
            )

        pipeline.vae.to(device)
        frames = VAEProcessor(pipeline.vae, device=device).decode(
            zt_global.to(torch.float32),
            batch_size=encode_batch_size,
        )
        if main_cond_index != NUM_FRAMES - 1:
            split = NUM_FRAMES - (main_cond_index + 1)
            frames = frames[split:] + frames[:split]

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(save_dir / f"{i:02d}.png")
        imageio.mimsave(save_dir / "preview.gif", [np.array(frame) for frame in frames], fps=3, loop=0)
        return save_dir
    finally:
        del frames
        del zt_global
        del aux_streams
        del main_stream
        del streams
        if model is not None:
            del model
        try:
            pipeline.vae.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        flush()
