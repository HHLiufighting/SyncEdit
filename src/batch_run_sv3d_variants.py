import argparse
import gc
import os
import re
import sys
from pathlib import Path

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from core_editp23_like_sv3d import load_z123_pipe, run_sv3d_edit
from version_presets import CUSTOM_SCHEDULER_CHOICES, PRESET_CONFIGS

DATASET_DIRS = {
    "add": Path("assets/add"),
    "delete": Path("assets/delete"),
    "replace": Path("assets/replace"),
}
GUIDE_MODES = ("guide1", "guide2", "guide3", "guide4")
EXPECTED_FRAME_NAMES = [f"{idx:02d}.png" for idx in range(21)]
CASE_TOKEN_ALIASES = {
    "bird_hat": {"brid_hat"},
    "girl_wing": {"gril_wing"},
    "girl_horn": {"gril_horn"},
    "ironman_glasses": {"ironmanm_glasses"},
    "elf_shoe": {"elfm_shoe"},
    "rabbitT_carrot": {"rabbit_carrot"},
    "cat_choker": {"cat_hair"},
}


def output_guide_dirname(guide_mode: str, n_max: int) -> str:
    return f"{guide_mode}_nmax{n_max}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified batch runner for the public SV3D editing release.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", type=str, required=True, choices=sorted(PRESET_CONFIGS))
    parser.add_argument("--guide_mode", type=str, required=True, choices=GUIDE_MODES)
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_DIRS))
    parser.add_argument("--case_dir", type=Path, default=Path("src/cases"))
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--case_ids", nargs="*", default=None, help="支持 case_id/src_dir/edit_dir，或 dataset:case_id")
    parser.add_argument(
        "--exclude_case_ids",
        nargs="*",
        default=None,
        help="排除的 case，支持 case_id/src_dir/edit_dir，或 dataset:case_id",
    )
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--T_steps", type=int, default=50)
    parser.add_argument("--n_max", type=int, default=31)
    parser.add_argument("--src_guidance_scale", type=float, default=3.5)
    parser.add_argument("--tar_guidance_scale", type=float, default=5.0)
    parser.add_argument("--tar_guidance_max", type=float, default=10.0)
    parser.add_argument("--guidance_power", type=float, default=1.0)
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="",
        choices=["", *CUSTOM_SCHEDULER_CHOICES],
        help="只对带自定义调度版本生效；留空则用该版本默认调度配置。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def parse_guide_pairs(text: str):
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("guide_pairs cannot be empty")
    pairs = []
    tokens = [token for token in re.split(r"[;,]+|\s+", raw) if token]
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"Invalid guide pair token: {token}")
        left, right = token.split(":", 1)
        pairs.append((int(left), int(right)))
    return pairs


def parse_frame_indices(text: str):
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("frame indices cannot be empty")

    indices = []
    for token in [token for token in re.split(r"[,\s]+", raw) if token]:
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid frame range: {token}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(token))

    unique = []
    seen = set()
    for index in indices:
        if not 0 <= index <= 20:
            raise ValueError(f"Frame index must be within [0, 20], got: {index}")
        if index not in seen:
            unique.append(index)
            seen.add(index)
    return unique


def parse_owned_indices(text: str):
    raw = str(text or "").strip()
    if not raw:
        return {}

    mapping = {}
    all_indices = set()
    groups = [group.strip() for group in raw.split(";") if group.strip()]
    for group in groups:
        if ":" not in group:
            raise ValueError(f"Invalid owned_indices token: {group}")
        left, right = group.split(":", 1)
        edit_idx = int(left.strip())
        owned = parse_frame_indices(right)
        overlap = all_indices.intersection(owned)
        if overlap:
            overlap_text = ",".join(f"{idx:02d}" for idx in sorted(overlap))
            raise ValueError(f"owned_indices contains duplicate frame assignments: {overlap_text}")
        mapping[edit_idx] = owned
        all_indices.update(owned)
    return mapping


def format_owned_indices(mapping: dict[int, list[int]] | None) -> str:
    if not mapping:
        return ""
    parts = []
    for edit_idx in sorted(mapping):
        indices_text = ",".join(f"{idx:02d}" for idx in mapping[edit_idx])
        parts.append(f"{edit_idx}:{indices_text}")
    return ";".join(parts)


def parse_case_file(case_file: Path):
    records = []
    with case_file.open("r", encoding="utf-8") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 4:
                raise ValueError(
                    f"{case_file}:{lineno} must contain at least 4 '|' separated fields: "
                    f"case_id | src_dir | edit_dir | guide_pairs=..."
                )
            record = {
                "case_id": parts[0],
                "src_dir": parts[1],
                "edit_dir": parts[2],
                "guide_pairs": None,
                "owned_indices": {},
                "note": "",
            }
            for extra in parts[3:]:
                if not extra:
                    continue
                if "=" in extra:
                    key, value = [x.strip() for x in extra.split("=", 1)]
                    if key in {"guide_pairs", "pairs"}:
                        record["guide_pairs"] = parse_guide_pairs(value)
                    elif key in {"owned_indices", "ownership"}:
                        record["owned_indices"] = parse_owned_indices(value)
                    elif key == "note":
                        record["note"] = value
                    else:
                        record[key] = value
                else:
                    record["note"] = (record["note"] + " " + extra).strip()
            if record["guide_pairs"] is None:
                raise ValueError(f"{case_file}:{lineno} missing guide_pairs=...")
            records.append(record)
    if not records:
        raise ValueError(f"No valid records found in case file: {case_file}")
    return records


def find_numeric_image_paths(folder: Path):
    numeric_paths = {}
    for path in folder.iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        try:
            numeric_paths[int(path.stem)] = path
        except ValueError:
            continue
    return numeric_paths


def resolve_source_frame_path(src_dir: Path, src_idx: int) -> Path:
    for path in [src_dir / f"{src_idx:02d}.png", src_dir / f"{src_idx}.png"]:
        if path.exists():
            return path
    numeric_paths = find_numeric_image_paths(src_dir)
    if not numeric_paths:
        raise FileNotFoundError(f"No numeric source frames found in: {src_dir}")
    nearest_idx = min(numeric_paths, key=lambda idx: (abs(idx - src_idx), idx))
    nearest_path = numeric_paths[nearest_idx]
    print(
        f"[InputResolver] Missing source frame {src_idx:02d}.png in {src_dir}; "
        f"use nearest existing frame {nearest_idx:02d}.png instead."
    )
    return nearest_path


def resolve_edit_image_path(edit_dir: Path, edit_idx: int) -> Path:
    candidates = [
        edit_dir / f"{edit_idx}.png",
        edit_dir / f"{edit_idx:02d}.png",
        edit_dir / f"{edit_idx}.jpg",
        edit_dir / f"{edit_idx:02d}.jpg",
        edit_dir / f"{edit_idx}.jpeg",
        edit_dir / f"{edit_idx:02d}.jpeg",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing edited guide image {edit_idx}.png/.jpg/.jpeg (or zero-padded variant) in {edit_dir}"
    )


def build_guide_specs(dataset_root: Path, record: dict):
    src_mv_dir = dataset_root / record["src_dir"]
    edit_dir = dataset_root / record["edit_dir"]
    owned_indices_map = dict(record.get("owned_indices") or {})
    valid_edit_indices = {edit_idx for edit_idx, _ in record["guide_pairs"]}
    unknown_edit_indices = sorted(set(owned_indices_map) - valid_edit_indices)
    if unknown_edit_indices:
        unknown_text = ",".join(str(idx) for idx in unknown_edit_indices)
        raise ValueError(f"owned_indices references unknown edit image ids: {unknown_text}")
    guide_specs = []
    for idx, (edit_idx, src_idx) in enumerate(record["guide_pairs"], start=1):
        guide_specs.append(
            {
                "name": f"guide{idx}_edit{edit_idx}_src{src_idx:02d}",
                "src_condition_path": str(resolve_source_frame_path(src_mv_dir, src_idx)),
                "tgt_condition_path": str(resolve_edit_image_path(edit_dir, edit_idx)),
                "cond_index": src_idx,
                "edit_idx": edit_idx,
                "owned_original_indices": list(owned_indices_map[edit_idx]) if edit_idx in owned_indices_map else None,
            }
        )
    return src_mv_dir, edit_dir, guide_specs


def validate_case_inputs(src_mv_dir: Path, edit_dir: Path, guide_specs: list[dict]) -> tuple[bool, list[str]]:
    missing = []
    if not src_mv_dir.exists():
        missing.append(f"src_mv_dir -> {src_mv_dir}")
    if not edit_dir.exists():
        missing.append(f"edit_dir -> {edit_dir}")
    for guide in guide_specs:
        for key in ["src_condition_path", "tgt_condition_path"]:
            if not Path(guide[key]).exists():
                missing.append(f"{guide['name']} {key} -> {guide[key]}")
    numeric_frames = find_numeric_image_paths(src_mv_dir) if src_mv_dir.is_dir() else {}
    if len(numeric_frames) < 21:
        missing.append(f"src_mv_dir has fewer than 21 numeric frames: {src_mv_dir}")
    return len(missing) == 0, missing


def match_case_token(dataset_name: str, record: dict, token: str) -> bool:
    token = str(token).strip()
    if not token:
        return False
    lower = token.lower()
    dataset_prefix = None
    inner = token
    if ":" in token:
        prefix, rest = token.split(":", 1)
        if prefix in DATASET_DIRS:
            dataset_prefix = prefix
            inner = rest
    elif "/" in token:
        prefix, rest = token.split("/", 1)
        if prefix in DATASET_DIRS:
            dataset_prefix = prefix
            inner = rest

    if dataset_prefix is not None and dataset_prefix != dataset_name:
        return False

    candidates = {
        str(record.get("case_id", "")).strip(),
        str(record.get("src_dir", "")).strip(),
        str(record.get("edit_dir", "")).strip(),
        f"{dataset_name}/{record.get('case_id', '')}",
    }
    expanded = set()
    for candidate in candidates:
        expanded.add(candidate)
        expanded.update(CASE_TOKEN_ALIASES.get(candidate, set()))

    return inner in expanded or lower == f"{dataset_name}:{str(record.get('case_id', '')).lower()}"


def format_guide_pairs(pairs):
    return ",".join(f"{edit_idx}:{src_idx:02d}" for edit_idx, src_idx in pairs)


def safe_token(value):
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    token = "".join(ch if ch in allowed else "_" for ch in str(value).strip())
    return token[:160] or "default"


def inspect_case_render(case_output_dir: Path) -> dict:
    existing_frames = [frame_name for frame_name in EXPECTED_FRAME_NAMES if (case_output_dir / frame_name).is_file()]
    missing_frames = [frame_name for frame_name in EXPECTED_FRAME_NAMES if frame_name not in existing_frames]
    return {
        "frame_count": len(existing_frames),
        "existing_frames": existing_frames,
        "missing_frames": missing_frames,
        "complete": len(existing_frames) == len(EXPECTED_FRAME_NAMES),
        "preview_exists": (case_output_dir / "preview.gif").is_file(),
        "run_config_exists": (case_output_dir / "run_config.txt").is_file(),
    }


def is_case_render_complete(case_output_dir: Path) -> bool:
    return inspect_case_render(case_output_dir)["complete"]


def cleanup_after_case_failure() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass


def failure_summary_path(output_root: Path, args: argparse.Namespace) -> Path:
    return output_root / f"failed_cases__{args.preset}__{args.guide_mode}__nmax{args.n_max}.txt"


def write_failure_summary(path: Path, failures: list[dict]) -> None:
    if not failures:
        if path.exists():
            path.unlink()
        return
    with path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(failures, start=1):
            f.write(f"[{idx}]\n")
            for key in ["dataset", "case_id", "src_dir", "edit_dir", "guide_pairs", "output_dir", "error_type", "error_message"]:
                f.write(f"{key}={item.get(key, '')}\n")
            f.write("\n")


def write_run_config(path: Path, args: argparse.Namespace, preset: dict, dataset_name: str, record: dict, src_mv_dir: Path, edit_dir: Path, guide_specs: list[dict], scheduler_type: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"preset={args.preset}\n")
        f.write(f"version_name={preset['version_name']}\n")
        f.write(f"description={preset['description']}\n")
        f.write(f"dataset={dataset_name}\n")
        f.write(f"guide_mode={args.guide_mode}\n")
        f.write(f"case_id={record['case_id']}\n")
        f.write(f"src_dir={record['src_dir']}\n")
        f.write(f"edit_dir={record['edit_dir']}\n")
        f.write(f"guide_pairs={format_guide_pairs(record['guide_pairs'])}\n")
        f.write(f"owned_indices={format_owned_indices(record.get('owned_indices'))}\n")
        f.write(f"note={record.get('note', '')}\n")
        f.write(f"src_mv_dir={src_mv_dir}\n")
        f.write(f"edit_dir_path={edit_dir}\n")
        f.write(f"device_number={args.device_number}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"T_steps={args.T_steps}\n")
        f.write(f"n_max={args.n_max}\n")
        f.write(f"src_guidance_scale={args.src_guidance_scale}\n")
        f.write(f"tar_guidance_scale={args.tar_guidance_scale}\n")
        f.write(f"tar_guidance_max={args.tar_guidance_max}\n")
        f.write(f"guidance_power={args.guidance_power}\n")
        f.write(f"scheduler_type={scheduler_type}\n")
        f.write(f"use_custom_scheduler={preset['use_custom_scheduler']}\n")
        f.write(f"use_adaptive_tgs={preset['use_adaptive_tgs']}\n")
        f.write(f"use_anchorflow={preset['use_anchorflow']}\n")
        f.write(f"use_soft_mask={preset.get('use_soft_mask', False)}\n")
        f.write(f"soft_mask_threshold_scale={preset.get('soft_mask_threshold_scale', '')}\n")
        f.write(f"soft_mask_sharpness={preset.get('soft_mask_sharpness', '')}\n")
        f.write(f"soft_mask_min={preset.get('soft_mask_min', '')}\n")
        f.write(f"use_late_boost={preset.get('use_late_boost', False)}\n")
        f.write(f"late_boost_gain={preset.get('late_boost_gain', '')}\n")
        f.write(f"use_source_velocity={preset.get('use_source_velocity', False)}\n")
        f.write(f"source_velocity_scale={preset.get('source_velocity_scale', '')}\n")
        f.write(f"use_latent_blend={preset.get('use_latent_blend', False)}\n")
        f.write(f"latent_blend_strength={preset.get('latent_blend_strength', '')}\n")
        f.write(f"latent_mask_sharpness={preset.get('latent_mask_sharpness', '')}\n")
        f.write(f"latent_mask_percentile={preset.get('latent_mask_percentile', '')}\n")
        f.write(f"use_local_aux_blend={preset.get('use_local_aux_blend', False)}\n")
        for idx, guide in enumerate(guide_specs, start=1):
            f.write(f"guide{idx}_name={guide['name']}\n")
            f.write(f"guide{idx}_src_condition_path={guide['src_condition_path']}\n")
            f.write(f"guide{idx}_tgt_condition_path={guide['tgt_condition_path']}\n")
            f.write(f"guide{idx}_cond_index={guide['cond_index']}\n")
            f.write(f"guide{idx}_owned_original_indices={','.join(f'{x:02d}' for x in guide.get('owned_original_indices') or [])}\n")


def main(args: argparse.Namespace) -> None:
    preset = PRESET_CONFIGS[args.preset]
    scheduler_type = args.scheduler_type or preset["default_scheduler_type"]
    requested_datasets = []
    for dataset in args.datasets:
        if dataset not in DATASET_DIRS:
            raise ValueError(f"Unsupported dataset: {dataset}. Choose from {sorted(DATASET_DIRS)}")
        if dataset not in requested_datasets:
            requested_datasets.append(dataset)

    case_dir = args.case_dir.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 96)
    print(f"[src Unified Variant Runner] preset={args.preset} guide_mode={args.guide_mode}")
    print(f"Description : {preset['description']}")
    print(f"Datasets    : {requested_datasets}")
    print(f"Scheduler   : {scheduler_type} (custom={preset['use_custom_scheduler']})")
    print(f"Adaptive TGS: {preset['use_adaptive_tgs']} | Anchor: {preset['use_anchorflow']}")
    print(
        "Extra Feat. : "
        f"soft_mask={preset.get('use_soft_mask', False)} | "
        f"late_boost={preset.get('use_late_boost', False)} | "
        f"source_vel={preset.get('use_source_velocity', False)} | "
        f"latent_blend={preset.get('use_latent_blend', False)} | "
        f"local_aux_blend={preset.get('use_local_aux_blend', False)}"
    )
    print(f"Output Root : {output_root}")
    print("#" * 96 + "\n")

    print(f"Loading pipeline on device {args.device_number} ...")
    pipeline = load_z123_pipe(args.device_number)

    total = success = skipped_existing = skipped_missing = failed = 0
    failures = []
    failed_cases_log = failure_summary_path(output_root, args)
    guide_output_dir = output_guide_dirname(args.guide_mode, args.n_max)
    try:
        for dataset_name in requested_datasets:
            dataset_root = DATASET_DIRS[dataset_name].resolve()
            case_file = (case_dir / f"{dataset_name}_{args.guide_mode}.txt").resolve()
            if not dataset_root.is_dir():
                print(f"[Skip dataset] missing dataset root: {dataset_root}")
                continue
            if not case_file.is_file():
                print(f"[Skip dataset] missing case file: {case_file}")
                continue

            records = parse_case_file(case_file)
            if args.case_ids:
                records = [
                    record for record in records
                    if any(match_case_token(dataset_name, record, token) for token in args.case_ids)
                ]
            if args.exclude_case_ids:
                records = [
                    record for record in records
                    if not any(match_case_token(dataset_name, record, token) for token in args.exclude_case_ids)
                ]
            if not records:
                print(f"[Skip dataset] no matched cases for dataset={dataset_name}")
                continue

            print(f"\n=== Dataset: {dataset_name} | cases: {len(records)} ===")
            for record in records:
                total += 1
                case_id = record["case_id"]
                try:
                    src_mv_dir, edit_dir, guide_specs = build_guide_specs(dataset_root, record)
                except Exception as exc:  # noqa: BLE001
                    skipped_missing += 1
                    print(f"[Skip input] {dataset_name}/{case_id}: {exc}")
                    continue

                valid, missing = validate_case_inputs(src_mv_dir, edit_dir, guide_specs)
                if not valid:
                    skipped_missing += 1
                    print(f"[Skip missing] {dataset_name}/{case_id}: {'; '.join(missing)}")
                    continue

                guide_token = safe_token(format_guide_pairs(record["guide_pairs"]).replace(":", "-").replace(",", "_"))
                run_tag = (
                    f"{args.preset}_{args.guide_mode}_{dataset_name}_"
                    f"nmax{args.n_max}_sch_{safe_token(scheduler_type)}_guides_{guide_token}"
                )
                case_output_dir = output_root / guide_output_dir / dataset_name / safe_token(case_id) / run_tag
                render_state = inspect_case_render(case_output_dir)
                if render_state["complete"] and not args.overwrite:
                    skipped_existing += 1
                    print(
                        f"[Skip existing] completed render found: {case_output_dir} "
                        f"(frames={render_state['frame_count']}/21)"
                    )
                    continue
                if case_output_dir.exists() and render_state["frame_count"] > 0 and not args.overwrite:
                    missing_preview = ""
                    if render_state["missing_frames"]:
                        missing_preview = ",".join(render_state["missing_frames"][:5])
                        if len(render_state["missing_frames"]) > 5:
                            missing_preview += ",..."
                    print(
                        f"[Resume] incomplete render detected: {case_output_dir} "
                        f"(frames={render_state['frame_count']}/21, "
                        f"missing={missing_preview or 'none'}, "
                        f"preview={'yes' if render_state['preview_exists'] else 'no'})"
                    )

                case_output_dir.mkdir(parents=True, exist_ok=True)
                write_run_config(
                    case_output_dir / "run_config.txt",
                    args,
                    preset,
                    dataset_name,
                    record,
                    src_mv_dir,
                    edit_dir,
                    guide_specs,
                    scheduler_type,
                )

                print(f"[Run] {dataset_name}/{case_id}")
                print(f"      source mv : {src_mv_dir}")
                print(f"      edit dir  : {edit_dir}")
                print(f"      guide set : {format_guide_pairs(record['guide_pairs'])}")
                print(f"      output    : {case_output_dir}")

                try:
                    run_sv3d_edit(
                        guide_specs=guide_specs,
                        original_mv=str(src_mv_dir),
                        save_dir=str(case_output_dir),
                        device_number=args.device_number,
                        seed=args.seed,
                        T_steps=args.T_steps,
                        n_max=args.n_max,
                        src_guidance_scale=args.src_guidance_scale,
                        tar_guidance_scale=args.tar_guidance_scale,
                        tar_guidance_max=args.tar_guidance_max,
                        guidance_power=args.guidance_power,
                        scheduler_type=scheduler_type,
                        use_custom_scheduler=preset["use_custom_scheduler"],
                        use_adaptive_tgs=preset["use_adaptive_tgs"],
                        use_anchorflow=preset["use_anchorflow"],
                        use_soft_mask=preset.get("use_soft_mask", False),
                        soft_mask_threshold_scale=preset.get("soft_mask_threshold_scale", 1.5),
                        soft_mask_sharpness=preset.get("soft_mask_sharpness", 12.0),
                        soft_mask_min=preset.get("soft_mask_min", 0.10),
                        use_late_boost=preset.get("use_late_boost", False),
                        late_boost_gain=preset.get("late_boost_gain", 0.30),
                        use_source_velocity=preset.get("use_source_velocity", False),
                        source_velocity_scale=preset.get("source_velocity_scale", 1.0),
                        use_latent_blend=preset.get("use_latent_blend", False),
                        latent_blend_strength=preset.get("latent_blend_strength", 0.9),
                        latent_mask_sharpness=preset.get("latent_mask_sharpness", 18.0),
                        latent_mask_percentile=preset.get("latent_mask_percentile", 0.96),
                        use_local_aux_blend=preset.get("use_local_aux_blend", False),
                        pipeline=pipeline,
                    )
                    success += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    error_type = type(exc).__name__
                    error_message = str(exc).strip()
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "case_id": case_id,
                            "src_dir": record.get("src_dir", ""),
                            "edit_dir": record.get("edit_dir", ""),
                            "guide_pairs": format_guide_pairs(record["guide_pairs"]),
                            "output_dir": str(case_output_dir),
                            "error_type": error_type,
                            "error_message": error_message,
                        }
                    )
                    print(f"[Failed] {dataset_name}/{case_id}: {error_type}: {error_message}")
                    cleanup_after_case_failure()
                    if not args.continue_on_error:
                        raise
    finally:
        write_failure_summary(failed_cases_log, failures)
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Unified batch summary ===")
    print(f"Total cases scanned : {total}")
    print(f"Successful runs     : {success}")
    print(f"Skipped existing    : {skipped_existing}")
    print(f"Skipped missing     : {skipped_missing}")
    print(f"Failed              : {failed}")
    print(f"Outputs root        : {output_root}")
    if failures:
        print(f"Failure log         : {failed_cases_log}")
        print("\n=== Failed case details ===")
        for item in failures:
            print(
                f"- {item['dataset']}/{item['case_id']} | "
                f"guides={item['guide_pairs']} | "
                f"{item['error_type']}: {item['error_message']}"
            )

    if failed > 0:
        if args.continue_on_error:
            print("\n[Batch finished with failures, but continue_on_error=true so exit code is kept as 0.]")
            return
        raise SystemExit(1)


if __name__ == "__main__":
    main(parse_args())
