#!/usr/bin/env python3
"""Convert paired ORION-CRC H&E/mIF tiles into HUSE H&E/mIHC splits."""

import argparse
import csv
import math
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from skimage.color import combine_stains, hed_from_rgb, rgb_from_hdx, separate_stains
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MIF_SUFFIXES = {".tif", ".tiff"}

# ORION mIF channel index -> HUSE marker directory. PD-1 (index 13) is excluded.
CHANNEL_TO_MARKER = (
    (0, "Hoechst"),
    (1, "CD31"),
    (2, "CD45"),
    (3, "CD68"),
    (4, "CD4"),
    (5, "FOXP3"),
    (6, "CD8a"),
    (7, "CD45RO"),
    (8, "CD20"),
    (9, "PD-L1"),
    (10, "CD3e"),
    (11, "CD163"),
    (12, "E-cadherin"),
    (14, "Ki67"),
    (15, "Pan-CK"),
    (16, "SMA"),
)


@dataclass(frozen=True)
class ConversionTask:
    he_path: Path
    mif_path: Path
    output_root: Path
    split: str
    alpha: float
    intensity: float
    jpeg_quality: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the paired H&E and 17-channel mIF tiles from "
            "ORIONCRC_dataset_tile_20x into HUSE's 16-marker mIHC layout."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Extracted ORIONCRC_dataset_tile_20x directory containing he/ and if/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root for the generated train/val/test directories.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of paired tiles to convert.",
    )
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--intensity", type=float, default=0.8)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated split directories and manifest.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("Split ratios must be greater than 0.")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    if args.train_ratio == 0:
        raise ValueError("The training ratio must be greater than 0.")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be greater than 0.")
    if args.workers <= 0:
        raise ValueError("--workers must be greater than 0.")
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100.")


def index_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in files:
            raise ValueError(
                f"Duplicate sample stem '{path.stem}' in {directory}: "
                f"{files[path.stem].name}, {path.name}"
            )
        files[path.stem] = path
    return files


def collect_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    he_dir = input_dir / "he"
    mif_dir = input_dir / "if"
    if not he_dir.is_dir() or not mif_dir.is_dir():
        raise FileNotFoundError(
            f"Expected '{he_dir}' and '{mif_dir}' from ORIONCRC_dataset_tile_20x."
        )

    he_files = index_by_stem(he_dir, IMAGE_SUFFIXES)
    mif_files = index_by_stem(mif_dir, MIF_SUFFIXES)
    missing_mif = sorted(set(he_files) - set(mif_files))
    missing_he = sorted(set(mif_files) - set(he_files))
    if missing_mif or missing_he:
        details = []
        if missing_mif:
            details.append(f"{len(missing_mif)} H&E tiles have no mIF pair")
        if missing_he:
            details.append(f"{len(missing_he)} mIF tiles have no H&E pair")
        raise ValueError("; ".join(details))
    if not he_files:
        raise ValueError(f"No paired H&E/mIF tiles found under {input_dir}.")

    return [(he_files[stem], mif_files[stem]) for stem in sorted(he_files)]


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    max_samples: int | None,
) -> dict[str, list[tuple[Path, Path]]]:
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)
    if max_samples is not None:
        shuffled = shuffled[:max_samples]

    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_output_dirs(output_dir: Path) -> None:
    marker_names = [marker for _, marker in CHANNEL_TO_MARKER]
    for split in ("train", "val", "test"):
        (output_dir / split / "he").mkdir(parents=True, exist_ok=True)
        for marker in marker_names:
            (output_dir / split / marker).mkdir(parents=True, exist_ok=True)


def reset_generated_outputs(output_dir: Path, overwrite: bool) -> None:
    generated_paths = [
        output_dir / "train",
        output_dir / "val",
        output_dir / "test",
        output_dir / "split_manifest.csv",
    ]
    existing_paths = [path for path in generated_paths if path.exists()]
    if existing_paths and not overwrite:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"Generated outputs already exist: {existing}. "
            "Choose a new --output-dir or pass --overwrite."
        )

    for path in existing_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def load_mif(path: Path, expected_height: int, expected_width: int) -> np.ndarray:
    mif = tifffile.imread(path)
    if mif.ndim != 3:
        raise ValueError(f"Expected a 3D mIF tensor, got shape {mif.shape}.")

    if mif.shape[-1] == 17:
        channels_last = mif
    elif mif.shape[0] == 17:
        channels_last = np.moveaxis(mif, 0, -1)
    else:
        raise ValueError(f"Expected 17 mIF channels, got shape {mif.shape}.")

    if channels_last.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            "H&E and mIF spatial sizes differ: "
            f"H&E={(expected_height, expected_width)}, mIF={channels_last.shape[:2]}."
        )
    return channels_last


def normalize_channel(channel: np.ndarray) -> np.ndarray:
    channel = channel.astype(np.float32)
    channel_min = float(channel.min())
    channel_range = float(channel.max()) - channel_min
    return (channel - channel_min) / (channel_range + 1e-8)


def process_pair(task: ConversionTask) -> str | None:
    try:
        he_bgr = cv2.imread(str(task.he_path), cv2.IMREAD_COLOR)
        if he_bgr is None:
            raise ValueError("OpenCV could not read the H&E image.")
        he_rgb = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mif = load_mif(task.mif_path, he_rgb.shape[0], he_rgb.shape[1])

        hematoxylin = separate_stains(he_rgb, hed_from_rgb)[:, :, 0]
        hematoxylin_texture = normalize_channel(hematoxylin)

        he_output = task.output_root / task.split / "he" / task.he_path.name
        shutil.copy2(task.he_path, he_output)

        for channel_index, marker_name in CHANNEL_TO_MARKER:
            fluorescence = normalize_channel(mif[:, :, channel_index])
            dab = (
                fluorescence
                * (1.0 + task.alpha * hematoxylin_texture)
                * task.intensity
            )

            stain_concentrations = np.zeros(
                (he_rgb.shape[0], he_rgb.shape[1], 3), dtype=np.float32
            )
            stain_concentrations[:, :, 0] = hematoxylin
            stain_concentrations[:, :, 1] = dab
            ihc_rgb = combine_stains(stain_concentrations, rgb_from_hdx)
            ihc_bgr = cv2.cvtColor(
                (np.clip(ihc_rgb, 0.0, 1.0) * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )

            output_path = task.output_root / task.split / marker_name / task.he_path.name
            options = (
                [cv2.IMWRITE_JPEG_QUALITY, task.jpeg_quality]
                if output_path.suffix.lower() in {".jpg", ".jpeg"}
                else []
            )
            if not cv2.imwrite(str(output_path), ihc_bgr, options):
                raise OSError(f"Failed to write {output_path}.")
        return None
    except Exception as error:
        return f"{task.he_path.name}: {error}"


def write_manifest(
    output_dir: Path, splits: dict[str, list[tuple[Path, Path]]], seed: int
) -> None:
    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("split", "he_file", "mif_file", "seed"))
        for split, pairs in splits.items():
            for he_path, mif_path in pairs:
                writer.writerow((split, he_path.name, mif_path.name, seed))


def main() -> None:
    args = parse_args()
    validate_args(args)
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if input_dir == output_dir:
        raise ValueError("--input-dir and --output-dir must be different.")

    pairs = collect_pairs(input_dir)
    splits = split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    empty_splits = [split for split, split_pairs_list in splits.items() if not split_pairs_list]
    if empty_splits:
        raise ValueError(
            "The selected sample count and ratios produced empty splits: "
            + ", ".join(empty_splits)
        )

    reset_generated_outputs(output_dir, args.overwrite)
    prepare_output_dirs(output_dir)
    write_manifest(output_dir, splits, args.seed)

    tasks = [
        ConversionTask(
            he_path=he_path,
            mif_path=mif_path,
            output_root=output_dir,
            split=split,
            alpha=args.alpha,
            intensity=args.intensity,
            jpeg_quality=args.jpeg_quality,
        )
        for split, split_pairs_list in splits.items()
        for he_path, mif_path in split_pairs_list
    ]

    print(f"Found {len(pairs)} paired tiles; converting {len(tasks)}.")
    print(
        "Split sizes: "
        + ", ".join(
            f"{split}={len(split_pairs_list)}"
            for split, split_pairs_list in splits.items()
        )
    )

    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for error in tqdm(
            executor.map(process_pair, tasks, chunksize=4),
            total=len(tasks),
            desc="Converting ORION-CRC",
        ):
            if error is not None:
                errors.append(error)

    if errors:
        preview = "\n".join(errors[:20])
        raise RuntimeError(
            f"Conversion failed for {len(errors)} tile(s). First failures:\n{preview}"
        )
    print(f"Converted dataset written to {output_dir}.")


if __name__ == "__main__":
    main()
