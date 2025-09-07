#!/usr/bin/env python3
"""
Script to merge overlapping video chunks using linear blending for smooth transitions.
Handles cases with more than 2 overlapping chunks.

Usage:
python merge_overlapping_chunks.py --input_dir results_SG_demo_20240807/ir_overlap32_dr_exp7_ir_720p_ft1_AV_000001750/camera_front_wide_120fov --output_dir merged_output --sample_n_frames 57 --overlap_n_frames 32
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import cv2


def parse_filename(filename):
    """
    Parse filename to extract chunk_id, frame_id, and pass_type.
    Expected format: {chunk_id:04d}.{frame_id:04d}.{pass_type}.jpg

    Returns:
        tuple: (chunk_id, frame_id, pass_type, extension) or None if parsing fails
    """
    pattern = r'^(\d{4})\.(\d{4})\.([^.]+)\.(.+)$'
    match = re.match(pattern, filename)
    if match:
        chunk_id = int(match.group(1))
        frame_id = int(match.group(2))
        pass_type = match.group(3)
        extension = match.group(4)
        return chunk_id, frame_id, pass_type, extension
    return None


def get_global_frame_id(chunk_id, frame_id, sample_n_frames, overlap_n_frames):
    """
    Convert chunk-local frame_id to global frame_id.

    Args:
        chunk_id: Chunk index
        frame_id: Frame index within chunk
        sample_n_frames: Number of frames per chunk (57)
        overlap_n_frames: Number of overlapping frames (32)

    Returns:
        int: Global frame index
    """
    step_size = sample_n_frames - overlap_n_frames  # 57 - 32 = 25
    return chunk_id * step_size + frame_id


def get_chunk_blend_weight(chunk_id, frame_id, sample_n_frames, overlap_n_frames):
    """
    Calculate the raw blend weight for a frame from a specific chunk.
    This weight represents the "strength" of this chunk's contribution at this frame position.

    The weight is:
    - 0.0 at the very beginning and end of overlaps
    - 1.0 in the middle non-overlapping region
    - Linear ramp in between

    Args:
        chunk_id: Chunk index
        frame_id: Frame index within chunk (0 to sample_n_frames-1)
        sample_n_frames: Number of frames per chunk
        overlap_n_frames: Number of overlapping frames

    Returns:
        float: Raw weight for this frame from this chunk
    """
    if frame_id < overlap_n_frames:
        # Beginning of chunk - linear ramp up from 0 to 1
        return frame_id / overlap_n_frames
    elif frame_id >= (sample_n_frames - overlap_n_frames):
        # End of chunk - linear ramp down from 1 to 0
        end_position = frame_id - (sample_n_frames - overlap_n_frames)
        return 1.0 - (end_position / overlap_n_frames)
    else:
        # Middle of chunk - full weight
        return 1.0


def load_image(filepath):
    """Load image and convert to numpy array in float32 [0, 1] range."""
    filepath = Path(filepath)  # Ensure it's a Path object

    if filepath.suffix.lower() == '.exr':
        # Handle EXR files
        img = cv2.imread(str(filepath), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.astype(np.float32)
        else:
            raise ValueError(f"Could not load EXR file: {filepath}")
    else:
        # Handle regular image files
        img = Image.open(filepath)
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array


def save_image(img_array, filepath):
    """Save numpy array as image."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix.lower() == '.exr':
        # Save as EXR
        img_bgr = cv2.cvtColor(img_array.astype(np.float32), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(filepath), img_bgr)
    else:
        # Save as regular image
        img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8)
        img_pil.save(filepath)


def merge_overlapping_chunks(input_dir, output_dir, sample_n_frames=57, overlap_n_frames=16):
    """
    Merge overlapping chunks using linear blending for smooth transitions.
    Handles cases with more than 2 overlapping chunks.

    Args:
        input_dir: Directory containing chunked results
        output_dir: Directory to save merged results
        sample_n_frames: Number of frames per chunk
        overlap_n_frames: Number of overlapping frames between chunks
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise ValueError(f"Input directory {input_dir} does not exist")

    step_size = sample_n_frames - overlap_n_frames
    max_overlaps = (overlap_n_frames + step_size - 1) // step_size + 1
    print(f"Step size: {step_size}, Maximum possible overlaps per frame: {max_overlaps}")

    # Group files by pass type and global frame id
    # Structure: {pass_type: {global_frame_id: [(filepath, chunk_id, frame_id), ...]}}
    frame_groups = defaultdict(lambda: defaultdict(list))

    print("Scanning input files...")

    # Scan all files and group them
    for filepath in input_path.iterdir():
        if filepath.is_file():
            parsed = parse_filename(filepath.name)
            if parsed is None:
                continue

            chunk_id, frame_id, pass_type, extension = parsed
            global_frame_id = get_global_frame_id(chunk_id, frame_id, sample_n_frames, overlap_n_frames)

            frame_groups[pass_type][global_frame_id].append((filepath, chunk_id, frame_id))

    print(f"Found {len(frame_groups)} pass types")

    # Process each pass type
    for pass_type in sorted(frame_groups.keys()):
        print(f"Processing pass type: {pass_type}")

        global_frames = frame_groups[pass_type]
        overlap_stats = defaultdict(int)
        total_frames = len(global_frames)

        for global_frame_id in sorted(global_frames.keys()):
            file_list = global_frames[global_frame_id]
            num_contributors = len(file_list)
            overlap_stats[num_contributors] += 1

            if num_contributors == 1:
                # No overlap, just copy the file
                src_filepath, chunk_id, frame_id = file_list[0]
                dst_filename = f"{global_frame_id:06d}.{pass_type}{src_filepath.suffix}"
                dst_filepath = output_path / dst_filename

                # Load and save (to ensure consistent format)
                img = load_image(src_filepath)
                save_image(img, dst_filepath)

            else:
                # Multiple chunks contribute to this frame, use linear blending
                if overlap_stats[num_contributors] <= 3:  # Only print first few examples
                    print(f"  Blending {num_contributors} chunks for global frame {global_frame_id}")

                # Calculate weighted average using linear blending
                weighted_sum = None
                total_weight = 0.0

                # Calculate raw weights for all contributors
                contributors_info = []
                for src_filepath, chunk_id, frame_id in file_list:
                    img = load_image(src_filepath)
                    raw_weight = get_chunk_blend_weight(chunk_id, frame_id, sample_n_frames, overlap_n_frames)
                    contributors_info.append((img, raw_weight, chunk_id, frame_id))

                # Normalize weights so they sum to 1.0
                total_raw_weight = sum(weight for _, weight, _, _ in contributors_info)

                for img, raw_weight, chunk_id, frame_id in contributors_info:
                    normalized_weight = raw_weight / total_raw_weight if total_raw_weight > 0 else 1.0 / len(
                        contributors_info)

                    if overlap_stats[num_contributors] <= 2:  # Debug info for first few overlaps
                        print(
                            f"    Chunk {chunk_id}, frame {frame_id} -> raw_weight {raw_weight:.3f}, normalized_weight {normalized_weight:.3f}")

                    if weighted_sum is None:
                        weighted_sum = img * normalized_weight
                    else:
                        weighted_sum += img * normalized_weight

                    total_weight += normalized_weight

                # Final result (total_weight should be ~1.0 after normalization)
                blended_img = weighted_sum

                # Save blended result
                src_filepath, _, _ = file_list[0]  # Get extension from first file
                dst_filename = f"{global_frame_id:06d}.{pass_type}{src_filepath.suffix}"
                dst_filepath = output_path / dst_filename
                save_image(blended_img, dst_filepath)

        # Print overlap statistics
        print(f"  Completed {pass_type}: {total_frames} frames")
        for num_overlaps in sorted(overlap_stats.keys()):
            print(f"    {overlap_stats[num_overlaps]} frames with {num_overlaps} overlapping chunks")

    print(f"Linear blending complete! Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge overlapping video chunks using linear blending")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing chunked results")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save merged results")
    parser.add_argument("--sample_n_frames", type=int, default=57,
                        help="Number of frames per chunk (default: 57)")
    parser.add_argument("--overlap_n_frames", type=int, default=16,
                        help="Number of overlapping frames between chunks (default: 16)")

    args = parser.parse_args()

    merge_overlapping_chunks(
        args.input_dir,
        args.output_dir,
        args.sample_n_frames,
        args.overlap_n_frames
    )


if __name__ == "__main__":
    main()