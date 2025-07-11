#!/usr/bin/env python3
import argparse
import os
import sys
import contextlib
from pathlib import Path
import xml.etree.ElementTree as ET  # Needed for ELAN segmentation parsing

import webvtt
import torch
import numpy as np
import mediapipe as mp
from tqdm import tqdm  # progress bar

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set(p for p_tup in mp_holistic.FACEMESH_CONTOURS for p in p_tup))]

from pose_format import Pose

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from mmpt.models import MMPTModel

# ----------------------------
# Subtitle helper functions
# ----------------------------
def vtt_time_to_seconds(time_str):
    """Convert a VTT time string (HH:MM:SS.mmm) to seconds."""
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds

def read_vtt(vtt_path):
    """
    Parse a VTT file using the webvtt library and return a list of subtitle units.
    Each unit is a dict with keys: 'start', 'end', and 'text'.
    """
    subtitles = []
    for caption in webvtt.read(vtt_path):
        subtitles.append({
            "start": vtt_time_to_seconds(caption.start),
            "end": vtt_time_to_seconds(caption.end),
            # Replace newlines in the caption text with spaces.
            "text": " ".join(caption.text.splitlines())
        })
    return subtitles

# ----------------------------
# End subtitle helpers
# ----------------------------

# Model configurations
model_configs = [
    ("multilingual", "signclip_v1_1/baseline_temporal_inference"),
    ("default", "signclip_bsl/bobsl_islr_finetune_long_context"),
    ("lip", "signclip_bsl/bobsl_islr_lip_long_context"),
    ("lip_only", "signclip_bsl/bobsl_islr_lip_only_long_context"),
]
models = {}

base_dir = Path(__file__).resolve().parent.parent
projects_dir = base_dir / "projects"

for model_name, config_path in model_configs:
    config_file = projects_dir / "retri" / f"{config_path}.yaml"
    model, tokenizer, aligner = MMPTModel.from_pretrained(
        str(config_file),
        video_encoder=None,
    )
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    models[model_name] = {
        "model": model,
        "tokenizer": tokenizer,
        "aligner": aligner,
    }

def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                              p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))
    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))
    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                              p2=("pose_keypoints_2d", "LShoulder"))
    raise ValueError(f"Could not parse normalization info, pose_header.components[0].name is {pose_header.components[0].name}.")

def pose_hide_legs(pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0
        pose.body.data[:, :, points, :] = 0
        return pose
    raise ValueError("Unknown pose header schema for hiding legs")

def preprocess_pose(pose, max_frames=None):
    pose = pose.get_components(
        ["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"],
        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS},
    )
    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)
    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)
    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()  # [1, frame_count, feature_dim]
    if max_frames is not None and pose_frames.size(1) > max_frames:
        print(f"Pose sequence length too long ({pose_frames.size(1)}) longer than {max_frames} frames. Truncating.")
        pose_frames = pose_frames[:, :max_frames, :]
    return pose_frames

def preprocess_text(text, model_name="default"):
    aligner = models[model_name]["aligner"]
    tokenizer = models[model_name]["tokenizer"]
    caps, cmasks = aligner._build_text_seq(
        tokenizer(text, add_special_tokens=False)["input_ids"],
    )
    caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1
    return caps, cmasks

def embed_pose(pose, model_name='default', lip_segments=None):
    """
    Embed a pose segment. If lip_segments is provided (a list of numpy arrays matching the pose segments),
    then for each segment, the lip features (converted to a torch tensor) are concatenated to the output
    of preprocess_pose along the feature dimension.
    """
    model = models[model_name]['model']
    caps, cmasks = preprocess_text('', model_name)
    poses = pose if type(pose) == list else [pose]
    pose_frames_l = []
    for i, p in enumerate(poses):
        pose_frames = preprocess_pose(p)
        if lip_segments is not None:
            # Get the corresponding lip segment and convert it to a tensor.
            lip_seg = lip_segments[i]  # Expected shape: (frame_count, lip_feature_dim)
            lip_tensor = torch.from_numpy(lip_seg).unsqueeze(0).float()  # Shape: [1, frame_count, lip_feature_dim]
            if pose_frames.shape[1] != lip_tensor.shape[1]:
                print(f"Warning: mismatched frame counts: pose_frames {pose_frames.shape[1]} vs lip_tensor {lip_tensor.shape[1]}")
            # Concatenate along the feature dimension (dim=2)
            pose_frames = torch.cat([pose_frames, lip_tensor], dim=2)
        pose_frames_l.append(pose_frames)
    pose_frames_l = torch.cat(pose_frames_l)
    batch_size = pose_frames_l.shape[0]
    with torch.no_grad():
        output = model(pose_frames_l, caps.repeat(batch_size, 1), cmasks.repeat(batch_size, 1), return_score=False)
        embeddings = output['pooled_video'].cpu().numpy()
    return embeddings

def embed_text(text, model_name='default'):
    model = models[model_name]['model']
    
    # Determine the placeholder dimension based on the model_name.
    if model_name == 'lip':
        placeholder_dim = 1377
    elif model_name == 'lip_only':
        placeholder_dim = 768
    else:
        placeholder_dim = 609

    # Ensure texts is a list.
    texts = text if isinstance(text, list) else [text]
    batch_size = len(texts)

    # Preprocess each text individually and store the results.
    caps_list = []
    cmasks_list = []
    for t in texts:
        caps, cmasks = preprocess_text(t, model_name)
        caps_list.append(caps)   # Each should have shape (1, 128)
        cmasks_list.append(cmasks)

    # Concatenate the individual results along the batch dimension.
    # The resulting shapes will be (batch_size, 128)
    caps_batch = torch.cat(caps_list, dim=0)
    cmasks_batch = torch.cat(cmasks_list, dim=0)

    # Create dummy pose_frames with shape (batch_size, 1, placeholder_dim).
    pose_frames = torch.randn(batch_size, 1, placeholder_dim)

    # Run the model forward pass only once with the full batch.
    with torch.no_grad():
        output = model(pose_frames, caps_batch, cmasks_batch, return_score=False)
    
    # Extract the pooled text embeddings and return as a NumPy array.
    embeddings = output['pooled_text'].cpu().numpy()
    return embeddings

def embed_lip(lip_segments, model_name='lip_only'):
    """
    Embed a list of lip segments using the lip_only model.
    Each lip segment is a numpy array with shape [frame_count, lip_feature_dim].
    """
    model = models[model_name]['model']
    # Use an empty text input as before.
    caps, cmasks = preprocess_text('', model_name)
    lip_tensor_list = []
    for lip_seg in lip_segments:
        # Convert the lip segment to a tensor of shape [1, frame_count, lip_feature_dim]
        lip_tensor = torch.from_numpy(lip_seg).unsqueeze(0).float()
        lip_tensor_list.append(lip_tensor)
    lip_tensor_batch = torch.cat(lip_tensor_list, dim=0)
    batch_size = lip_tensor_batch.shape[0]
    with torch.no_grad():
        output = model(lip_tensor_batch, caps.repeat(batch_size, 1), cmasks.repeat(batch_size, 1), return_score=False)
        embeddings = output['pooled_video'].cpu().numpy()
    return embeddings

def process_batch(batch, vid, batch_index, save_dir, window_size, model_name, lip_segments=None):
    """
    Process a batch of pose segments (and optionally corresponding lip segments).
    
    When model_name is "lip_only", ignore the pose input and process only lip_segments.
    For segments with frame count ≤ window_size, pad with zeros to reach exactly window_size frames.
    For segments with frame count > window_size, process them individually.
    The embeddings are then reassembled in the original order.
    """
    if model_name == "lip_only":
        if lip_segments is None:
            raise ValueError("lip_segments must be provided for lip_only mode")
        normal_indices = []
        long_indices = []
        normal_lip_segments = []
        long_lip_segments = []
        # Partition lip segments based on length.
        for idx, lip_seg in enumerate(lip_segments):
            current_length = lip_seg.shape[0]
            if current_length > window_size:
                long_indices.append(idx)
                long_lip_segments.append(lip_seg)
            else:
                normal_indices.append(idx)
                normal_lip_segments.append(lip_seg)
        # Pad normal lip segments if needed.
        for i, lip_seg in enumerate(normal_lip_segments):
            current_length = lip_seg.shape[0]
            if current_length < window_size:
                missing_frames = window_size - current_length
                pad_lip = np.zeros((missing_frames, lip_seg.shape[1]), dtype=lip_seg.dtype)
                normal_lip_segments[i] = np.concatenate([lip_seg, pad_lip], axis=0)
        # Process the normal lip segments in batch.
        if normal_lip_segments:
            embeddings_normal = embed_lip(normal_lip_segments, model_name=model_name)
        else:
            embeddings_normal = None
        # Process long lip segments one by one.
        embeddings_long_list = []
        for i, seg in enumerate(long_lip_segments):
            emb = embed_lip([seg], model_name=model_name)
            embeddings_long_list.append(emb[0])
        total = len(lip_segments)
        result = [None] * total
        if embeddings_normal is not None:
            for i, idx in enumerate(normal_indices):
                result[idx] = embeddings_normal[i]
        if embeddings_long_list:
            for i, idx in enumerate(long_indices):
                result[idx] = embeddings_long_list[i]
        result = np.stack(result, axis=0)
        return result

    else:
        # Existing logic for "default" and "lip" modes.
        normal_segments = []
        normal_indices = []
        long_segments = []
        long_indices = []
        normal_lip_segments = []  # List for corresponding lip segments.
        long_lip_segments = []

        for idx, pose_segment in enumerate(batch):
            current_length = pose_segment.body.data.shape[0]
            if current_length > window_size:
                print(f"Warning: Segment at batch index {idx} from video {vid} has length {current_length} exceeding window_size {window_size}. Processing separately.")
                long_segments.append(pose_segment)
                long_indices.append(idx)
                if lip_segments is not None:
                    long_lip_segments.append(lip_segments[idx])
            else:
                normal_segments.append(pose_segment)
                normal_indices.append(idx)
                if lip_segments is not None:
                    normal_lip_segments.append(lip_segments[idx])
        
        # Pad pose segments for normal segments.
        for seg in normal_segments:
            current_length = seg.body.data.shape[0]
            if current_length < window_size:
                missing_frames = window_size - current_length
                pad_data = np.zeros((missing_frames, *seg.body.data.shape[1:]), dtype=seg.body.data.dtype)
                seg.body.data = np.concatenate([seg.body.data, pad_data], axis=0)
                pad_conf = np.zeros((missing_frames, *seg.body.confidence.shape[1:]), dtype=seg.body.confidence.dtype)
                seg.body.confidence = np.concatenate([seg.body.confidence, pad_conf], axis=0)
        
        # Pad the corresponding lip segments for normal segments.
        if lip_segments is not None:
            for i, lip_seg in enumerate(normal_lip_segments):
                current_length = lip_seg.shape[0]
                if current_length < window_size:
                    missing_frames = window_size - current_length
                    pad_lip = np.zeros((missing_frames, lip_seg.shape[1]), dtype=lip_seg.dtype)
                    normal_lip_segments[i] = np.concatenate([lip_seg, pad_lip], axis=0)
        
        if normal_segments:
            embeddings_normal = embed_pose(normal_segments, model_name=model_name,
                                           lip_segments=normal_lip_segments if lip_segments is not None else None)
        else:
            embeddings_normal = None

        embeddings_long_list = []
        for i, seg in enumerate(long_segments):
            emb = embed_pose([seg], model_name=model_name,
                             lip_segments=[long_lip_segments[i]] if lip_segments is not None else None)
            embeddings_long_list.append(emb[0])
        
        total = len(batch)
        result = [None] * total
        if embeddings_normal is not None:
            for i, idx in enumerate(normal_indices):
                result[idx] = embeddings_normal[i]
        if embeddings_long_list:
            for i, idx in enumerate(long_indices):
                result[idx] = embeddings_long_list[i]
        result = np.stack(result, axis=0)
        return result

def get_sign_segments(segmentation_file, video_id):
    """Parse an ELAN (.eaf) file and return all segments from the SIGN tier."""
    segments = []
    try:
        tree = ET.parse(segmentation_file)
        root = tree.getroot()
    except Exception as e:
        return segments
    time_order = root.find("TIME_ORDER")
    time_slots = {}
    if time_order is not None:
        for ts in time_order.findall("TIME_SLOT"):
            ts_id = ts.get("TIME_SLOT_ID")
            ts_value = ts.get("TIME_VALUE")
            if ts_value is not None:
                try:
                    time_slots[ts_id] = float(ts_value) / 1000.0
                except ValueError:
                    time_slots[ts_id] = None
    else:
        return segments
    sign_tier = None
    for tier in root.findall("TIER"):
        if tier.get("TIER_ID") == "SIGN":
            sign_tier = tier
            break
    if sign_tier is None:
        return segments
    for annotation in sign_tier.findall("ANNOTATION"):
        annotation_elem = None
        for child in annotation:
            annotation_elem = child
            break
        if annotation_elem is None:
            continue
        text_elem = annotation_elem.find("ANNOTATION_VALUE")
        text = text_elem.text if text_elem is not None else ""
        start_time = None
        end_time = None
        if "TIME_SLOT_REF1" in annotation_elem.attrib and "TIME_SLOT_REF2" in annotation_elem.attrib:
            ts1 = annotation_elem.attrib["TIME_SLOT_REF1"]
            ts2 = annotation_elem.attrib["TIME_SLOT_REF2"]
            start_time = time_slots.get(ts1, None)
            end_time = time_slots.get(ts2, None)
        if start_time is not None and end_time is not None:
            mid = (start_time + end_time) / 2
        else:
            mid = None
        segments.append({'start': start_time, 'end': end_time, 'mid': mid, 'text': text})
    return segments

def main():
    parser = argparse.ArgumentParser(description="Extract video embeddings using SignCLIP.")
    parser.add_argument(
        "--video_ids",
        type=str,
        default="/users/zifan/subtitle_align/data/bobsl_align.txt",
        help="Path to text file containing video ids (one per line)."
    )
    parser.add_argument(
        "--pose_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe_v2_refine_face_complexity_2",
        help="Directory where pose files are stored."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/sign_clip_bobsl",
        help="Directory to store SignCLIP embedding results."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Stride for sliding window."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=32,
        help="Window size for sliding window (and target frame count for segments)."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing windows."
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing feature files if set"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "segmentation", "subtitle"],
        help="Processing mode: sliding_window (default), segmentation, or subtitle."
    )
    parser.add_argument(
        "--segmentation_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/segmentation",
        help="Directory with segmentation ELAN (.eaf) files."
    )
    parser.add_argument(
        "--subtitle_dir",
        type=str,
        default="/users/zifan/BOBSL/v1.4/automatic_annotations/signing_aligned_subtitles/audio_aligned_heuristic_correction",
        help="Directory where subtitle (VTT) files are stored."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="Frames per second for converting segmentation times to frame indices."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="default",
        choices=["default", "lip", "lip_only", "multilingual"],
        help="Model name to use ('default', 'lip', or 'lip_only')."
    )
    parser.add_argument(
        "--lip_feat_dir",
        type=str,
        default="/scratch/shared/beegfs/zifan/bobsl/video_features/auto_avsr",
        help="Directory where lip feature npy files are stored (used when model_name is 'lip' or 'lip_only')."
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Determine the list of video ids.
    if args.video_ids == "all":
        if not os.path.exists(args.pose_dir):
            print(f"Error: Pose directory not found: {args.pose_dir}")
            return
        video_ids = [f[:-5] for f in os.listdir(args.pose_dir) if f.endswith(".pose")]
        if not video_ids:
            print(f"No pose files found in the pose directory: {args.pose_dir}")
            return
    else:
        if not os.path.exists(args.video_ids):
            print(f"Error: Video IDs file not found: {args.video_ids}")
            return
        with open(args.video_ids, "r") as f:
            video_ids = [line.strip() for line in f if line.strip()]

    # Print the number of videos to be processed.
    print(f"Processing {len(video_ids)} videos")

    total_pose_files_found = 0

    # video_ids.reverse()
    for vid in video_ids:
        save_path = os.path.join(args.save_dir, f"{vid}.npy")
        if os.path.exists(save_path) and not args.overwrite:
            print(f"Embeddings file already exists for video {vid} at {save_path}. Skipping.")
            continue

        if args.mode == "sliding_window":
            pose_path = os.path.join(args.pose_dir, f"{vid}.pose")
            if not os.path.exists(pose_path):
                print(f"Pose file not found for video id: {vid}")
                continue
            total_pose_files_found += 1
            with open(pose_path, "rb") as f:
                buffer = f.read()
            num_frames = Pose.read(buffer).body.data.shape[0]
            print(f"Video id {vid} has {num_frames} frames.")

            # For "lip" and "lip_only", load the corresponding lip features.
            if args.model_name in ["lip", "lip_only"]:
                lip_feat_path = os.path.join(args.lip_feat_dir, f"{vid}.npy")
                if not os.path.exists(lip_feat_path):
                    print(f"Lip features file not found for video id: {vid} at {lip_feat_path}")
                    continue
                lip_features = np.load(lip_feat_path)
            else:
                lip_features = None

            video_embedding_batches = []
            batch = []
            lip_batch_current = [] if args.model_name in ["lip", "lip_only"] else None
            batch_index = 0
            total_windows = (num_frames - args.window_size) // args.stride + 1
            for start_frame in tqdm(range(0, num_frames - args.window_size + 1, args.stride),
                                    total=total_windows,
                                    desc=f"Processing video {vid}"):
                pose_window = Pose.read(buffer, start_frame=start_frame, end_frame=start_frame+args.window_size)
                if args.model_name in ["lip", "lip_only"]:
                    lip_window = lip_features[start_frame:start_frame+args.window_size]
                    lip_batch_current.append(lip_window)
                batch.append(pose_window)
                if len(batch) == args.batch_size:
                    embeddings = process_batch(
                        batch, vid, batch_index, args.save_dir, args.window_size, args.model_name,
                        lip_segments=lip_batch_current if args.model_name in ["lip", "lip_only"] else None
                    )
                    video_embedding_batches.append(embeddings)
                    batch_index += 1
                    batch = []
                    if args.model_name in ["lip", "lip_only"]:
                        lip_batch_current = []
            if batch:
                embeddings = process_batch(
                    batch, vid, batch_index, args.save_dir, args.window_size, args.model_name,
                    lip_segments=lip_batch_current if args.model_name in ["lip", "lip_only"] else None
                )
                video_embedding_batches.append(embeddings)
            if video_embedding_batches:
                final_embeddings = np.concatenate(video_embedding_batches, axis=0)
                print(final_embeddings.shape)
                np.save(save_path, final_embeddings)
                print(f"Saved embeddings for video {vid} at {save_path} with shape {final_embeddings.shape}")
            else:
                print(f"No embeddings generated for video {vid}.")

        elif args.mode == "segmentation":
            pose_path = os.path.join(args.pose_dir, f"{vid}.pose")
            if not os.path.exists(pose_path):
                print(f"Pose file not found for video id: {vid}")
                continue
            total_pose_files_found += 1
            with open(pose_path, "rb") as f:
                buffer = f.read()
            num_frames = Pose.read(buffer).body.data.shape[0]
            print(f"Video id {vid} has {num_frames} frames.")
            segmentation_path = os.path.join(args.segmentation_dir, f"{vid}.eaf")
            if not os.path.exists(segmentation_path):
                print(f"Segmentation file not found for video id: {vid}")
                continue
            segments = get_sign_segments(segmentation_path, vid)
            if not segments:
                print(f"No segments found in segmentation file for video id: {vid}")
                continue
            valid_segments = [
                seg for seg in segments 
                if seg['start'] is not None and seg['end'] is not None and seg['end'] > seg['start']
            ]
            if valid_segments:
                lengths = [int((seg['end'] - seg['start']) * args.fps) for seg in valid_segments]
                count = len(lengths)
                mean_length = np.mean(lengths)
                min_length = np.min(lengths)
                max_length = np.max(lengths)
                print(f"Segment statistics for video {vid}: count = {count}, mean length = {mean_length:.2f} frames, min length = {min_length} frames, max length = {max_length} frames.")
            else:
                print(f"No valid segments found in segmentation file for video id: {vid}")
                continue

            if args.model_name in ["lip", "lip_only"]:
                lip_feat_path = os.path.join(args.lip_feat_dir, f"{vid}.npy")
                if not os.path.exists(lip_feat_path):
                    print(f"Lip features file not found for video id: {vid} at {lip_feat_path}")
                    continue
                lip_features = np.load(lip_feat_path)
            else:
                lip_features = None

            video_embedding_batches = []
            batch = []
            lip_batch_current = [] if args.model_name in ["lip", "lip_only"] else None
            batch_index = 0
            for segment in tqdm(valid_segments, desc=f"Processing segments for video {vid}"):
                start_frame = int(segment['start'] * args.fps)
                end_frame = int(segment['end'] * args.fps)
                start_frame = max(0, start_frame)
                end_frame = min(num_frames, end_frame)
                if end_frame <= start_frame:
                    end_frame = start_frame + 1
                if args.model_name in ["lip", "lip_only"]:
                    lip_segment = lip_features[start_frame:end_frame]
                    lip_batch_current.append(lip_segment)
                pose_segment = Pose.read(buffer, start_frame=start_frame, end_frame=end_frame)
                batch.append(pose_segment)
                if len(batch) == args.batch_size:
                    embeddings = process_batch(
                        batch, vid, batch_index, args.save_dir, args.window_size, args.model_name,
                        lip_segments=lip_batch_current if args.model_name in ["lip", "lip_only"] else None
                    )
                    assert embeddings.shape[0] == len(batch), f"Mismatch: got {embeddings.shape[0]} embeddings, expected {len(batch)} for video {vid}, batch {batch_index}"
                    video_embedding_batches.append(embeddings)
                    batch_index += 1
                    batch = []
                    if args.model_name in ["lip", "lip_only"]:
                        lip_batch_current = []
            if batch:
                embeddings = process_batch(
                    batch, vid, batch_index, args.save_dir, args.window_size, args.model_name,
                    lip_segments=lip_batch_current if args.model_name in ["lip", "lip_only"] else None
                )
                assert embeddings.shape[0] == len(batch), f"Mismatch: got {embeddings.shape[0]} embeddings, expected {len(batch)} for video {vid}, final batch"
                video_embedding_batches.append(embeddings)
            if video_embedding_batches:
                final_embeddings = np.concatenate(video_embedding_batches, axis=0)
                print(final_embeddings.shape)
                np.save(save_path, final_embeddings)
                print(f"Saved embeddings for video {vid} at {save_path} with shape {final_embeddings.shape}")
            else:
                print(f"No embeddings generated for video {vid}.")

        elif args.mode == "subtitle":
            subtitle_path = os.path.join(args.subtitle_dir, f"{vid}.vtt")
            if not os.path.exists(subtitle_path):
                print(f"Subtitle file not found for video id: {vid}")
                continue
            subtitles = read_vtt(subtitle_path)
            if not subtitles:
                print(f"No subtitles found for video id: {vid}")
                continue
            print(f"Found {len(subtitles)} subtitle units for video {vid}.")
            subtitle_texts = [sub["text"] for sub in subtitles]
            subtitle_embedding_batches = []
            for i in range(0, len(subtitle_texts), args.batch_size):
                batch_texts = subtitle_texts[i:i+args.batch_size]
                batch_embeddings = []
                for text in batch_texts:
                    text_prompt = f"<en> <bfi> {text}"
                    emb = embed_text(text_prompt, model_name=args.model_name)
                    batch_embeddings.append(emb[0])
                batch_embeddings = np.stack(batch_embeddings, axis=0)
                subtitle_embedding_batches.append(batch_embeddings)
            if subtitle_embedding_batches:
                final_embeddings = np.concatenate(subtitle_embedding_batches, axis=0)
                np.save(save_path, final_embeddings)
                print(f"Saved subtitle embeddings for video {vid} at {save_path} with shape {final_embeddings.shape}")
            else:
                print(f"No embeddings generated for video {vid}.")

    print(f"Found and processed {total_pose_files_found} pose files out of {len(video_ids)} video ids.")


# -----------------------------------------------------------------------------
# New functions for external use
# -----------------------------------------------------------------------------

# Default values taken from CLI argument defaults
_DEFAULT_POSE_DIR = "/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe_v2_refine_face_complexity_2"
_DEFAULT_LIP_FEAT_DIR = "/scratch/shared/beegfs/zifan/bobsl/video_features/auto_avsr"
_DEFAULT_WINDOW_SIZE = 32
_DEFAULT_BATCH_SIZE = 1024
_DEFAULT_FPS = 25

# Context manager to suppress output (stdout and stderr)
@contextlib.contextmanager
def suppress_fairseq_output():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def live_embed_subtitles(cues, model_name="default", batch_size=_DEFAULT_BATCH_SIZE, tokenize_text_embedding=False):
    """
    Compute subtitle embeddings for a list of cues.
    
    Each cue is expected to be a dict with a "text" key.
    Returns a tuple:
      - First element: a NumPy array where each row is the embedding for a cue.
      - Second element: if tokenize_text_embedding is True, a list where each element is a NumPy array 
        of token embeddings (obtained by splitting on space) for that cue; otherwise, None.
    """
    texts = [cue["text"] for cue in cues]
    # Compute cue-level embeddings.
    embedding_batches = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding subtitles"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = []
        for text in batch_texts:
            with suppress_fairseq_output():
                text_prompt = f"<en> <bfi> {text}"
                emb = embed_text(text_prompt, model_name=model_name)
            # emb is expected to have shape [1, embedding_dim]; take the first row.
            batch_embeddings.append(emb[0])
        batch_embeddings = np.stack(batch_embeddings, axis=0)
        embedding_batches.append(batch_embeddings)
    if embedding_batches:
        cue_level_embeddings = np.concatenate(embedding_batches, axis=0)
    else:
        cue_level_embeddings = np.array([])

    token_level_embeddings = None
    if tokenize_text_embedding:
        # Tokenize each cue on space.
        token_lists = [text.split() for text in texts]
        # Record the number of tokens per cue.
        token_counts = [len(tokens) for tokens in token_lists]
        # Flatten all tokens.
        all_tokens = [token for tokens in token_lists for token in tokens]
        
        # Process token embeddings in batches.
        token_embedding_batches = []
        for i in tqdm(range(0, len(all_tokens), batch_size), desc="Embedding tokens"):
            batch_tokens = all_tokens[i:i+batch_size]
            batch_token_embeddings = []
            for token in batch_tokens:
                with suppress_fairseq_output():
                    text_prompt = f"<en> <bfi> {token}"
                    emb = embed_text(text_prompt, model_name=model_name)
                batch_token_embeddings.append(emb[0])
            batch_token_embeddings = np.stack(batch_token_embeddings, axis=0)
            token_embedding_batches.append(batch_token_embeddings)
        if token_embedding_batches:
            all_token_embeddings = np.concatenate(token_embedding_batches, axis=0)
        else:
            all_token_embeddings = np.array([])
        
        # Regroup the token embeddings so that the output list has one entry per cue.
        token_level_embeddings = []
        idx = 0
        for count in token_counts:
            # Each cue's token embeddings will be returned as a NumPy array.
            token_level_embeddings.append(all_token_embeddings[idx: idx+count])
            idx += count

    return cue_level_embeddings, token_level_embeddings

def live_embed_signs(signs, video_id, model_name="default", batch_size=_DEFAULT_BATCH_SIZE*2,
                     window_size=_DEFAULT_WINDOW_SIZE, fps=_DEFAULT_FPS):
    """
    Compute sign (pose) embeddings for a list of sign segments for the specified video_id.
    
    Each sign in 'signs' should be a dict with 'start' and 'end' times (in seconds).
    The function reads the episode-level pose from the default pose directory and, if needed,
    the corresponding lip features. It then extracts the pose segments corresponding to each sign,
    processes them in batches, and returns a NumPy array where each row is a sign embedding.
    """
    pose_path = os.path.join(_DEFAULT_POSE_DIR, f"{video_id}.pose")
    if not os.path.exists(pose_path):
        raise FileNotFoundError(f"Pose file not found for video_id: {video_id} at {pose_path}")
    with open(pose_path, "rb") as f:
        buffer = f.read()
    pose_obj = Pose.read(buffer)
    num_frames = pose_obj.body.data.shape[0]
    
    if model_name in ["lip", "lip_only"]:
        lip_feat_path = os.path.join(_DEFAULT_LIP_FEAT_DIR, f"{video_id}.npy")
        if not os.path.exists(lip_feat_path):
            raise FileNotFoundError(f"Lip features file not found for video_id: {video_id} at {lip_feat_path}")
        lip_features = np.load(lip_feat_path)
    else:
        lip_features = None

    sign_pose_segments = []
    sign_lip_segments = [] if model_name in ["lip", "lip_only"] else None
    # Extract pose segments for each valid sign.
    for sign in tqdm(signs, desc="Extracting sign segments", total=len(signs)):
        start_time = sign.get('start')
        end_time = sign.get('end')
        if start_time is None or end_time is None or end_time <= start_time:
            continue
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        pose_segment = Pose.read(buffer, start_frame=start_frame, end_frame=end_frame)
        sign_pose_segments.append(pose_segment)
        if model_name in ["lip", "lip_only"]:
            lip_segment = lip_features[start_frame:end_frame]
            sign_lip_segments.append(lip_segment)
    
    video_embedding_batches = []
    batch = []
    lip_batch_current = [] if model_name in ["lip", "lip_only"] else None
    batch_index = 0
    # Process the sign segments in batches, tracking progress.
    for i, seg in enumerate(tqdm(sign_pose_segments, desc="Embedding sign segments", total=len(sign_pose_segments))):
        batch.append(seg)
        if model_name in ["lip", "lip_only"]:
            lip_batch_current.append(sign_lip_segments[i])
        if len(batch) == batch_size:
            with suppress_fairseq_output():
                embeddings = process_batch(
                    batch, video_id, batch_index, "", window_size, model_name,
                    lip_segments=lip_batch_current if model_name in ["lip", "lip_only"] else None
                )
            video_embedding_batches.append(embeddings)
            batch_index += 1
            batch = []
            if model_name in ["lip", "lip_only"]:
                lip_batch_current = []
    if batch:
        with suppress_fairseq_output():
            embeddings = process_batch(
                batch, video_id, batch_index, "", window_size, model_name,
                lip_segments=lip_batch_current if model_name in ["lip", "lip_only"] else None
            )
        video_embedding_batches.append(embeddings)
    if video_embedding_batches:
        final_embeddings = np.concatenate(video_embedding_batches, axis=0)
    else:
        final_embeddings = np.array([])
    return final_embeddings

__all__ = ['live_embed_subtitles', 'live_embed_signs']

if __name__ == "__main__":
    main()
