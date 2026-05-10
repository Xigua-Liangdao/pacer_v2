#!/usr/bin/env python3

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


LABEL_ID_TO_NAME = {
    0: "normal",
    1: "talking",
    2: "yawning",
}

RAW_VIDEO_LABELS = ("normal", "talking", "yawning", "talking_yawning")
DEFAULT_SAMPLE_STEP = 2
DEFAULT_IMAGE_SIZE = 224
DEFAULT_TRACK_FRAMES = 6


@dataclass(frozen=True)
class ExtractionTask:
    video_path: Path
    output_dir: Path
    label: str
    raw_video_label: str
    start_sec: float = 0.0
    end_sec: Optional[float] = None
    split_file: Optional[Path] = None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    dataset_root = project_root / "YawDD_dataset" / "Mirror"

    parser = argparse.ArgumentParser(
        description=(
            "Extract 224x224 face crops for YawDD Mirror Normal, Talking and "
            "Talking&Yawning videos into per-video folders."
        )
    )
    parser.add_argument(
        "--mirror-root",
        type=Path,
        default=dataset_root,
        help="Root directory that contains Female_mirror and Male_mirror Avi Videos.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "extracted_face" / "train_face_image",
        help="Root directory where per-video/per-clip folders will be created.",
    )
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=None,
        help="Optional train split output root. When set with --test-output-root, videos are routed by subject id.",
    )
    parser.add_argument(
        "--test-output-root",
        type=Path,
        default=None,
        help="Optional test split output root. When set with --train-output-root, videos are routed by subject id.",
    )
    parser.add_argument(
        "--test-subjects",
        type=str,
        default="",
        help="Comma separated subject ids assigned to the test split when using train/test output roots.",
    )
    parser.add_argument(
        "--detector",
        choices=["auto", "mediapipe", "facenet-pytorch", "haar"],
        default="auto",
        help="Face detector backend. auto prefers mediapipe, then facenet-pytorch, then Haar.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=DEFAULT_SAMPLE_STEP,
        help="Keep one frame every N frames.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square output size for saved faces.",
    )
    parser.add_argument(
        "--bbox-margin",
        type=float,
        default=0.2,
        help="Margin ratio added around the detected face before square cropping.",
    )
    parser.add_argument(
        "--track-frames",
        type=int,
        default=DEFAULT_TRACK_FRAMES,
        help="Reuse the last detected bbox for at most this many sampled frames.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N planned folders. 0 means no limit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip an output folder when it already exists and contains files.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing output folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned folders and counts, without extracting faces.",
    )
    return parser.parse_args()


def normalize_video_name(name: str) -> str:
    return name.strip().lower().replace("&", "")


def is_talking_yawning_name(name: str) -> bool:
    lowered = normalize_video_name(name)
    return "talkingyawning" in lowered


def resolve_raw_video_label(video_name: str) -> Optional[str]:
    lowered = normalize_video_name(video_name)
    if is_talking_yawning_name(lowered):
        return "talking_yawning"
    if "-yawning" in lowered:
        return "yawning"
    if "-talking" in lowered:
        return "talking"
    if "-normal" in lowered:
        return "normal"
    return None


def iter_mirror_videos(mirror_root: Path) -> Iterable[Path]:
    female_dir = mirror_root / "Female_mirror"
    male_dir = mirror_root / "Male_mirror Avi Videos"
    for directory in (female_dir, male_dir):
        if not directory.exists():
            continue
        for video_path in sorted(directory.glob("*.avi")):
            yield video_path


def build_output_dir(output_root: Path, video_path: Path) -> Path:
    return output_root / video_path.stem


def parse_subject_id_from_name(name: str) -> str:
    return name.split("-", 1)[0].strip()


def parse_subject_list(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def resolve_split_output_root(
    video_path: Path,
    output_root: Path,
    train_output_root: Optional[Path],
    test_output_root: Optional[Path],
    test_subjects: Sequence[str],
) -> Tuple[Path, Optional[str]]:
    if train_output_root is None or test_output_root is None:
        return output_root, None
    subject_id = parse_subject_id_from_name(video_path.stem)
    split_name = "test" if subject_id in set(test_subjects) else "train"
    return (test_output_root if split_name == "test" else train_output_root), split_name


def plan_tasks(
    mirror_root: Path,
    output_root: Path,
    skip_existing: bool,
    train_output_root: Optional[Path] = None,
    test_output_root: Optional[Path] = None,
    test_subjects: Optional[Sequence[str]] = None,
) -> Tuple[List[ExtractionTask], List[Tuple[Path, str]]]:
    tasks: List[ExtractionTask] = []
    skipped: List[Tuple[Path, str]] = []
    test_subjects = list(test_subjects or [])

    for video_path in iter_mirror_videos(mirror_root):
        raw_label = resolve_raw_video_label(video_path.name)
        if raw_label not in RAW_VIDEO_LABELS:
            continue

        split_output_root, split_name = resolve_split_output_root(
            video_path,
            output_root,
            train_output_root,
            test_output_root,
            test_subjects,
        )
        output_dir = build_output_dir(split_output_root, video_path)
        if skip_existing and output_dir.exists() and any(output_dir.iterdir()):
            skipped.append((output_dir, "existing_output"))
            continue
        tasks.append(
            ExtractionTask(
                video_path=video_path,
                output_dir=output_dir,
                label=raw_label,
                raw_video_label=raw_label,
                split_file=None if split_name is None else Path(split_name),
            )
        )

    return tasks, skipped


class BaseFaceDetector:
    name = "base"

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        raise NotImplementedError


class HaarFaceDetector(BaseFaceDetector):
    name = "haar"

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return float(x), float(y), float(x + w), float(y + h)


class MediaPipeFaceDetector(BaseFaceDetector):
    name = "mediapipe"

    def __init__(self) -> None:
        import mediapipe as mp

        self._mp = mp
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)
        if not results.detections:
            return None
        height, width = frame_bgr.shape[:2]
        best_bbox: Optional[Tuple[float, float, float, float]] = None
        best_area = -1.0
        for detection in results.detections:
            rel_bbox = detection.location_data.relative_bounding_box
            x1 = rel_bbox.xmin * width
            y1 = rel_bbox.ymin * height
            x2 = (rel_bbox.xmin + rel_bbox.width) * width
            y2 = (rel_bbox.ymin + rel_bbox.height) * height
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best_bbox = (x1, y1, x2, y2)
        return best_bbox


class FacenetMTCNNDetector(BaseFaceDetector):
    name = "facenet-pytorch"

    def __init__(self) -> None:
        from PIL import Image
        import torch
        from facenet_pytorch import MTCNN

        self._image_cls = Image
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.detector = MTCNN(keep_all=True, device=device)

    def detect(self, frame_bgr: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._image_cls.fromarray(frame_rgb)
        boxes, probs = self.detector.detect(image)
        if boxes is None or len(boxes) == 0:
            return None
        best_index = 0
        best_score = -1.0
        for index, box in enumerate(boxes):
            prob = 0.0 if probs is None else float(probs[index])
            area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
            score = prob * area
            if score > best_score:
                best_score = score
                best_index = index
        box = boxes[best_index]
        return float(box[0]), float(box[1]), float(box[2]), float(box[3])


def create_detector(name: str) -> BaseFaceDetector:
    if name == "mediapipe":
        return MediaPipeFaceDetector()
    if name == "facenet-pytorch":
        return FacenetMTCNNDetector()
    if name == "haar":
        return HaarFaceDetector()

    for factory in (MediaPipeFaceDetector, FacenetMTCNNDetector, HaarFaceDetector):
        try:
            return factory()
        except Exception:
            continue
    raise RuntimeError(
        "No usable detector backend is available. Install mediapipe or facenet-pytorch, or use OpenCV Haar fallback."
    )


def clamp_bbox_to_image(
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int, int],
    margin_ratio: float,
) -> Optional[Tuple[int, int, int, int]]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox
    face_width = max(1.0, x2 - x1)
    face_height = max(1.0, y2 - y1)
    size = max(face_width, face_height) * (1.0 + 2.0 * margin_ratio)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    crop_x1 = int(round(center_x - size / 2.0))
    crop_y1 = int(round(center_y - size / 2.0))
    crop_x2 = int(round(center_x + size / 2.0))
    crop_y2 = int(round(center_y + size / 2.0))

    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(width, crop_x2)
    crop_y2 = min(height, crop_y2)
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return crop_x1, crop_y1, crop_x2, crop_y2


def crop_face(
    frame_bgr: np.ndarray,
    bbox: Tuple[float, float, float, float],
    margin_ratio: float,
    image_size: int,
) -> Optional[np.ndarray]:
    clipped = clamp_bbox_to_image(bbox, frame_bgr.shape, margin_ratio)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    face = frame_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return cv2.resize(face, (image_size, image_size), interpolation=cv2.INTER_AREA)


def ensure_empty_or_new_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def seconds_to_frame(frame_rate: float, seconds: float) -> int:
    return max(0, int(round(frame_rate * seconds)))


def extract_task(
    task: ExtractionTask,
    detector: BaseFaceDetector,
    sample_step: int,
    image_size: int,
    margin_ratio: float,
    track_frames: int,
) -> Dict[str, object]:
    cap = cv2.VideoCapture(str(task.video_path))
    if not cap.isOpened():
        return {
            "output_dir": str(task.output_dir),
            "label": task.label,
            "status": "video_open_failed",
            "saved_frames": 0,
        }

    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = seconds_to_frame(frame_rate, task.start_sec)
    if task.end_sec is None:
        end_frame = frames_total
    else:
        end_frame = min(frames_total, seconds_to_frame(frame_rate, task.end_sec))
    if end_frame <= start_frame:
        cap.release()
        return {
            "output_dir": str(task.output_dir),
            "label": task.label,
            "status": "empty_range",
            "saved_frames": 0,
        }

    ensure_empty_or_new_dir(task.output_dir)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_index = start_frame
    saved_frames = 0
    recent_bbox: Optional[Tuple[float, float, float, float]] = None
    recent_bbox_budget = 0
    detection_failures = 0

    while frame_index < end_frame:
        success, frame_bgr = cap.read()
        if not success:
            break
        if (frame_index - start_frame) % sample_step != 0:
            frame_index += 1
            continue

        bbox = detector.detect(frame_bgr)
        if bbox is not None:
            recent_bbox = bbox
            recent_bbox_budget = track_frames
        elif recent_bbox is not None and recent_bbox_budget > 0:
            bbox = recent_bbox
            recent_bbox_budget -= 1
        else:
            detection_failures += 1
            frame_index += 1
            continue

        face = crop_face(frame_bgr, bbox, margin_ratio=margin_ratio, image_size=image_size)
        if face is None:
            detection_failures += 1
            frame_index += 1
            continue

        image_path = task.output_dir / f"{saved_frames:06d}.jpg"
        cv2.imwrite(str(image_path), face)
        saved_frames += 1
        frame_index += 1

    cap.release()

    if saved_frames == 0:
        shutil.rmtree(task.output_dir, ignore_errors=True)
        return {
            "output_dir": str(task.output_dir),
            "label": task.label,
            "status": "no_faces_saved",
            "saved_frames": 0,
        }

    manifest = {
        "source_video": str(task.video_path),
        "label": task.label,
        "raw_video_label": task.raw_video_label,
        "start_sec": task.start_sec,
        "end_sec": task.end_sec,
        "saved_frames": saved_frames,
        "detector": detector.name,
        "split_file": None if task.split_file is None else str(task.split_file),
    }
    with (task.output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "output_dir": str(task.output_dir),
        "label": task.label,
        "status": "ok",
        "saved_frames": saved_frames,
        "detection_failures": detection_failures,
    }


def infer_folder_label(folder_path: Path) -> Optional[str]:
    manifest_path = folder_path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            label = manifest.get("label")
            if label in {"normal", "talking", "yawning", "talking_yawning"}:
                return label
        except (OSError, json.JSONDecodeError):
            pass

    lowered = folder_path.name.lower()
    if is_talking_yawning_name(lowered):
        return "talking_yawning"
    if "-normal" in lowered:
        return "normal"
    if "-talking" in lowered:
        return "talking"
    if "-yawning" in lowered:
        return "yawning"
    return None


def count_class_folders(output_root: Path) -> Dict[str, int]:
    counts = {
        "normal": 0,
        "talking": 0,
        "yawning": 0,
        "talking_yawning": 0,
    }
    if not output_root.exists():
        return counts
    for child in output_root.iterdir():
        if not child.is_dir():
            continue
        label = infer_folder_label(child)
        if label is not None:
            counts[label] += 1
    return counts


def summarize_tasks(tasks: Sequence[ExtractionTask]) -> Dict[str, int]:
    counts = {label: 0 for label in ("normal", "talking", "yawning", "talking_yawning")}
    for task in tasks:
        counts[task.label] = counts.get(task.label, 0) + 1
    return counts


def summarize_tasks_by_split(tasks: Sequence[ExtractionTask]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for task in tasks:
        split_name = "unspecified" if task.split_file is None else str(task.split_file)
        split_summary = summary.setdefault(
            split_name,
            {label: 0 for label in ("normal", "talking", "yawning", "talking_yawning")},
        )
        split_summary[task.label] = split_summary.get(task.label, 0) + 1
    return summary


def summarize_raw_sources(tasks: Sequence[ExtractionTask]) -> Dict[str, int]:
    unique_sources: Dict[str, set] = {
        "normal": set(),
        "talking": set(),
        "yawning": set(),
        "talking_yawning": set(),
    }
    for task in tasks:
        unique_sources.setdefault(task.raw_video_label, set()).add(str(task.video_path))
    return {label: len(paths) for label, paths in unique_sources.items()}


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    train_output_root = None if args.train_output_root is None else args.train_output_root.resolve()
    test_output_root = None if args.test_output_root is None else args.test_output_root.resolve()
    test_subjects = parse_subject_list(args.test_subjects)

    if (train_output_root is None) != (test_output_root is None):
        raise ValueError("--train-output-root and --test-output-root must be provided together")
    if train_output_root is not None:
        train_output_root.mkdir(parents=True, exist_ok=True)
        test_output_root.mkdir(parents=True, exist_ok=True)

    tasks, skipped = plan_tasks(
        mirror_root=args.mirror_root,
        output_root=output_root,
        skip_existing=args.skip_existing,
        train_output_root=train_output_root,
        test_output_root=test_output_root,
        test_subjects=test_subjects,
    )
    if args.limit > 0:
        tasks = tasks[: args.limit]

    planned_counts = summarize_tasks(tasks)
    raw_source_counts = summarize_raw_sources(tasks)
    print("Mirror root:", args.mirror_root)
    print("Output root:", output_root)
    if train_output_root is not None:
        print("Train output root:", train_output_root)
        print("Test output root:", test_output_root)
        print("Test subjects:", test_subjects)
    print("Planned folders:", len(tasks))
    print("Planned class counts:", json.dumps(planned_counts, indent=2))
    print("Planned split class counts:", json.dumps(summarize_tasks_by_split(tasks), indent=2))
    print("Planned raw source counts:", json.dumps(raw_source_counts, indent=2))
    if skipped:
        skipped_summary: Dict[str, int] = {}
        for _, reason in skipped:
            skipped_summary[reason] = skipped_summary.get(reason, 0) + 1
        print("Skipped summary:", json.dumps(skipped_summary, indent=2))

    if args.dry_run:
        for task in tasks[:20]:
            print(f"[DRY RUN] {task.label:>7} -> {task.output_dir.name}")
        final_counts = count_class_folders(output_root)
        print("Existing folder counts:", json.dumps(final_counts, indent=2))
        return

    detector = create_detector(args.detector)
    print("Using detector:", detector.name)

    results = []
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {task.output_dir.name}")
        result = extract_task(
            task,
            detector=detector,
            sample_step=args.sample_step,
            image_size=args.image_size,
            margin_ratio=args.bbox_margin,
            track_frames=args.track_frames,
        )
        print("  status:", result["status"], "saved_frames:", result["saved_frames"])
        results.append(result)

    status_summary: Dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        status_summary[status] = status_summary.get(status, 0) + 1
    print("Run summary:", json.dumps(status_summary, indent=2))
    print("Final output folder counts:", json.dumps(count_class_folders(output_root), indent=2))


if __name__ == "__main__":
    main()