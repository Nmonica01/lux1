from collections import defaultdict
from datetime import datetime
from math import hypot
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"
TRACKER_CONFIG = "botsort.yaml"
CONFIDENCE_THRESHOLD = 0.55

# Region to black out before detection, as fractions of frame width/height (0 to 1).
# Tune these to cover the reflective glass/mirror area in your footage.
IGNORE_REGION = {"x": 0.55, "y": 0.0, "width": 0.45, "height": 0.6}

# --- Suspicious-behavior thresholds -----------------------------------------
# A track "loiters" if it stays inside a small radius of its starting point for
# at least this many seconds.
LOITER_SECONDS_THRESHOLD = 20.0
# Radius, as a fraction of the frame diagonal, a loitering track is allowed to
# drift within. Smaller = stricter.
LOITER_DISPLACEMENT_RATIO = 0.035
# A track counts as moving unusually fast if it covers more than this fraction
# of the frame diagonal per second between two consecutive detections.
FAST_MOVEMENT_SPEED_RATIO = 0.6
PERSON_CLASS_NAME = "person"
# Default restricted window (24h clock). 22 -> 6 means 10pm to 6am.
DEFAULT_RESTRICTED_START_HOUR = 22
DEFAULT_RESTRICTED_END_HOUR = 6

app = FastAPI(title="Nightwatch Video Analytics API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:5176"], allow_methods=["*"], allow_headers=["*"])
model = YOLO(MODEL_PATH)


def mask_reflective_region(source_path: str) -> str:
    """Writes a copy of the video with IGNORE_REGION blacked out on every frame."""
    capture = cv2.VideoCapture(source_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open video for masking.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25

    x1 = int(IGNORE_REGION["x"] * width)
    y1 = int(IGNORE_REGION["y"] * height)
    x2 = int((IGNORE_REGION["x"] + IGNORE_REGION["width"]) * width)
    y2 = int((IGNORE_REGION["y"] + IGNORE_REGION["height"]) * height)

    masked_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    masked_path = masked_file.name
    masked_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(masked_path, fourcc, fps, (width, height))

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
        writer.write(frame)

    capture.release()
    writer.release()
    return masked_path


def _source_fps(path: str) -> float:
    capture = cv2.VideoCapture(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    capture.release()
    return fps


def _behavior_events(
    track_positions: dict[int, list[tuple[int, float, float]]],
    track_classes: dict[int, str],
    fps: float,
    frame_diagonal: float | None,
) -> list[dict]:
    """Turns raw per-frame track positions into loitering / fast-movement flags."""
    events: list[dict] = []
    if not frame_diagonal:
        return events

    for track_id, positions in track_positions.items():
        if len(positions) < 2:
            continue
        object_class = track_classes[track_id]
        first_frame, first_x, first_y = positions[0]
        last_frame, _, _ = positions[-1]
        duration_seconds = (last_frame - first_frame) / fps

        max_displacement = max(hypot(x - first_x, y - first_y) for _, x, y in positions)
        if duration_seconds >= LOITER_SECONDS_THRESHOLD and (max_displacement / frame_diagonal) <= LOITER_DISPLACEMENT_RATIO:
            events.append(
                {
                    "type": "loitering",
                    "object_class": object_class,
                    "track_id": track_id,
                    "duration_seconds": round(duration_seconds, 1),
                    "detail": f"{object_class} stayed in roughly the same spot for {round(duration_seconds, 1)}s",
                }
            )

        peak_speed_ratio = 0.0
        for (frame_a, x_a, y_a), (frame_b, x_b, y_b) in zip(positions, positions[1:]):
            elapsed = (frame_b - frame_a) / fps
            if elapsed <= 0:
                continue
            distance = hypot(x_b - x_a, y_b - y_a)
            speed_ratio = (distance / frame_diagonal) / elapsed
            peak_speed_ratio = max(peak_speed_ratio, speed_ratio)

        if peak_speed_ratio >= FAST_MOVEMENT_SPEED_RATIO:
            events.append(
                {
                    "type": "fast_movement",
                    "object_class": object_class,
                    "track_id": track_id,
                    "duration_seconds": None,
                    "detail": f"{object_class} (track {track_id}) moved unusually fast across the frame",
                }
            )

    return events


def analyze_video(path: str, restricted_start_hour: int, restricted_end_hour: int) -> dict:
    unique_ids: dict[str, set[int]] = defaultdict(set)
    observations: dict[str, int] = defaultdict(int)
    peak_visible: dict[str, int] = defaultdict(int)
    confidence_totals: dict[str, float] = defaultdict(float)
    confidence_samples: dict[str, int] = defaultdict(int)
    track_positions: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    track_classes: dict[int, str] = {}
    frame_diagonal: float | None = None
    person_seen = False
    frames = 0

    fps = _source_fps(path)
    results = model.track(source=path, persist=True, tracker=TRACKER_CONFIG, conf=CONFIDENCE_THRESHOLD, stream=True, verbose=False)
    for result in results:
        frames += 1
        if frame_diagonal is None:
            height, width = result.orig_shape
            frame_diagonal = hypot(width, height)
        if result.boxes is None or result.boxes.id is None:
            continue
        boxes = result.boxes.cpu()
        track_ids = boxes.id.int().tolist()
        class_ids = boxes.cls.int().tolist()
        scores = boxes.conf.tolist()
        coordinates = boxes.xyxy.tolist()
        visible: dict[str, int] = defaultdict(int)
        for track_id, class_id, score, (x1, y1, x2, y2) in zip(track_ids, class_ids, scores, coordinates):
            object_class = model.names.get(class_id, f"class-{class_id}")
            unique_ids[object_class].add(track_id)
            observations[object_class] += 1
            visible[object_class] += 1
            confidence_totals[object_class] += score
            confidence_samples[object_class] += 1
            track_classes[track_id] = object_class
            track_positions[track_id].append((frames, (x1 + x2) / 2, (y1 + y2) / 2))
            if object_class == PERSON_CLASS_NAME:
                person_seen = True
        for object_class, count in visible.items():
            peak_visible[object_class] = max(peak_visible[object_class], count)

    summary = {
        object_class: {
            "unique": len(ids),
            "peak": peak_visible[object_class],
            "observations": observations[object_class],
            "confidence": round(confidence_totals[object_class] / confidence_samples[object_class], 4),
        }
        for object_class, ids in unique_ids.items()
    }

    suspicious_events = _behavior_events(track_positions, track_classes, fps, frame_diagonal)

    analyzed_at = datetime.now()
    if restricted_start_hour > restricted_end_hour:
        restricted_hours_active = analyzed_at.hour >= restricted_start_hour or analyzed_at.hour < restricted_end_hour
    else:
        restricted_hours_active = restricted_start_hour <= analyzed_at.hour < restricted_end_hour
    if person_seen and restricted_hours_active:
        suspicious_events.append(
            {
                "type": "after_hours_activity",
                "object_class": PERSON_CLASS_NAME,
                "track_id": None,
                "duration_seconds": None,
                "detail": f"Person activity detected during restricted hours (analyzed at {analyzed_at.strftime('%H:%M')})",
            }
        )

    return {
        "frames": frames,
        "raw_detections": sum(observations.values()),
        "objects": summary,
        "suspicious_events": suspicious_events,
        "analyzed_at": analyzed_at.isoformat(),
        "restricted_hours_active": restricted_hours_active,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    restricted_start_hour: int = Form(DEFAULT_RESTRICTED_START_HOUR),
    restricted_end_hour: int = Form(DEFAULT_RESTRICTED_END_HOUR),
) -> dict:
    if not file.filename or not Path(file.filename).suffix.lower() in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Upload an MP4, WebM, MOV, AVI, or MKV video.")
    if not (0 <= restricted_start_hour <= 23) or not (0 <= restricted_end_hour <= 23):
        raise HTTPException(status_code=400, detail="Restricted hours must be between 0 and 23.")
    temporary_path = None
    masked_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temporary:
            temporary.write(await file.read())
            temporary_path = temporary.name
        masked_path = mask_reflective_region(temporary_path)
        return {"filename": file.filename, **analyze_video(masked_path, restricted_start_hour, restricted_end_hour)}
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {error}") from error
    finally:
        if temporary_path:
            Path(temporary_path).unlink(missing_ok=True)
        if masked_path:
            Path(masked_path).unlink(missing_ok=True)
