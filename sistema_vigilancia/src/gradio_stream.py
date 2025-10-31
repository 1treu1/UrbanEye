from __future__ import annotations

import os
from typing import Generator, List, Tuple, Optional, Dict, Any
from collections import deque

import cv2
import gradio as gr
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO


_GPU_READY = False
_ROI_MODE: str = "none"  # none | consolidated | visits
_CSV_PATH: str = ""
_FPS_ASSUMED: int = 22
_WINDOW_FRAMES: int = 0
_GLOBAL_FRAME_IDX: int = 0

# YOLOv11 tracker
_YOLO_MODEL: Optional[YOLO] = None
_YOLO_MODEL_INITIALIZED: bool = False

# Track data storage
_TRACKS: Dict[int, Dict[str, Any]] = {}  # track_id -> {bbox, attributes, trail, enter_time, etc.}
_TRAILS: Dict[int, deque] = {}  # track_id -> deque of (x, y) positions for trail drawing
_MAX_TRAIL_LENGTH: int = 50  # Maximum points in trail
_LEAVE_TOL: int = 5  # visits mode tolerance


def _init_yolo_model() -> None:
    """Initialize YOLOv11 model for person tracking."""
    global _YOLO_MODEL, _YOLO_MODEL_INITIALIZED
    if _YOLO_MODEL_INITIALIZED and _YOLO_MODEL is not None:
        return
    
    try:
        # Use YOLO11 nano for speed - use local file if available
        import os
        model_path = "yolo11n.pt"
        # Check if model exists in current directory or workspace root
        if not os.path.exists(model_path):
            # Try workspace root
            workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path_alt = os.path.join(workspace_root, "yolo11n.pt")
            if os.path.exists(model_path_alt):
                model_path = model_path_alt
        
        _YOLO_MODEL = YOLO(model_path)  # nano version for speed
        _YOLO_MODEL_INITIALIZED = True
        print(f"YOLO11 model initialized successfully from {model_path}")
    except Exception as e:
        print(f"Warning: Could not initialize YOLO11 model: {e}")
        _YOLO_MODEL = None
        _YOLO_MODEL_INITIALIZED = False




def _flush_window() -> None:
    """Flush tracks that are still in ROI after window expires or video ends."""
    global _GLOBAL_FRAME_IDX, _FPS_ASSUMED
    
    tracks_to_flush = []
    for tid, track in list(_TRACKS.items()):
        # Check if track has entered ROI but hasn't exited yet
        if "enter_time" in track and track.get("inside", False):
            # Set exit time to current frame
            track["exit_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0  # minutes
            track["inside"] = False
            tracks_to_flush.append((tid, track.copy()))  # Copy to avoid modification during iteration
        # Also check inactive tracks that might have entered but weren't detected exiting
        elif "enter_time" in track and "exit_time" not in track:
            # Track entered ROI but never exited - flush it now
            track["exit_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0  # minutes
            track["inside"] = False
            tracks_to_flush.append((tid, track.copy()))
    
    for tid, track in tracks_to_flush:
        _write_track_to_csv(tid, track)


def _accumulate_roi(tid: int, inside: bool, det: Dict[str, Any]) -> None:
    """Accumulate ROI data for a track."""
    global _GLOBAL_FRAME_IDX, _FPS_ASSUMED
    
    if tid not in _TRACKS:
        return
    
    track = _TRACKS[tid]
    was_inside = track.get("inside", False)
    
    if inside:
        if not was_inside:
            # Just entered ROI - record entry time
            # time = frames / fps / 60 (to get minutes)
            track["enter_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0
            track["frames_in"] = 0
            track["inside"] = True
            # Initialize attribute lists if not exists
            if "ages" not in track:
                track["ages"] = []
                track["genders"] = []
                track["races"] = []
                track["emotions"] = []
        
        # Track is inside ROI - accumulate data
        track["frames_in"] = track.get("frames_in", 0) + 1
        
        # Update attributes only if detection data is available
        # Only update if we have new data (not empty dict)
        if det and det.get("age") is not None:
            track["ages"].append(float(det["age"]))
        if det and det.get("gender"):
            track["genders"].append(str(det["gender"]))
        if det and det.get("race"):
            track["races"].append(str(det["race"]))
        if det and det.get("emotion"):
            track["emotions"].append(str(det["emotion"]))
    else:
        if was_inside:
            # Just exited ROI - record exit time and write to CSV
            track["exit_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0
            track["inside"] = False
            _write_track_to_csv(tid, track)
            # Clear entry time and attributes for potential re-entry
            # But keep the track data structure
            if "enter_time" in track:
                del track["enter_time"]
            if "exit_time" in track:
                del track["exit_time"]
            track["ages"] = []
            track["genders"] = []
            track["races"] = []
            track["emotions"] = []
            track["frames_in"] = 0


def _write_track_to_csv(tid: int, track: Dict[str, Any]) -> None:
    """Write a completed track to CSV."""
    global _CSV_PATH, _FPS_ASSUMED
    
    if not _CSV_PATH or _ROI_MODE == "none":
        return
    
    if "enter_time" not in track or "exit_time" not in track:
        return
    
    # Calculate values
    time_input = track["enter_time"]  # already in minutes
    time_out = track["exit_time"]  # already in minutes
    time_2 = time_out - time_input  # duration in minutes
    
    # Calculate average age
    if track.get("ages") and len(track["ages"]) > 0:
        avg_age = sum(track["ages"]) / len(track["ages"])
    else:
        avg_age = 0.0
    
    # For gender, race, emotion: join unique values with commas if multiple
    genders = track.get("genders", [])
    if genders:
        genders_unique = list(dict.fromkeys(genders))  # Preserve order, remove duplicates
        genders_str = ",".join(genders_unique) if len(genders_unique) > 1 else genders_unique[0] if genders_unique else ""
    else:
        genders_str = ""
    
    races = track.get("races", [])
    if races:
        races_unique = list(dict.fromkeys(races))  # Preserve order, remove duplicates
        races_str = ",".join(races_unique) if len(races_unique) > 1 else races_unique[0] if races_unique else ""
    else:
        races_str = ""
    
    emotions = track.get("emotions", [])
    if emotions:
        emotions_unique = list(dict.fromkeys(emotions))  # Preserve order, remove duplicates
        emotions_str = ",".join(emotions_unique) if len(emotions_unique) > 1 else emotions_unique[0] if emotions_unique else ""
    else:
        emotions_str = ""
    
    # Write to CSV
    import csv
    import os
    file_exists = os.path.exists(_CSV_PATH) and os.path.getsize(_CSV_PATH) > 0
    
    with open(_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header if file is new or empty
        if not file_exists:
            writer.writerow(["time_input", "time_out", "track_id", "age", "gender", "race", "emotion", "time_2"])
        
        writer.writerow([
            f"{time_input:.6f}",
            f"{time_out:.6f}",
            tid,
            f"{avg_age:.1f}",
            genders_str,
            races_str,
            emotions_str,
            f"{time_2:.6f}"
        ])


def set_roi_config(roi_mode: str, csv_path: str, duration_min: int, fps: int) -> None:
    global _ROI_MODE, _CSV_PATH, _FPS_ASSUMED, _WINDOW_FRAMES, _GLOBAL_FRAME_IDX
    _ROI_MODE = roi_mode
    _CSV_PATH = csv_path
    _FPS_ASSUMED = fps
    _WINDOW_FRAMES = max(1, duration_min * 60 * fps)
    _GLOBAL_FRAME_IDX = 0

    # reset tracker
    global _NEXT_TRACK_ID, _TRACKS
    _NEXT_TRACK_ID = 1
    _TRACKS.clear()

    # write header if file not exists
    if _ROI_MODE != "none" and _CSV_PATH:
        if not os.path.exists(_CSV_PATH) or os.path.getsize(_CSV_PATH) == 0:
            with open(_CSV_PATH, "w", encoding="utf-8", newline="") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["time_input", "time_out", "track_id", "age", "gender", "race", "emotion", "time_2"])


def _ensure_tf_gpu() -> None:
    global _GPU_READY
    if _GPU_READY:
        return
    # Prefer graceful memory growth to avoid pre-allocating full VRAM
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
    try:
        import tensorflow as tf  # type: ignore
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)  # type: ignore
            except Exception:
                pass
        # Optional XLA; keep conservative to avoid surprises
        try:
            tf.config.optimizer.set_jit(False)  # type: ignore
        except Exception:
            pass
    except Exception:
        # If TF not available, proceed silently (DeepFace may still use PyTorch)
        pass
    _GPU_READY = True


def analyze_frame(frame_bgr: np.ndarray, roi_size: float = 0.65) -> np.ndarray:
    """Run YOLOv11 for person tracking and DeepFace for face attributes.
    
    Uses YOLOv11 to track people continuously, and DeepFace to analyze faces
    inside the ROI for age, gender, race, and emotion.
    """
    global _GLOBAL_FRAME_IDX, _TRACKS, _TRAILS
    
    # Initialize YOLOv11 model if not already done
    _init_yolo_model()
    
    # GPU/TF setup for DeepFace (no-op on subsequent calls)
    _ensure_tf_gpu()
    
    annotated = frame_bgr.copy()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # Compute ROI rectangle
    h_img, w_img = annotated.shape[:2]
    rw = int(roi_size * w_img)
    rh = int(roi_size * h_img)
    rx0 = (w_img - rw) // 2
    ry0 = (h_img - rh) // 2
    rx1 = rx0 + rw
    ry1 = ry0 + rh
    
    # Step 1: Use YOLOv11 to detect and track persons (class 0 in COCO)
    yolo_tracks: Dict[int, Dict[str, Any]] = {}  # track_id -> {bbox, conf, inside_roi}
    
    if _YOLO_MODEL is not None:
        try:
            # Run YOLOv11 tracking (persist=True maintains IDs across frames)
            results = _YOLO_MODEL.track(
                frame_bgr,
                persist=True,
                classes=[0],  # Only detect persons (class 0)
                conf=0.25,    # Confidence threshold
                verbose=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes
                    # Process each tracked person
                    for i in range(len(boxes)):
                        track_id = int(boxes.id[i].item())
                        box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                        conf = float(boxes.conf[i].item())
                        
                        x1, y1, x2, y2 = box
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        cx, cy = x + w // 2, y + h // 2
                        
                        # Check if center is inside ROI
                        inside_roi = (rx0 <= cx <= rx1) and (ry0 <= cy <= ry1)
                        
                        yolo_tracks[track_id] = {
                            "bbox": (x, y, w, h),
                            "bbox_xyxy": box.tolist(),
                            "conf": conf,
                            "inside_roi": inside_roi,
                            "center": (cx, cy)
                        }
        except Exception as e:
            print(f"YOLOv11 tracking error: {e}")
    
    # Step 2: For persons inside ROI, extract face region and run DeepFace
    # Update track data with DeepFace attributes
    deepface_results: Dict[int, Dict[str, Any]] = {}  # track_id -> attributes
    
    for track_id, yolo_track in yolo_tracks.items():
        if not yolo_track["inside_roi"]:
            continue
        
        x, y, w, h = yolo_track["bbox"]
        # Extract face region (expand a bit for better detection)
        face_margin = 20
        face_x = max(0, x - face_margin)
        face_y = max(0, y - face_margin)
        face_w = min(w_img - face_x, w + 2 * face_margin)
        face_h = min(h_img - face_y, h + 2 * face_margin)
        
        if face_w > 0 and face_h > 0:
            face_roi = frame_rgb[face_y:face_y + face_h, face_x:face_x + face_w]
            
            try:
                # Run DeepFace on face region
                df_results = DeepFace.analyze(
                    face_roi,
                    actions=["age", "gender", "race", "emotion"],
                    enforce_detection=False,
                    align=True,
                    detector_backend="retinaface",
                    silent=True
                )
                
                if isinstance(df_results, dict):
                    df_results = [df_results]
                
                if df_results and len(df_results) > 0:
                    res = df_results[0]
                    deepface_results[track_id] = {
                        "age": res.get("age"),
                        "gender": res.get("dominant_gender") or res.get("gender"),
                        "race": res.get("dominant_race") or res.get("race"),
                        "emotion": res.get("dominant_emotion") or res.get("emotion"),
                    }
            except Exception:
                # DeepFace failed for this face, continue
                pass
    
    # Step 3: Update global track storage and ROI tracking
    # Update or create tracks from YOLOv11 results
    current_track_ids = set()
    
    for track_id, yolo_track in yolo_tracks.items():
        current_track_ids.add(track_id)
        
        # Update or create track
        if track_id not in _TRACKS:
            _TRACKS[track_id] = {
                "bbox": yolo_track["bbox"],
                "ages": [],
                "genders": [],
                "races": [],
                "emotions": [],
                "inside": False,
                "frames_in": 0,
            }
            # Initialize trail
            _TRAILS[track_id] = deque(maxlen=_MAX_TRAIL_LENGTH)
        
        track = _TRACKS[track_id]
        track["bbox"] = yolo_track["bbox"]
        track["last_seen"] = _GLOBAL_FRAME_IDX
        
        # Update trail with center position
        cx, cy = yolo_track["center"]
        _TRAILS[track_id].append((cx, cy))
        
        # Process ROI tracking
        inside_roi = yolo_track["inside_roi"]
        was_inside = track.get("inside", False)
        
        # Get DeepFace attributes if available
        df_attrs = deepface_results.get(track_id, {})
        
        if inside_roi:
            if not was_inside:
                # Just entered ROI
                track["enter_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0
                track["frames_in"] = 0
                track["inside"] = True
                track["ages"] = []
                track["genders"] = []
                track["races"] = []
                track["emotions"] = []
            
            track["frames_in"] = track.get("frames_in", 0) + 1
            
            # Accumulate DeepFace attributes
            if df_attrs.get("age"):
                track["ages"].append(float(df_attrs["age"]))
            if df_attrs.get("gender"):
                track["genders"].append(str(df_attrs["gender"]))
            if df_attrs.get("race"):
                track["races"].append(str(df_attrs["race"]))
            if df_attrs.get("emotion"):
                track["emotions"].append(str(df_attrs["emotion"]))
        else:
            if was_inside:
                # Just exited ROI
                track["exit_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0
                track["inside"] = False
                _write_track_to_csv(track_id, track)
                # Clear for potential re-entry
                if "enter_time" in track:
                    del track["enter_time"]
                if "exit_time" in track:
                    del track["exit_time"]
                track["ages"] = []
                track["genders"] = []
                track["races"] = []
                track["emotions"] = []
                track["frames_in"] = 0
    
    # Remove old tracks (not seen in current frame)
    tracks_to_remove = []
    for track_id in list(_TRACKS.keys()):
        if track_id not in current_track_ids:
            # Track disappeared - if inside ROI, flush it
            if _TRACKS[track_id].get("inside", False):
                track = _TRACKS[track_id]
                track["exit_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0
                track["inside"] = False
                _write_track_to_csv(track_id, track)
                # Clear for potential re-entry
                if "enter_time" in track:
                    del track["enter_time"]
                if "exit_time" in track:
                    del track["exit_time"]
            tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        if track_id in _TRACKS:
            del _TRACKS[track_id]
        if track_id in _TRAILS:
            del _TRAILS[track_id]
    
    # Step 4: Draw everything
    # Draw trails first (so boxes appear on top)
    for track_id, trail in _TRAILS.items():
        if track_id not in yolo_tracks:
            continue
        
        # Draw trail as connected lines
        if len(trail) > 1:
            points = list(trail)
            for i in range(1, len(points)):
                pt1 = points[i - 1]
                pt2 = points[i]
                # Use gradient color - newer points brighter
                alpha = i / len(points)
                color_intensity = int(255 * alpha)
                cv2.line(annotated, pt1, pt2, (color_intensity, color_intensity, 255), 2)
    
    # Draw YOLOv11 tracks (cyan for all tracks)
    num_tracks = 0
    for track_id, yolo_track in yolo_tracks.items():
        x, y, w, h = yolo_track["bbox"]
        
        # Draw track box in cyan
        color = (255, 255, 0)  # Cyan in BGR
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        cv2.putText(annotated, f"ID: {track_id}", (x, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        num_tracks += 1
        
        # If inside ROI and has DeepFace attributes, draw them
        if yolo_track["inside_roi"] and track_id in deepface_results:
            df_attrs = deepface_results[track_id]
            age = df_attrs.get("age")
            gender = df_attrs.get("gender")
            race = df_attrs.get("race")
            emotion = df_attrs.get("emotion")
            
            elapsed_txt = ""
            if track_id in _TRACKS and "enter_time" in _TRACKS[track_id]:
                frames_in = _TRACKS[track_id].get("frames_in", 0)
                elapsed_txt = f" • {frames_in / max(1, _FPS_ASSUMED):.1f}s"
            
            label = f"age: {age}  gender: {gender}  race: {race}  emotion: {emotion}{elapsed_txt}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_text = max(0, y - 35)
            cv2.rectangle(annotated, (x, max(0, y_text - th - 4)), (x + tw + 6, y_text + 2), (0, 0, 0), -1)
            cv2.putText(annotated, label, (x + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Status text
    status_text = f"tracks: {num_tracks}"
    cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if num_tracks == 0:
        cv2.putText(annotated, "No persons detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Draw ROI rectangle
    cv2.rectangle(annotated, (rx0, ry0), (rx1, ry1), (0, 0, 255), 2)
    
    # Window flush
    if _ROI_MODE != "none" and _WINDOW_FRAMES > 0 and _GLOBAL_FRAME_IDX >= _WINDOW_FRAMES:
        _flush_window()
    
    return annotated


def stream_generator(video_path: str, analyze_every_n: int = 10, max_width: int = 640, roi_size: float = 0.65) -> Generator[np.ndarray, None, None]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    try:
        # small buffer to reduce latency
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        frame_idx = 0
        last_annotated: Optional[np.ndarray] = None
        global _GLOBAL_FRAME_IDX, _FPS_ASSUMED
        
        while True:
            ok, frame = cap.read()
            if not ok:
                # Video ended - flush any remaining tracks in ROI
                if _ROI_MODE != "none":
                    _flush_window()
                break

            # Resize for speed, preserve aspect ratio
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                frame_small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame

            # Increment global frame counter for each frame (for accurate time tracking)
            _GLOBAL_FRAME_IDX += 1

            # Analyze every N frames; reuse last annotation otherwise for smoothness
            if frame_idx % analyze_every_n == 0 or last_annotated is None:
                annotated_small = analyze_frame(frame_small, roi_size=roi_size)
                # If we downscaled, upscale annotations back to original size for display stability
                if annotated_small.shape[1] != w:
                    annotated = cv2.resize(annotated_small, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    annotated = annotated_small
                last_annotated = annotated
            else:
                annotated = last_annotated

            frame_idx += 1

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            yield annotated_rgb
    finally:
        # Final flush of any remaining tracks in ROI when stream ends
        if _ROI_MODE != "none":
            # Force flush all tracks that have entered ROI but haven't exited
            _flush_window()
            # Additional check: flush any tracks with enter_time but no exit_time
            for tid, track in list(_TRACKS.items()):
                if "enter_time" in track and "exit_time" not in track:
                    track["exit_time"] = _GLOBAL_FRAME_IDX / _FPS_ASSUMED / 60.0
                    track["inside"] = False
                    _write_track_to_csv(tid, track)
        cap.release()


def build_demo(default_video: str, roi_mode: str = "consolidated", csv_path: str = "", duration_min: int = 30, fps: int = 22) -> gr.Blocks:
    # set ROI configuration - auto-configure CSV path if not provided
    if not csv_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "roi_stats.csv")
    set_roi_config(roi_mode=roi_mode, csv_path=csv_path, duration_min=duration_min, fps=fps)
    
    with gr.Blocks() as demo:
        gr.Markdown("## Sistema de Vigilancia - DeepFace Stream (video)")
        path_in = gr.Textbox(value=default_video, label="Ruta del video", interactive=True)
        roi_size_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.65,
            step=0.05,
            label="Tamaño del rectángulo ROI (porcentaje)",
            info="0.7 = 70% del tamaño de la imagen"
        )
        output = gr.Image(label="Stream", streaming=True, type="numpy")
        start_btn = gr.Button("Iniciar")

        def start(vpath: str, roi_size: float) -> Generator[np.ndarray, None, None]:
            # Reset ROI config for new video
            set_roi_config(roi_mode=roi_mode, csv_path=csv_path, duration_min=duration_min, fps=fps)
            # Stream with throttled DeepFace analysis for smoother playback
            yield from stream_generator(vpath, analyze_every_n=10, max_width=640, roi_size=roi_size)

        start_btn.click(fn=start, inputs=[path_in, roi_size_slider], outputs=output)

    # Warm-up: run a quick analyze on first frame to load models
    try:
        cap = cv2.VideoCapture(default_video)
        ok, frame = cap.read()
        if ok:
            _ = analyze_frame(frame)
        cap.release()
    except Exception:
        pass

    return demo


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(base_dir, "videos", "face-demographics-walking-and-pause.mp4")
    port = int(os.getenv("PORT", "7860"))
    demo = build_demo(default_path)
    demo.queue().launch(server_name="0.0.0.0", server_port=port, share=True)


