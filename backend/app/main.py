from __future__ import annotations

# =========================
# Env & tunables (override via env)
# =========================
import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Camera (request 1080p; falls back if device can't)
CAM_INDEX_DEFAULT = int(os.getenv("CAM_INDEX", "0"))
DEFAULT_BACKEND   = os.getenv("CAM_BACKEND", "dshow")      # dshow|msmf
FRAME_W           = int(os.getenv("CAM_W", "1920"))
FRAME_H           = int(os.getenv("CAM_H", "1080"))
TARGET_FPS        = int(os.getenv("CAM_FPS", "30"))
JPEG_QUALITY      = int(os.getenv("JPEG_QUALITY", "65"))   # 60–75 is good at 1080p MJPEG

# Inference & eventing
MODEL_NAME   = os.getenv("MODEL_NAME", "yolov8n.pt")       # 'yolov8s.pt' = more recall, slower
IMG_SIZE     = int(os.getenv("IMG_SIZE", "416"))           # 320/416/480 advisable on CPU
CONF_THRESH  = float(os.getenv("CONF_THRESH", "0.35"))     # lower → more boxes
IOU_THRESH   = float(os.getenv("IOU_THRESH", "0.45"))
MAX_DET      = int(os.getenv("MAX_DET", "100"))
INFER_HZ     = float(os.getenv("INFER_HZ", "7"))           # cap detector Hz
EVENT_HEARTBEAT_SEC = float(os.getenv("EVENT_HEARTBEAT_SEC", "2.0"))  # minimal keep-alive
MOVE_THRESH_FRAC    = float(os.getenv("MOVE_THRESH_FRAC", "0.10"))    # ~10% of diagonal to count as "moved"
TOPK         = int(os.getenv("TOPK", "8"))                 # cap outgoing boxes by area (UI perf)

# Group / "team" detection
TEAM_MIN_COUNT   = int(os.getenv("TEAM_MIN_COUNT", "6"))   # >= this many persons → one team box
TEAM_LINK_THRESH = float(os.getenv("TEAM_LINK_THRESH", "0.12"))  # links persons within this frac of diag

# Which classes to keep for obstacle warnings
ANNOUNCE = {"person", "laptop", "chair", "pottedplant", "mouse", "pen"}

# =========================
# Imports
# =========================
import asyncio
import json
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# =========================
# FastAPI app + CORS
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# YOLOv8 model (CPU)
# =========================
model = YOLO(MODEL_NAME)   # auto-downloads if needed
try: model.fuse()
except Exception: pass
model.to("cpu")
try: model.model.eval()
except Exception: pass
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

# =========================
# Camera manager (single-reader)
# =========================
BACKENDS: Dict[str, int] = {
    "dshow": cv2.CAP_DSHOW,
    "msmf":  cv2.CAP_MSMF,
    "any":   cv2.CAP_ANY,
}
FOURCC_TRIALS = ["MJPG", "YUY2", None]  # MJPG usually enables 1080p at decent FPS on Windows

cap_lock = threading.Lock()
cap: Optional[cv2.VideoCapture] = None
cap_index = CAM_INDEX_DEFAULT
cap_backend_name = DEFAULT_BACKEND

latest_frame: Optional[np.ndarray] = None
frame_lock = threading.Lock()
frames_read = 0
last_frame_ts = 0.0

def _open_camera(index: int, backend_name: str, w: int, h: int, fps: int):
    backend = BACKENDS.get(backend_name, cv2.CAP_DSHOW)
    c = cv2.VideoCapture(index, backend)
    if not c.isOpened() and backend_name != "msmf":
        try: c.release()
        except Exception: pass
        backend_name = "msmf"
        c = cv2.VideoCapture(index, BACKENDS["msmf"])
    if not c.isOpened() and backend_name != "any":
        try: c.release()
        except Exception: pass
        backend_name = "any"
        c = cv2.VideoCapture(index, BACKENDS["any"])

    if not c.isOpened():
        return None, backend_name

    # request resolution & fps
    c.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    c.set(cv2.CAP_PROP_FPS,          fps)
    # reduce buffering & pick compressed format
    try: c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception: pass
    for four in FOURCC_TRIALS:
        if four is None: break
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*four))
    return c, backend_name

def camera_open(index: int, backend_name: str, w: int, h: int, fps: int) -> bool:
    global cap, cap_index, cap_backend_name
    with cap_lock:
        if cap is not None:
            try: cap.release()
            except Exception: pass
            cap = None
        c, used = _open_camera(index, backend_name, w, h, fps)
        if c is None or not c.isOpened():
            print(f"[ERROR] Could not open camera {index} (tried {backend_name}→fallbacks).")
            return False
        cap = c
        cap_index = index
        cap_backend_name = used
        print(f"[OK] Opened camera {index} via {used} at requested {w}x{h}@{fps}.")
        return True

def camera_reader():
    global latest_frame, frames_read, last_frame_ts
    while True:
        with cap_lock:
            c = cap
        if c is None or not c.isOpened():
            time.sleep(0.05); continue
        ok, frame = c.read()
        if not ok or frame is None:
            time.sleep(0.01); continue
        # If your cam is mirrored, uncomment the next line:
        # frame = cv2.flip(frame, 1)
        with frame_lock:
            latest_frame = frame
            frames_read += 1
            last_frame_ts = time.time()
        time.sleep(0.001)

# boot camera
camera_open(CAM_INDEX_DEFAULT, DEFAULT_BACKEND, FRAME_W, FRAME_H, TARGET_FPS)
threading.Thread(target=camera_reader, daemon=True).start()

# =========================
# Pub/Sub + helpers
# =========================
subscribers: Set[asyncio.Queue] = set()
def broadcast(event: Dict[str, Any]) -> None:
    for q in list(subscribers):
        try: q.put_nowait(event)
        except asyncio.QueueFull: pass

def danger_from_bbox(w: float, h: float) -> str:
    area = w * h
    if area > 0.25: return "high"
    if area > 0.12: return "med"
    if area > 0.05:  return "low"
    return "none"

def _cluster_people(people_boxes: List[Tuple[float,float,float,float]], team_min: int, link_thresh: float, W: int, H: int):
    if not people_boxes: return []
    cxcy = [((x1+x2)/2.0, (y1+y2)/2.0) for (x1,y1,x2,y2) in people_boxes]
    diag = (W**2 + H**2) ** 0.5
    thr = link_thresh * diag
    n = len(people_boxes)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    for i in range(n):
        x1, y1 = cxcy[i]
        for j in range(i+1, n):
            x2, y2 = cxcy[j]
            if ((x1-x2)**2 + (y1-y2)**2) ** 0.5 <= thr:
                union(i, j)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)
    clusters = []
    for idxs in groups.values():
        if len(idxs) < team_min: continue
        xs1, ys1, xs2, ys2 = [], [], [], []
        for i in idxs:
            x1, y1, x2, y2 = people_boxes[i]
            xs1.append(x1); ys1.append(y1); xs2.append(x2); ys2.append(y2)
        clusters.append((min(xs1), min(ys1), max(xs2), max(ys2), len(idxs)))
    return clusters

def _signature_of_threats(threats: List[Dict[str, Any]]) -> str:
    """
    Build a coarse signature so small movement doesn't spam.
    Quantize centers/sizes to 5% bins; include class+danger only for med/high.
    """
    sig_parts = []
    for d in threats:
        bx = d["box"]
        cx = bx["x"] + bx["w"]/2
        cy = bx["y"] + bx["h"]/2
        qw = round(bx["w"]/0.05)    # 5% bins
        qh = round(bx["h"]/0.05)
        qx = round(cx/0.05)
        qy = round(cy/0.05)
        sig_parts.append((d["class"], d["danger"], qx, qy, qw, qh))
    sig_parts.sort()
    return str(sig_parts)

# =========================
# Inference loop (throttled, event-based)
# =========================
_last_sig = ""
_last_broadcast_ts = 0.0

def inference_loop():
    global _last_sig, _last_broadcast_ts
    prev_infer = 0.0

    with torch.inference_mode():
        while True:
            now = time.time()
            if now - prev_infer < (1.0 / INFER_HZ):
                time.sleep(0.001); continue
            prev_infer = now

            with frame_lock:
                frame = None if latest_frame is None else latest_frame.copy()
            if frame is None:
                time.sleep(0.005); continue

            H, W = frame.shape[:2]
            results = model(
                frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                max_det=MAX_DET,
                verbose=False
            )
            r = results[0]

            detections: List[Dict[str, Any]] = []
            person_boxes_abs: List[Tuple[float,float,float,float]] = []

            # Collect detections
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                cls_name = r.names.get(cls_idx, str(cls_idx)) if hasattr(r, "names") else str(cls_idx)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                nx, ny = x1 / W, y1 / H
                nw, nh = (x2 - x1) / W, (y2 - y1) / H
                dang = danger_from_bbox(nw, nh)

                # gather for clustering
                if cls_name == "person" and conf >= CONF_THRESH:
                    person_boxes_abs.append((x1, y1, x2, y2))

                # we keep only interesting classes to minimize payload
                if cls_name in ANNOUNCE and conf >= CONF_THRESH:
                    detections.append({
                        "id": f"{int(x1)}-{int(y1)}-{int(now*1000)}",
                        "class": cls_name,
                        "conf": round(conf, 2),
                        "box": {"x": round(nx,4), "y": round(ny,4), "w": round(nw,4), "h": round(nh,4)},
                        "distance_m": None,
                        "danger": dang,
                    })

            # Team clustering → synthetic 'team' box (always high danger)
            teams = _cluster_people(person_boxes_abs, TEAM_MIN_COUNT, TEAM_LINK_THRESH, W, H)
            for (x1,y1,x2,y2,count) in teams:
                nx, ny = x1 / W, y1 / H
                nw, nh = (x2 - x1) / W, (y2 - y1) / H
                detections.append({
                    "id": f"team-{int(x1)}-{int(y1)}-{int(now*1000)}",
                    "class": "team",
                    "conf": 1.0,
                    "count": count,
                    "box": {"x": round(nx,4), "y": round(ny,4), "w": round(nw,4), "h": round(nh,4)},
                    "distance_m": None,
                    "danger": "high",
                })

            # Only keep MED/HIGH threats for the event stream (reduces UI work)
            threats = [d for d in detections if d["danger"] in ("med", "high") or d["class"] == "team"]

            # Limit payload to top-K largest threats
            threats.sort(key=lambda d: d["box"]["w"] * d["box"]["h"], reverse=True)
            threats = threats[:TOPK]

            # Build coarse signature so tiny motion doesn't count as change
            sig = _signature_of_threats(threats)
            changed = (sig != _last_sig)
            max_danger = "none"
            if threats:
                if any(d["danger"] == "high" or d["class"] == "team" for d in threats):
                    max_danger = "high"
                elif any(d["danger"] == "med" for d in threats):
                    max_danger = "med"

            # Heartbeat: always send at least every N seconds, otherwise only on "changed"
            now2 = time.time()
            should_send = changed or (now2 - _last_broadcast_ts >= EVENT_HEARTBEAT_SEC)

            if should_send:
                _last_sig = sig
                _last_broadcast_ts = now2
                event = {
                    "v": 1,
                    "stream_id": "front_cam",
                    "ts": int(now2 * 1000),
                    "detections": threats,   # ONLY threats → fewer boxes to draw
                    "meta": {
                        "fps": round(1.0 / max(now2 - prev_infer, 1e-6), 1),
                        "imgsz": IMG_SIZE,
                        "signature": sig,
                        "max_danger": max_danger,
                        "changed": changed,
                    },
                }
                broadcast(event)

threading.Thread(target=inference_loop, daemon=True).start()

# =========================
# Endpoints
# =========================
@app.websocket("/ws/detections")
async def ws_detections(ws: WebSocket):
    await ws.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=3)
    subscribers.add(q)
    try:
        await ws.send_text(json.dumps({"v": 1, "hello": True, "stream_id": "front_cam"}))
        while True:
            event = await q.get()
            while not q.empty():
                event = q.get_nowait()
            await ws.send_text(json.dumps(event))
    except WebSocketDisconnect:
        pass
    finally:
        subscribers.discard(q)

def mjpeg_generator():
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.putText(frame, "No camera frame", (30, FRAME_H // 2), font, 1.0, (0,255,255), 2, cv2.LINE_AA)
            time.sleep(0.02)
        # watermark
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{w}x{h}", (10, 25), font, 0.7, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{w}x{h}", (10, 25), font, 0.7, (255,255,255), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", frame, params)
        if not ok: continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.get("/preview.mjpg")
def preview():
    return StreamingResponse(mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/camera/status")
def camera_status():
    with cap_lock:
        opened = (cap is not None) and cap.isOpened()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
        fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0.0
    with frame_lock:
        ts = last_frame_ts
        has = latest_frame is not None
    return {
        "opened": opened, "index": cap_index, "backend": cap_backend_name,
        "width": w, "height": h, "fps_prop": fps,
        "frames_read": frames_read, "last_frame_ms": int(ts * 1000) if ts else None,
        "has_frame": has,
    }

@app.get("/camera/select")
def camera_select(
    index: int = Query(..., ge=0),
    backend: str = Query(DEFAULT_BACKEND),
    w: int = Query(FRAME_W, ge=160),
    h: int = Query(FRAME_H, ge=120),
    fps: int = Query(TARGET_FPS, ge=1)
):
    ok = camera_open(index, backend, w, h, fps)
    return {"ok": ok, "index": index, "backend": backend, "w": w, "h": h, "fps": fps}

@app.get("/camera/scan")
def camera_scan(max_index: int = 5):
    out = []
    for idx in range(max_index + 1):
        for be_name in ["dshow", "msmf"]:
            c, used = _open_camera(idx, be_name, FRAME_W, FRAME_H, TARGET_FPS)
            got = False; w = h = 0
            if c and c.isOpened():
                ok, frame = c.read()
                if ok and frame is not None:
                    got = True; h, w = frame.shape[:2]
                c.release()
            out.append({"index": idx, "backend": used, "opened": bool(c), "has_frame": got, "width": w, "height": h})
    return out

@app.get("/")
def root():
    return {
        "status": "ok",
        "ws": "/ws/detections",
        "preview": "/preview.mjpg",
        "camera_status": "/camera/status",
        "camera_select": "/camera/select?index=1&backend=msmf&w=1920&h=1080&fps=30",
        "camera_scan": "/camera/scan",
        "notes": "Env: CAM_* for camera; MODEL_NAME, IMG_SIZE, CONF_THRESH, IOU_THRESH, INFER_HZ, EVENT_HEARTBEAT_SEC, MOVE_THRESH_FRAC, TOPK, TEAM_*.",
    }
