# app.py — RTSP "clock" detector (YOLOv5 ONNX) + Flask status + periodic push
# - Ping kamera (Windows/Linux) → alasan: OK / RTO / Destination host unreachable
# - Ambil 1 frame RTSP, deteksi COCO class "clock" (default id=74)
# - Kirim status tiap INTERVAL_SEC (default 120 detik) ke WEBHOOK_URL (opsional)
# - Endpoint: GET /status, POST /force-check, GET /health

from __future__ import annotations

import os
import re
import cv2
import time
import json
import threading
import platform
import subprocess
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from flask import Flask, jsonify, request

# -------- Optional imports (tidak fatal bila tidak ada) --------
try:
    import onnxruntime as ort
except Exception:
    ort = None  # type: ignore
try:
    import requests
except Exception:
    requests = None  # type: ignore

# ============================ CONFIG ============================

@dataclass
class Config:
    RTSP_URL: str = os.getenv("RTSP_URL", "rtsp://admin:Damin3001@192.168.12.101:554/")
    DETECT_ONNX: str = os.getenv("DETECT_ONNX", r"yolov5s.onnx")
    CAM_PARAM_NPZ: str = os.getenv("CAM_PARAM_NPZ", r"camera_param_HVC.npz")  # "" jika tidak ada
    WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
    INTERVAL_SEC: int = int(os.getenv("INTERVAL_SEC", "120"))  # 2 menit
    CONF_THRESH: float = float(os.getenv("CONF_THRESH", "0.35"))
    IOU_THRESH: float = float(os.getenv("IOU_THRESH", "0.50"))
    IMG_SIZE: int = int(os.getenv("IMG_SIZE", "640"))
    CLOCK_CLASS_ID: int = int(os.getenv("CLOCK_CLASS_ID", "74"))  # COCO "clock"=74
    WARMUP_READS: int = int(os.getenv("WARMUP_READS", "3"))
    FORCE_FP16: bool = os.getenv("FORCE_FP16", "0").lower() in ("1", "true", "yes")

CFG = Config()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("clock-detector")

app = Flask(__name__)

STATE: Dict[str, Any] = {
    "last_check_at": None,
    "camera_reachable": False,
    "ping_reason": "",
    "rtsp_ok": False,
    "detected": False,
    "detections": [],
    "model_loaded": False,
    "error": "",
    "debug": {},
}
_state_lock = threading.Lock()

# ============================ UTILS ============================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def get_ip_from_rtsp(rtsp: str) -> str:
    try:
        u = urlparse(rtsp)
        return u.hostname or ""
    except Exception:
        return ""

def ping_host(host: str, count: int = 2, timeout_ms: int = 1000) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Cross-platform ping. Return (ok, reason, summary).
    reason: "RTO" | "Destination host unreachable" | "Ping failed" | ""
    """
    if not host:
        return False, "No host in RTSP URL", {}

    system = platform.system().lower()
    if "windows" in system:
        cmd = ["ping", "-n", str(count), "-w", str(timeout_ms), host]
    else:
        cmd = ["ping", "-c", str(count), "-W", str(max(1, timeout_ms // 1000)), host]

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(2, count * (timeout_ms / 1000 + 1)),
        )
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        ok = (p.returncode == 0)
        reason = ""
        if re.search(r"Request timed out", out, re.I):
            reason = "RTO"
        elif re.search(r"Destination host unreachable", out, re.I):
            reason = "Destination host unreachable"
        elif not ok:
            reason = "Ping failed"
        return ok, reason, {"raw": out[:1200]}
    except Exception as e:
        return False, f"Ping error: {e}", {}

def load_camera_params(npz_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not CFG.CAM_PARAM_NPZ or not os.path.isfile(npz_path):
        return None, None
    try:
        data = np.load(npz_path)
        mtx = data.get("mtx", None)
        dist = data.get("dist", None)
        if mtx is not None and dist is not None:
            log.info("Camera params loaded (mtx, dist).")
            return mtx, dist
    except Exception as e:
        log.warning(f"Load camera params failed: {e}")
    return None, None

def undistort_if_needed(frame: np.ndarray, mtx, dist) -> np.ndarray:
    if mtx is None or dist is None:
        return frame
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w2, h2 = roi
    if w2 > 0 and h2 > 0:
        dst = dst[y : y + h2, x : x + w2]
    return dst

def letterbox(im: np.ndarray, new_shape: int | Tuple[int, int], color=(114, 114, 114)):
    """
    Resize + pad image to fit new_shape while keeping aspect ratio.
    Return: image, ratio, (dw, dh)
    """
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> List[int]:
    """Greedy NMS — boxes: Nx4 (x1,y1,x2,y2)"""
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]
    keep: List[int] = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idxs[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idxs[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idxs[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idxs[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_o = (boxes[idxs[1:], 2] - boxes[idxs[1:], 0]) * (boxes[idxs[1:], 3] - boxes[idxs[1:], 1])
        ovr = inter / (area_i + area_o - inter + 1e-6)
        idxs = idxs[1:][ovr <= iou_thresh]
    return keep

# ============================ MODEL ============================

class YoloV5Onnx:
    """
    YOLOv5 ONNX runner
    - Auto-cast input to FP16/FP32 (sesuai model) + opsi FORCE_FP16
    - Output shape fleksibel (single head / multi-head) → di-flatten ke (N, 85)
    """

    def __init__(self, onnx_path: str, img_size: int, conf_th: float, iou_th: float, clock_cls: int):
        if ort is None:
            raise RuntimeError("onnxruntime belum terpasang. `pip install onnxruntime`")

        self.sess = self._make_session(onnx_path)
        self.input = self.sess.get_inputs()[0]
        self.input_name = self.input.name

        # Robust input type detect
        itype = getattr(self.input, "type", None)
        if not itype or not isinstance(itype, str):
            itype = str(self.input)
        itype_lower = itype.lower()

        self.input_type_str = itype
        self.expect_fp16 = ("float16" in itype_lower)

        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.img_size = img_size
        self.conf_th = conf_th
        self.iou_th = iou_th
        self.clock_cls = clock_cls

        log.info(
            f"[ONNX] input_name={self.input_name} input_type={self.input_type_str} "
            f"expect_fp16={self.expect_fp16} FORCE_FP16={CFG.FORCE_FP16}"
        )

    def _make_session(self, path: str):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        for prov in (["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"]):
            try:
                s = ort.InferenceSession(path, sess_options=so, providers=prov)
                log.info(f"ONNX providers={s.get_providers()}")
                return s
            except Exception as e:
                log.warning(f"Provider {prov} failed: {e}")
        raise RuntimeError("Tidak bisa membuat InferenceSession untuk ONNX.")

    def _prepare_input(self, bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float], Tuple[int, int], str]:
        im, ratio, dwdh = letterbox(bgr, self.img_size)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]  # 1x3xHxW

        need_fp16 = self.expect_fp16 or CFG.FORCE_FP16
        x = x.astype(np.float16) if need_fp16 else x.astype(np.float32)
        fed_dtype = "float16" if need_fp16 else "float32"
        return x, ratio, dwdh, bgr.shape[:2], fed_dtype

    @staticmethod
    def _flatten(outs: List[np.ndarray]) -> np.ndarray:
        """Gabungkan semua head menjadi (N, 85)."""
        arrs = []
        for o in outs:
            a = o
            if isinstance(a, list):
                a = np.array(a)
            if a.ndim >= 2 and a.shape[0] == 1:
                a = np.squeeze(a, axis=0)  # drop batch
            a = a.reshape(-1, a.shape[-1])
            arrs.append(a)
        return np.concatenate(arrs, axis=0) if len(arrs) > 1 else arrs[0]

    def detect_clock(self, bgr: np.ndarray) -> Tuple[bool, List[Dict[str, Any]], str]:
        x, ratio, dwdh, (H0, W0), fed_dtype = self._prepare_input(bgr)
        outs = self.sess.run(self.output_names, {self.input_name: x})
        pred = self._flatten(outs)  # (M, 85): [cx, cy, w, h, obj, cls0..cls79]

        boxes: List[List[float]] = []
        scores: List[float] = []
        classes: List[int] = []

        for row in pred:
            obj = row[4]
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            conf = float(obj * cls_scores[cls_id])
            if conf < self.conf_th:
                continue
            cx, cy, w, h = row[:4]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            classes.append(cls_id)

        if not boxes:
            return False, [], fed_dtype

        boxes_np = np.array(boxes, dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32)
        classes_np = np.array(classes, dtype=np.int32)

        keep = nms(boxes_np, scores_np, self.iou_th)
        boxes_np = boxes_np[keep]
        scores_np = scores_np[keep]
        classes_np = classes_np[keep]

        # Un-letterbox ke koordinat asli
        dw, dh = dwdh
        boxes_np[:, [0, 2]] -= dw
        boxes_np[:, [1, 3]] -= dh
        boxes_np[:, [0, 2]] /= ratio[0]
        boxes_np[:, [1, 3]] /= ratio[1]
        boxes_np[:, 0::2] = np.clip(boxes_np[:, 0::2], 0, W0)
        boxes_np[:, 1::2] = np.clip(boxes_np[:, 1::2], 0, H0)

        dets: List[Dict[str, Any]] = []
        has_clock = False
        for (x1, y1, x2, y2), sc, cls in zip(boxes_np, scores_np, classes_np):
            is_clock = (int(cls) == self.clock_cls)
            has_clock = has_clock or is_clock
            dets.append(
                {
                    "cls": int(cls),
                    "is_clock": bool(is_clock),
                    "score": float(round(float(sc), 4)),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        return has_clock, dets, fed_dtype

# ============================ PIPELINE ============================

def open_rtsp(rtsp_url: str, timeout_ms: int = 5000) -> Optional[cv2.VideoCapture]:
    """
    Buka RTSP dengan backend FFMPEG. Lakukan warm-up beberapa read untuk menghindari frame awal korup.
    """
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    t0 = time.time()
    ok = False
    for _ in range(max(1, CFG.WARMUP_READS)):
        ok, _frame = cap.read()
        if ok:
            break
        if (time.time() - t0) * 1000 > timeout_ms:
            break
    return cap if ok else None

def check_once(model: Optional[YoloV5Onnx], mtx=None, dist=None) -> Dict[str, Any]:
    waktu = now_iso()
    detected_clock = False
    jumlah_jam = 0

    # 1) Ping kamera
    host = get_ip_from_rtsp(CFG.RTSP_URL)
    ping_ok, _reason, _ = ping_host(host, count=2, timeout_ms=1000)
    if not ping_ok:
        result = {"waktu": waktu, "detected_clock": False, "jumlah_jam": 0}
        with _state_lock:
            STATE.update(result)  # update hanya kunci sederhana
        return result

    # 2) RTSP open + ambil 1 frame
    cap = open_rtsp(CFG.RTSP_URL)
    if cap is None or not cap.isOpened():
        result = {"waktu": waktu, "detected_clock": False, "jumlah_jam": 0}
        with _state_lock:
            STATE.update(result)
        return result

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        result = {"waktu": waktu, "detected_clock": False, "jumlah_jam": 0}
        with _state_lock:
            STATE.update(result)
        return result

    # 3) Undistort (opsional)
    frame = undistort_if_needed(frame, mtx, dist)

    # 4) Inference
    if model is None:
        result = {"waktu": waktu, "detected_clock": False, "jumlah_jam": 0}
        with _state_lock:
            STATE.update(result)
        return result

    try:
        has_clock, dets, _fed_dtype = model.detect_clock(frame)
        detected_clock = bool(has_clock)
        jumlah_jam = sum(1 for d in dets if d.get("is_clock", False))
    except Exception:
        detected_clock = False
        jumlah_jam = 0

    result = {"waktu": waktu, "detected_clock": detected_clock, "jumlah_jam": jumlah_jam}
    with _state_lock:
        STATE.update(result)  # simpan hasil sederhana ke STATE
    return result

def push_status(status: Dict[str, Any]) -> None:
    if not CFG.WEBHOOK_URL or requests is None:
        return
    try:
        r = requests.post(CFG.WEBHOOK_URL, json=status, timeout=5)
        log.info(f"Webhook POST {r.status_code}")
    except Exception as e:
        log.warning(f"Webhook send failed: {e}")

def scheduler_loop(model: Optional[YoloV5Onnx], mtx, dist) -> None:
    while True:
        try:
            status = check_once(model, mtx, dist)
            push_status(status)
        except Exception as e:
            log.exception(e)
        time.sleep(max(5, CFG.INTERVAL_SEC))

# ============================ BOOT ============================

def init_model() -> Optional[YoloV5Onnx]:
    if not os.path.isfile(CFG.DETECT_ONNX):
        log.warning(f"ONNX not found: {CFG.DETECT_ONNX}")
        return None
    try:
        mdl = YoloV5Onnx(
            CFG.DETECT_ONNX,
            CFG.IMG_SIZE,
            CFG.CONF_THRESH,
            CFG.IOU_THRESH,
            CFG.CLOCK_CLASS_ID,
        )
        log.info("ONNX model loaded.")
        return mdl
    except Exception as e:
        log.error(f"Load model error: {e}")
        return None

MODEL = init_model()
if MODEL is not None:
    with _state_lock:
        STATE["model_loaded"] = True

MTX, DIST = load_camera_params(CFG.CAM_PARAM_NPZ)

_thread = threading.Thread(target=scheduler_loop, args=(MODEL, MTX, DIST), daemon=True)
_thread.start()

# ============================ ROUTES ============================

@app.get("/status")
def status():
    with _state_lock:
        return jsonify(STATE)

@app.post("/force-check")
def force_check():
    status = check_once(MODEL, MTX, DIST)
    push_status(status)
    return jsonify(status)

@app.get("/health")
def health():
    return jsonify({"ok": True, "time": now_iso()})

@app.get("/")
def root():
    return jsonify(
        {
            "service": "clock-detector",
            "interval_sec": CFG.INTERVAL_SEC,
            "model_loaded": STATE["model_loaded"],
            "endpoints": ["/status", "/force-check", "/health"],
        }
    )

if __name__ == "__main__":
    # Env opsional:
    #   RTSP_URL, DETECT_ONNX, CAM_PARAM_NPZ, WEBHOOK_URL,
    #   INTERVAL_SEC, CONF_THRESH, IOU_THRESH, IMG_SIZE, CLOCK_CLASS_ID,
    #   WARMUP_READS, FORCE_FP16
    app.run(host="0.0.0.0", port=5000)
