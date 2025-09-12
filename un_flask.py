# flask_api.py
import os, math, time, platform, subprocess, json, threading, socket
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, request, jsonify

# =========================
# KONFIGURASI
# =========================
CFG = {
    "RTSP_URL":   "rtsp://admin:Damin3001@192.168.12.101:554/",
    "DETECT_ONNX":"yolov5s.onnx",                 # YOLOv5 COCO (clock=74, person=0)
    "DEPTH_ONNX": "midas_v21_small_256.onnx",     # opsional (MiDaS small 256)
    "IMG_SIZE":   640,
    "CONF_THRES": 0.25,
    "IOU_THRES":  0.45,
    "ALLOWED_IDS": {0, 74},  # person=0, clock=74
    "CLOCK_ID":   74,
    # Depth / occlusion
    "DEPTH_CLOSER_IS_HIGHER": True,
    "FRONT_MARGIN": 0.03,
    "IOU_PAIR_THRESHOLD": 0.05,
    "OCC_FRONT_THR": 0.60,
    "OCC_BACK_THR":  0.40,
    "CENTER_PATCH_FRAC": 0.20,
}

# ROUTING: map IP kamera (src) -> URL tujuan (dest)
# format: "IP_KAMERA": "http://IP_TUJUAN:PORT/path"
DEST_ROUTE = {
    # contoh: http://192.168.68.146:8080
    "192.168.12.101": "http://192.168.68.146:8080/collect",
    # "192.168.20.45": "http://192.168.20.10:9000/api/ingest",
}

# Fallback kalau IP kamera tidak ada di DEST_ROUTE
DEST_DEFAULT_URL = os.getenv("PUSH_URL", "").strip()     # ex: http://192.168.12.50:8080/collect
DEST_IP   = os.getenv("DEST_IP", "").strip()             # ex: 192.168.12.50
DEST_PORT = int(os.getenv("DEST_PORT", "8080"))
DEST_PATH = os.getenv("DEST_PATH", "/collect")

# Server Flask
LISTEN_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("FLASK_PORT", "5000"))

# Auto push (tanpa argumen CLI)
AUTO_PUSH = True
PUSH_INTERVAL_SEC = int(os.getenv("PUSH_INTERVAL_SEC", "300"))  # default 300s (5 menit)

# =========================
# UTIL: waktu & ping
# =========================
def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def get_host_from_rtsp(rtsp_url: str) -> str:
    p = urlparse(rtsp_url)
    if p.hostname: return p.hostname
    s = rtsp_url.split("://", 1)[-1]
    if '@' in s: s = s.split('@', 1)[1]
    if '/' in s: s = s.split('/', 1)[0]
    if ':' in s: s = s.split(':', 1)[0]
    return s

def interpret_ping_output(out: str, is_windows: bool):
    lo = (out or "").lower()
    if is_windows:
        if ("reply from" in lo) and ("unreachable" not in lo): return True, "OK"
        if "request timed out" in lo: return False, "RTO (Request timed out)"
        if "destination host unreachable" in lo: return False, "Destination host unreachable"
        if "could not find host" in lo: return False, "Host not found"
        if "general failure" in lo or "transmit failed" in lo: return False, "General failure"
        if "0% loss" in lo or "0 percent loss" in lo: return True, "OK"
        return False, "Ping failed"
    else:
        if "bytes from" in lo or ("icmp_seq" in lo and "ttl=" in lo): return True, "OK"
        if "destination host unreachable" in lo or "host unreachable" in lo: return False, "Destination host unreachable"
        if "100% packet loss" in lo or "request timeout" in lo: return False, "RTO (timeout)"
        if "unknown host" in lo or "name or service not known" in lo: return False, "Host not found"
        if "network is unreachable" in lo or "no route to host" in lo: return False, "Network unreachable"
        return False, "Ping failed"

def ping_host(host: str, count: int = 2, timeout_ms: int = 1000):
    is_windows = platform.system().lower().startswith('win')
    if is_windows:
        cmd = ["ping", "-n", str(count), "-w", str(timeout_ms), host]
    else:
        sec = max(1, int(math.ceil(timeout_ms / 1000.0)))
        cmd = ["ping", "-c", str(count), "-W", str(sec), host]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        out = (cp.stdout or "") + "\n" + (cp.stderr or "")
        ok, reason = interpret_ping_output(out, is_windows)
        # Fallback macOS lama (-W tidak ada)
        if not ok and "illegal option -- w" in out.lower():
            cmd = ["ping", "-c", str(count), host]
            cp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
            out = (cp.stdout or "") + "\n" + (cp.stderr or "")
            ok, reason = interpret_ping_output(out, is_windows)
        return ok, reason, out.strip()
    except Exception as e:
        return False, f"Ping error: {e}", ""

# =========================
# UTIL deteksi (YOLOv5 ONNX)
# =========================
def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if (w, h) != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def xywh2xyxy(x):
    y = np.empty_like(x)
    y[:,0] = x[:,0] - x[:,2]/2
    y[:,1] = x[:,1] - x[:,3]/2
    y[:,2] = x[:,0] + x[:,2]/2
    y[:,3] = x[:,1] + x[:,3]/2
    return y

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        inter = np.clip(xx2-xx1, 0, None) * np.clip(yy2-yy1, 0, None)
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_o = (boxes[idxs[1:],2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1])
        iou = inter / (area_i + area_o - inter + 1e-6)
        idxs = idxs[1:][iou < iou_thres]
    return np.array(keep, dtype=np.int32)

def scale_coords_from_letterboxed(coords_xyxy, r, pad, orig_shape):
    coords_xyxy[:, [0,2]] -= pad[0]
    coords_xyxy[:, [1,3]] -= pad[1]
    coords_xyxy[:, :4] /= r
    h,w = orig_shape[:2]
    coords_xyxy[:, 0::2] = np.clip(coords_xyxy[:, 0::2], 0, w)
    coords_xyxy[:, 1::2] = np.clip(coords_xyxy[:, 1::2], 0, h)
    return coords_xyxy

def parse_yolov5_output(out: np.ndarray) -> np.ndarray:
    arr = out
    if arr.ndim == 3: arr = arr[0]
    if arr.ndim == 2 and arr.shape[1] == 85: return arr.astype(np.float32, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 85 and arr.shape[1] != 85: return arr.T.astype(np.float32, copy=False)
    flat = arr.reshape(-1)
    n = flat.size // 85
    arr = flat[:n*85].reshape(n, 85)
    return arr.astype(np.float32, copy=False)

def bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    A = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    B = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter / (A + B - inter + 1e-6), inter, A, B

# =========================
# UTIL depth & occlusion
# =========================
def patch_median(img, cx, cy, size):
    h, w = img.shape[:2]
    x1 = max(0, int(cx - size//2)); y1 = max(0, int(cy - size//2))
    x2 = min(w, int(cx + size//2)); y2 = min(h, int(cy + size//2))
    if x2 <= x1 or y2 <= y1: return np.nan
    return float(np.median(img[y1:y2, x1:x2]))

def occlusion_score(d_norm, inter_box, margin=0.03, step=4, closer_is_higher=True, ref_clock_depth=None):
    x1, y1, x2, y2 = map(int, inter_box)
    if x2<=x1 or y2<=y1: return float('nan')
    patch = d_norm[y1:y2:step, x1:x2:step]
    if patch.size==0: return float('nan')
    if ref_clock_depth is None: ref_clock_depth = float(np.median(patch))
    return float((patch > (ref_clock_depth+margin)).mean() if closer_is_higher
                 else (patch < (ref_clock_depth-margin)).mean())

# =========================
# Analog clock reader
# =========================
UPSCALE_ROI     = 2.8
CANNY1, CANNY2  = 60, 140
HOUGH_THRESH    = 26
MAX_GAP         = 8
HOUR_MIN_FRAC   = 0.20
HOUR_MAX_FRAC   = 0.70
MINUTE_MIN_FRAC = 0.70
ANGLE_SEP_DEG   = 10.0
HOUR_FLIP_THR   = 120.0

def angle_deg_from_top_cw(center, pt):
    cx,cy = center; x,y = pt
    dx = x - cx; dy = cy - y
    ang = math.degrees(math.atan2(dx, dy))
    return ang + 360.0 if ang < 0 else ang

def estimate_center_by_circle(gray):
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=gray.shape[0]//4,
                               param1=120, param2=30,
                               minRadius=int(min(gray.shape[:2])*0.25),
                               maxRadius=int(min(gray.shape[:2])*0.55))
    if circles is not None and len(circles)>0:
        c = np.uint16(np.around(circles))[0,0]
        return (int(c[0]), int(c[1])), int(c[2])
    h,w = gray.shape[:2]
    return (w//2, h//2), min(w,h)//2

def _dist_point_to_segment(px, py, x1, y1, x2, y2):
    vx, vy = x2-x1, y2-y1
    wx, wy = px-x1, py-y1
    c1 = vx*wx + vy*wy
    if c1 <= 0: return math.hypot(px-x1, py-y1)
    c2 = vx*vx + vy*vy
    if c2 <= c1: return math.hypot(px-x2, py-y2)
    b = c1 / c2
    bx, by = x1 + b*vx, y1 + b*vy
    return math.hypot(px-bx, py-by)

def _choose_tip_endpoint(cx, cy, x1, y1, x2, y2, R):
    d1 = math.hypot(x1-cx, y1-cy)
    d2 = math.hypot(x2-cx, y2-cy)
    e1 = abs(R - d1); e2 = abs(R - d2)
    return ((x2,y2), d2) if e2 < e1 else ((x1,y1), d1)

def ang_diff(a, b):
    d = abs(a - b) % 360.0
    return d if d <= 180 else 360.0 - d

def read_analog_time(roi_bgr):
    big = cv2.resize(roi_bgr, None, fx=UPSCALE_ROI, fy=UPSCALE_ROI, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    gray  = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8,8)).apply(gray)

    center, R = estimate_center_by_circle(gray)
    cx, cy = center
    mask = np.zeros_like(gray, np.uint8)
    cv2.circle(mask, center, int(R*0.92), 255, -1)
    gmask = cv2.bitwise_and(gray, gray, mask=mask)
    black = cv2.morphologyEx(gmask, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
    black = cv2.normalize(black, None, 0, 255, cv2.NORM_MINMAX)
    edges = cv2.Canny(black, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=HOUGH_THRESH,
                            minLineLength=int(R*HOUR_MIN_FRAC), maxLineGap=MAX_GAP)
    if lines is None:
        return None, None
    cand_all, cand_hour, cand_minute = [], [], []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if _dist_point_to_segment(cx, cy, x1,y1,x2,y2) > 0.07*R: continue
        tip, _ = _choose_tip_endpoint(cx, cy, x1,y1,x2,y2, R)
        L   = max(math.hypot(x1-cx, y1-cy), math.hypot(x2-cx, y2-cy))
        Lfrac = L / R
        ang = angle_deg_from_top_cw((cx,cy), tip)
        item = {"len":L, "lfrac":Lfrac, "ang":ang}
        cand_all.append(item)
        if HOUR_MIN_FRAC <= Lfrac <= HOUR_MAX_FRAC: cand_hour.append(item)
        if Lfrac >= MINUTE_MIN_FRAC: cand_minute.append(item)
    if not cand_all:
        return None, None
    minute_line = max(cand_minute, key=lambda d: d["len"]) if cand_minute else max(cand_all, key=lambda d: d["len"])
    minute_angle = minute_line["ang"]
    pool = cand_hour if cand_hour else cand_all
    def hour_score(c): return 2.0*abs(c["lfrac"]-0.45) + 0.02*ang_diff(c["ang"], minute_angle)
    hour_line = min(pool, key=hour_score)
    if ang_diff(minute_angle, hour_line["ang"]) > HOUR_FLIP_THR:
        hour_line["ang"] = (hour_line["ang"] + 180.0) % 360.0
    minute = int(round(minute_angle / 6.0)) % 60
    hour_raw   = (hour_line["ang"] / 30.0) % 12.0
    hour_float = (hour_raw + (minute/60.0)) % 12.0
    hour       = int(hour_float) if int(hour_float)!=0 else 12
    return hour, minute

# =========================
# ROUTING UTIL
# =========================
def choose_push_url(src_cam_ip: str) -> str:
    """
    Pilih URL tujuan berdasarkan IP kamera (src).
    Prioritas: DEST_ROUTE[src] -> DEST_DEFAULT_URL -> (DEST_IP, DEST_PORT, DEST_PATH) -> ""
    """
    if src_cam_ip in DEST_ROUTE:
        return DEST_ROUTE[src_cam_ip]
    if DEST_DEFAULT_URL:
        return DEST_DEFAULT_URL
    if DEST_IP:
        return f"http://{DEST_IP}:{DEST_PORT}{DEST_PATH}"
    return ""

def route_meta(rtsp_url: str, push_url: str):
    """
    Metadata rute: IP asal kamera, IP pengirim (NIC lokal), IP tujuan.
    """
    src_cam_ip = get_host_from_rtsp(rtsp_url)
    dest_ip = urlparse(push_url).hostname if push_url else None
    sender_ip = None
    if dest_ip:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((dest_ip, 80))  # tidak kirim data; hanya untuk dapat IP lokal
            sender_ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass
    return {
        "source_camera_ip": src_cam_ip,
        "destination_ip": dest_ip,
        "sender_ip": sender_ip,
        "push_url": push_url or None
    }

# =========================
# Kelas utama (deteksi + pembacaan)
# =========================
class ClockReaderStream:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.det_sess = None
        self.det_in_name = None
        self.det_out_name = None
        self.USE_FP16_DET = False

        self.depth_sess = None
        self.depth_in_name = None
        self.DEPTH_W = 256
        self.DEPTH_H = 256

        self._load_models()

    def _load_models(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        # YOLO
        if Path(self.cfg["DETECT_ONNX"]).exists():
            try:
                self.det_sess = ort.InferenceSession(self.cfg["DETECT_ONNX"], providers=providers)
            except Exception:
                self.det_sess = ort.InferenceSession(self.cfg["DETECT_ONNX"], providers=['CPUExecutionProvider'])
            self.det_in_name  = self.det_sess.get_inputs()[0].name
            self.det_out_name = self.det_sess.get_outputs()[0].name
            self.USE_FP16_DET = 'float16' in self.det_sess.get_inputs()[0].type.lower()
        # MiDaS (opsional)
        if Path(self.cfg["DEPTH_ONNX"]).exists():
            try:
                self.depth_sess = ort.InferenceSession(self.cfg["DEPTH_ONNX"], providers=providers)
            except Exception:
                self.depth_sess = ort.InferenceSession(self.cfg["DEPTH_ONNX"], providers=['CPUExecutionProvider'])
            d_in_meta = self.depth_sess.get_inputs()[0]
            self.depth_in_name = d_in_meta.name
            shape = d_in_meta.shape
            if isinstance(shape,(list,tuple)) and len(shape)==4 and isinstance(shape[2],int) and isinstance(shape[3],int):
                self.DEPTH_H, self.DEPTH_W = shape[2], shape[3]

    def _detect(self, frame_bgr):
        img_lb, r, pad = letterbox(frame_bgr, (self.cfg["IMG_SIZE"], self.cfg["IMG_SIZE"]))
        x = img_lb[:, :, ::-1].transpose(2,0,1)
        x = np.ascontiguousarray(x, dtype=np.float16 if self.USE_FP16_DET else np.float32) / 255.0
        x = x[None, ...]
        out = self.det_sess.run([self.det_out_name], {self.det_in_name: x})[0]
        out = parse_yolov5_output(out)
        H, W = frame_bgr.shape[:2]

        persons, clocks = [], []
        if out.shape[1] >= 6:
            boxes   = out[:, :4]; obj_cf = out[:, 4:5]; cls_vec = out[:, 5:]
            cls_ids = cls_vec.argmax(axis=1); cls_cf = cls_vec.max(axis=1, keepdims=True)
            conf    = (obj_cf * cls_cf).squeeze(1)
            mask = (conf >= self.cfg["CONF_THRES"]) & np.isin(cls_ids, list(self.cfg["ALLOWED_IDS"]))
            boxes, conf, cls_ids = boxes[mask], conf[mask], cls_ids[mask]
            if boxes.size:
                boxes_xyxy = xywh2xyxy(boxes.copy())
                keep_all = []
                for cid in np.unique(cls_ids):
                    idx = np.where(cls_ids == cid)[0]
                    k = nms(boxes_xyxy[idx], conf[idx], self.cfg["IOU_THRES"])
                    keep_all.extend(idx[k])
                if keep_all:
                    keep_all = np.array(keep_all, dtype=np.int32)
                    boxes_xyxy = boxes_xyxy[keep_all]; conf = conf[keep_all]; cls_ids = cls_ids[keep_all]
                    boxes_xyxy = scale_coords_from_letterboxed(boxes_xyxy, r, pad, frame_bgr.shape)
                    for (x1,y1,x2,y2), cid in zip(boxes_xyxy, cls_ids):
                        bb = (int(x1),int(y1),int(x2),int(y2))
                        if cid == 0: persons.append(bb)
                        elif cid == self.cfg["CLOCK_ID"]: clocks.append(bb)
        return persons, clocks

    def _depth_map(self, frame_bgr):
        if self.depth_sess is None:
            return None
        d_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        d_inp = cv2.resize(d_rgb, (self.DEPTH_W, self.DEPTH_H), interpolation=cv2.INTER_LINEAR)
        d_inp = d_inp.transpose(2,0,1)[None, ...].astype(np.float32)/255.0
        d_map = self.depth_sess.run(None, {self.depth_in_name: d_inp})[0]
        if d_map.ndim==4: d_map = d_map[0,0]
        elif d_map.ndim==3: d_map = d_map[0]
        d_map = cv2.resize(d_map, (frame_bgr.shape[1], frame_bgr.shape[0]))
        mmin, mmax = np.percentile(d_map,1), np.percentile(d_map,99)
        d_norm = np.clip((d_map-mmin)/(mmax-mmin+1e-6), 0, 1)
        if not self.cfg["DEPTH_CLOSER_IS_HIGHER"]:
            d_norm = 1.0 - d_norm
        return d_norm

    def check_once(self, rtsp_url: str = None):
        rtsp_url = rtsp_url or self.cfg["RTSP_URL"]
        host = get_host_from_rtsp(rtsp_url)
        ping_ok, ping_reason, _ = ping_host(host, count=2, timeout_ms=1000)

        base = {
            "last_check_at": now_iso(),
            "rtsp_url": rtsp_url,
            "status_ip_cam": bool(ping_ok),
            "ping_reason": ping_reason,
            "model": {
                "yolo_loaded": self.det_sess is not None,
                "depth_loaded": self.depth_sess is not None
            },
            "detected_clock": False,
            "jumlah_jam": 0,
            "waktu": None,
            "status": None,  # TERBACA | TERHALANG | TIDAK TERDETEKSI | DISCONNECTED
            "occlusion_status": None
        }

        if not ping_ok:
            base["status"] = "DISCONNECTED"
            return base

        if self.det_sess is None:
            base["status"] = "TIDAK TERDETEKSI"
            base["ping_reason"] += " | YOLO model not loaded"
            return base

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            base["status"] = "DISCONNECTED"
            base["ping_reason"] += " | RTSP open failed"
            return base

        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            tries = 0; ok=False; frame=None
            while tries < 3:
                time.sleep(0.5)
                cap = cv2.VideoCapture(rtsp_url)
                if cap.isOpened():
                    ok, frame = cap.read()
                    if ok and frame is not None: break
                tries += 1
            if not ok or frame is None:
                base["status"] = "DISCONNECTED"
                return base

        H,W = frame.shape[:2]
        persons, clocks = self._detect(frame)
        base["jumlah_jam"] = int(len(clocks))
        if len(clocks) == 0:
            base["status"] = "TIDAK TERDETEKSI"
            cap.release(); return base

        d_norm = self._depth_map(frame) if (self.depth_sess is not None) else None

        x1,y1,x2,y2 = clocks[0]
        roi = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)].copy()

        occ_label = "TIDAK YAKIN"
        is_front = False
        if d_norm is not None:
            ccx, ccy = (x1+x2)//2, (y1+y2)//2
            csize = max(5, int(CFG["CENTER_PATCH_FRAC"] * min(x2-x1, y2-y1)))
            d_clock = patch_median(d_norm, ccx, ccy, csize)
            any_overlap, any_front, any_behind = False, False, False
            for pb in persons:
                iou, inter_area, area_c, area_p = bbox_iou((x1,y1,x2,y2), pb)
                if iou >= CFG["IOU_PAIR_THRESHOLD"]:
                    any_overlap = True
                    px1,py1,px2,py2 = pb
                    pcx,pcy = (px1+px2)//2, (py1+py2)//2
                    psize = max(5, int(CFG["CENTER_PATCH_FRAC"] * min(px2-px1, py2-py1)))
                    d_person = patch_median(d_norm, pcx, pcy, psize)
                    if np.isnan(d_person) or np.isnan(d_clock): continue
                    ix1,iy1 = max(x1,px1), max(y1,py1)
                    ix2,iy2 = min(x2,px2), min(y2,py2)
                    occ = occlusion_score(d_norm, (ix1,iy1,ix2,iy2),
                                          margin=CFG["FRONT_MARGIN"], step=4,
                                          closer_is_higher=CFG["DEPTH_CLOSER_IS_HIGHER"],
                                          ref_clock_depth=d_clock)
                    front  = (d_person > d_clock + CFG["FRONT_MARGIN"]) and (not np.isnan(occ) and occ >= CFG["OCC_FRONT_THR"])
                    behind = (d_clock  > d_person + CFG["FRONT_MARGIN"]) and (not np.isnan(occ) and occ <= CFG["OCC_BACK_THR"])
                    any_front  = any_front  or front
                    any_behind = any_behind or behind
            if any_front:
                occ_label = "TERTUTUP"; is_front = True
            elif any_overlap and any_behind:
                occ_label = "TIDAK TERTUTUP"
            elif not any_overlap:
                occ_label = "TIDAK TERTUTUP"
            else:
                occ_label = "TIDAK YAKIN"

        base["occlusion_status"] = occ_label

        if is_front:
            base["status"] = "TERHALANG"
            base["detected_clock"] = True
            base["waktu"] = None
            cap.release(); return base

        hh, mm = read_analog_time(roi)
        if hh is not None and mm is not None:
            base["status"] = "TERBACA"
            base["detected_clock"] = True
            base["waktu"] = f"{hh:02d}:{mm:02d}"
        else:
            base["status"] = "TIDAK TERDETEKSI"
            base["detected_clock"] = True
            base["waktu"] = None

        cap.release()
        return base

# =========================
# Flask app + endpoint
# =========================
app = Flask(__name__)
READER = ClockReaderStream(CFG)

@app.route("/send_data", methods=["POST"])
def send_data():
    payload = request.get_json(silent=True) or {}
    rtsp_url = payload.get("rtsp_url") or CFG["RTSP_URL"]
    result = READER.check_once(rtsp_url=rtsp_url)
    minimal = {
        "last_check_at": result["last_check_at"],
        "status": result["status"],                 # TERBACA / TERHALANG / TIDAK TERDETEKSI / DISCONNECTED
        "detected_clock": result["detected_clock"],
        "waktu": result["waktu"],
        "jumlah_jam": result["jumlah_jam"],
        "status_ip_cam": result["status_ip_cam"],
        "ping_reason": result["ping_reason"],
        "occlusion_status": result.get("occlusion_status"),
    }
    return jsonify(minimal), 200

@app.get("/health")
def health():
    return jsonify(ok=True, time=now_iso()), 200

# =========================
# Background sender (tiap 5 menit)
# =========================
def save_local(payload: dict, fname: str = "last_send_data.json"):
    try:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[AUTO] gagal simpan {fname}: {e}")

def periodic_sender():
    print(f"[AUTO] Auto-push aktif tiap {PUSH_INTERVAL_SEC}s. Routing berdasar IP kamera.")
    while True:
        t0 = time.time()
        try:
            src_cam_ip = get_host_from_rtsp(CFG["RTSP_URL"])
            push_url = choose_push_url(src_cam_ip)

            data = READER.check_once(rtsp_url=CFG["RTSP_URL"])
            data.update(route_meta(CFG["RTSP_URL"], push_url))

            if push_url:
                try:
                    resp = requests.post(push_url, json=data, timeout=15)
                    print(f"[AUTO] {data['last_check_at']} {src_cam_ip} -> {data['destination_ip']} [{resp.status_code}]")
                except Exception as e:
                    print(f"[AUTO] push error ke {push_url}: {e}; simpan lokal.")
                    save_local(data)
            else:
                print("[AUTO] tidak ada push_url yang cocok; simpan lokal.")
                save_local(data)
        except Exception as e:
            print(f"[AUTO] error: {e}")

        sleep_left = PUSH_INTERVAL_SEC - (time.time() - t0)
        time.sleep(max(1.0, sleep_left))

if __name__ == "__main__":
    if AUTO_PUSH:
        th = threading.Thread(target=periodic_sender, daemon=True)
        th.start()
    app.run(host=LISTEN_HOST, port=LISTEN_PORT, debug=False)
