# stream_clock_reader_yolo_depth_v3_3.py
# - Cek PING ke IP cam (dari RTSP URL). Jika gagal -> exit dgn pesan alasan (RTO, Unreachable, dll)
# - Jika PING OK -> RTSP + YOLOv5(ONNX) person/clock
# - (opsional) MiDaS depth untuk status occlusion
# - Analog clock reader v3.1 (fix jarum jam terbalik)
# - Diagonal hanya visual (ROI pembacaan dari frame_clean)
# - Toggle inset ROI debug: tekan 'i' ; keluar: 'q' / Esc

import os, math, time, platform, subprocess
from urllib.parse import urlparse
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# ================== KONFIGURASI ==================
RTSP_URL     = 'rtsp://admin:Damin3001@192.168.12.101:554/'
DETECT_ONNX  = r"yolov5s.onnx"                 # YOLOv5 COCO
DEPTH_ONNX   = r"midas_v21_small_256.onnx"     # opsional
CAM_PARAM_NPZ= r"camera_param_HVC.npz"         # opsional

IMG_SIZE    = 640
CONF_THRES  = 0.25
IOU_THRES   = 0.45
ALLOWED_IDS = {0, 74}        # person=0, clock=74
CLOCK_ID    = 74

# Depth / occlusion
DEPTH_CLOSER_IS_HIGHER = True
DEPTH_DEFAULT_SIZE     = 256
FRONT_MARGIN           = 0.03
IOU_PAIR_THRESHOLD     = 0.05
OCC_FRONT_THR          = 0.60
OCC_BACK_THR           = 0.40
CENTER_PATCH_FRAC      = 0.20
FULL_OCCL_COVER_THR    = 0.70
MISSING_KEEP_FRAMES    = 10

SHOW_INSET     = True
INSET_MAX_FRAC = 0.35

CLASS_NAMES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard',
    'cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase',
    'scissors','teddy bear','hair drier','toothbrush'
]

# ================== UTIL PING ==================
def get_host_from_rtsp(rtsp_url: str) -> str:
    """
    Ekstrak host/IP dari RTSP URL.
    Contoh: rtsp://user:pass@192.168.12.101:554/ -> 192.168.12.101
    """
    try:
        p = urlparse(rtsp_url)
        host = p.netloc
        if '@' in host:
            host = host.split('@', 1)[1]
        if ':' in host:
            host = host.split(':', 1)[0]
        return host
    except Exception:
        # fallback kasar
        s = rtsp_url.split('://', 1)[-1]
        if '@' in s: s = s.split('@', 1)[1]
        if '/' in s: s = s.split('/', 1)[0]
        if ':' in s: s = s.split(':', 1)[0]
        return s

def interpret_ping_output(out: str, is_windows: bool):
    lo = out.lower()
    if is_windows:
        if ("reply from" in lo) and ("unreachable" not in lo):
            return True, "OK"
        if "request timed out" in lo:
            return False, "RTO (Request timed out)"
        if "destination host unreachable" in lo:
            return False, "Destination host unreachable"
        if "general failure" in lo or "transmit failed" in lo:
            return False, "General failure"
        if "could not find host" in lo:
            return False, "Host not found"
        # fallback: cek loss
        if "0% loss" in lo or "0 percent loss" in lo:
            return True, "OK"
        return False, "Ping failed"
    else:
        if "bytes from" in lo or ("icmp_seq" in lo and "ttl=" in lo):
            if "0% packet loss" in lo or " 0.0% packet loss" in lo:
                return True, "OK"
            if "1 received" in lo or "2 received" in lo or "received, 0% packet loss" in lo:
                return True, "OK (partial)"
            return True, "OK"
        if "destination host unreachable" in lo or "host unreachable" in lo:
            return False, "Destination host unreachable"
        if "100% packet loss" in lo or "request timeout" in lo:
            return False, "RTO (timeout)"
        if "unknown host" in lo or "name or service not known" in lo:
            return False, "Host not found"
        if "network is unreachable" in lo or "no route to host" in lo:
            return False, "Network unreachable"
        return False, "Ping failed"

def ping_host(host: str, count: int = 2, timeout_ms: int = 1000):
    """
    Ping host. Return (ok:bool, reason:str, raw_output:str)
    - Windows: ping -n <count> -w <timeout_ms>
    - Linux/mac: ping -c <count> -W <timeout_sec>  (Linux), macOS juga menerima -W (ms) di beberapa versi;
      kita pakai detik agar aman lintas distro: ceil(timeout_ms/1000).
    """
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
        return ok, reason, out.strip()
    except Exception as e:
        return False, f"Ping error: {e}", ""

# ================== UTIL DETEKSI ==================
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

def bbox_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    A = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    B = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter / (A + B - inter + 1e-6), inter, A, B

def nms(boxes, scores, iou_thres):
    idxs = scores.argsort()[::-1]; keep=[]
    while idxs.size>0:
        i = idxs[0]; keep.append(i)
        if idxs.size==1: break
        xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[idxs[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[idxs[1:],3])
        inter = np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None)
        area_i = (boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1])
        area_o = (boxes[idxs[1:],2]-boxes[idxs[1:],0])*(boxes[idxs[1:],3]-boxes[idxs[1:],1])
        iou = inter/(area_i+area_o-inter+1e-6)
        idxs = idxs[1:][iou<iou_thres]
    return np.array(keep, dtype=np.int32)

def scale_coords_from_letterboxed(coords_xyxy, r, pad, orig_shape):
    coords_xyxy[:, [0,2]] -= pad[0]
    coords_xyxy[:, [1,3]] -= pad[1]
    coords_xyxy[:, :4] /= r
    h,w = orig_shape[:2]
    coords_xyxy[:, 0::2] = np.clip(coords_xyxy[:, 0::2], 0, w)
    coords_xyxy[:, 1::2] = np.clip(coords_xyxy[:, 1::2], 0, h)
    return coords_xyxy

# ================== UTIL DEPTH & OVERLAY ==================
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

def place_inset(overlay, inset, x=10, y=10, max_frac=0.35):
    H,W = overlay.shape[:2]
    if H<=1 or W<=1: return overlay
    max_w = max(1, int(W*max_frac)); max_h = max(1, int(H*max_frac))
    h,w = inset.shape[:2]
    s = min(max_w/w, max_h/h, 1.0)
    new_w, new_h = max(1,int(w*s)), max(1,int(h*s))
    resized = cv2.resize(inset, (new_w,new_h))
    x = min(max(0,x), W-new_w); y = min(max(0,y), H-new_h)
    overlay[y:y+new_h, x:x+new_w] = resized
    return overlay

# ================== ANALOG CLOCK READER (v3.1) ==================
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

    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.circle(dbg, center, 4, (0,0,255), -1); cv2.circle(dbg, center, int(R), (120,120,120), 1)
    if lines is None:
        return None, None, cv2.resize(dbg, (roi_bgr.shape[1], roi_bgr.shape[0])), "no lines"

    cand_all, cand_hour, cand_minute = [], [], []
    for l in lines:
        x1,y1,x2,y2 = l[0]
        if _dist_point_to_segment(cx, cy, x1,y1,x2,y2) > 0.07*R: continue
        tip, _ = _choose_tip_endpoint(cx, cy, x1,y1,x2,y2, R)
        other = (x2,y2) if tip==(x1,y1) else (x1,y1)
        L   = max(math.hypot(x1-cx, y1-cy), math.hypot(x2-cx, y2-cy))
        Lfrac = L / R
        ang = angle_deg_from_top_cw((cx,cy), tip)
        item = {"p1":(x1,y1), "p2":(x2,y2), "tip":tip, "other":other, "len":L, "lfrac":Lfrac, "ang":ang}
        cand_all.append(item)
        if HOUR_MIN_FRAC <= Lfrac <= HOUR_MAX_FRAC: cand_hour.append(item)
        if Lfrac >= MINUTE_MIN_FRAC: cand_minute.append(item)
    if not cand_all:
        return None, None, cv2.resize(dbg, (roi_bgr.shape[1], roi_bgr.shape[0])), "no radial lines"

    minute_line = max(cand_minute, key=lambda d: d["len"]) if cand_minute else max(cand_all, key=lambda d: d["len"])
    minute_angle = minute_line["ang"]
    pool = cand_hour if cand_hour else cand_all
    def hour_score(c): return 2.0*abs(c["lfrac"]-0.45) + 0.02*ang_diff(c["ang"], minute_angle)
    hour_line = min(pool, key=hour_score)
    if ang_diff(minute_angle, hour_line["ang"]) > HOUR_FLIP_THR:
        hour_line["ang"] = (hour_line["ang"] + 180.0) % 360.0
        hour_line["tip"], hour_line["other"] = hour_line["other"], hour_line["tip"]

    # debug draw
    cv2.line(dbg, minute_line["other"], minute_line["tip"], (0,255,0), 2)
    cv2.circle(dbg, minute_line["tip"], 5, (0,255,0), -1)
    cv2.line(dbg, hour_line["other"], hour_line["tip"], (255,0,255), 3)
    cv2.circle(dbg, hour_line["tip"], 5, (255,0,255), -1)

    minute = int(round(minute_angle / 6.0)) % 60
    hour_raw   = (hour_line["ang"] / 30.0) % 12.0
    hour_float = (hour_raw + (minute/60.0)) % 12.0
    hour       = int(hour_float) if int(hour_float)!=0 else 12
    dbg = cv2.resize(dbg, (roi_bgr.shape[1], roi_bgr.shape[0]))
    return hour, minute, dbg, None

# ================== ENTRYPOINT ==================
def main():
    # 0) Cek PING ke IP cam lebih dulu
    host = get_host_from_rtsp(RTSP_URL)
    ok, reason, pong = ping_host(host, count=2, timeout_ms=1000)
    print(f"[PING] Host: {host} -> {reason}")
    if pong: print(pong)
    if not ok:
        print(f"[ERROR] IP cam tidak connect: {reason}. Stop.")
        return

    # 1) Siapkan sesi model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        det_sess = ort.InferenceSession(DETECT_ONNX, providers=providers)
    except Exception as e:
        print("[WARN] CUDA EP deteksi off, fallback CPU:", e)
        det_sess = ort.InferenceSession(DETECT_ONNX, providers=['CPUExecutionProvider'])
    det_in_name = det_sess.get_inputs()[0].name
    det_out_name= det_sess.get_outputs()[0].name
    USE_FP16_DET = 'float16' in det_sess.get_inputs()[0].type.lower()

    # Depth (opsional)
    use_depth = Path(DEPTH_ONNX).exists()
    if use_depth:
        try:
            depth_sess = ort.InferenceSession(DEPTH_ONNX, providers=providers)
        except Exception as e:
            print("[WARN] Depth CUDA EP off, fallback CPU:", e)
            depth_sess = ort.InferenceSession(DEPTH_ONNX, providers=['CPUExecutionProvider'])
        d_in_meta = depth_sess.get_inputs()[0]
        d_in_name = d_in_meta.name
        shape = d_in_meta.shape
        if isinstance(shape,(list,tuple)) and len(shape)==4 and isinstance(shape[2],int) and isinstance(shape[3],int):
            DEPTH_H, DEPTH_W = shape[2], shape[3]
        else:
            DEPTH_H = DEPTH_W = DEPTH_DEFAULT_SIZE
    else:
        print("[INFO] Depth model tidak ditemukan; status occlusion = TIDAK YAKIN.")

    # 2) Buka RTSP
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("[ERROR] RTSP tidak bisa dibuka walau ping OK. Periksa kredensial/port/stream.")
        return
    ok, frame0 = cap.read()
    if not ok:
        print("[ERROR] Gagal membaca frame awal dari RTSP.")
        cap.release(); return
    H0, W0 = frame0.shape[:2]

    # 3) (opsional) undistort
    use_undistort = False
    if Path(CAM_PARAM_NPZ).exists():
        try:
            with np.load(CAM_PARAM_NPZ) as data:
                cm, dist = data['camera_matrix'], data['dist_coeffs']
            new_cm, _ = cv2.getOptimalNewCameraMatrix(cm, dist, (W0, H0), alpha=0)
            mapx, mapy = cv2.initUndistortRectifyMap(cm, dist, None, new_cm, (W0, H0), cv2.CV_32FC1)
            use_undistort = True
            print("[INFO] Undistort aktif.")
        except Exception as e:
            print("[WARN] Gagal inisialisasi undistort:", e)

    # 4) Loop
    prev_clocks = []
    frame_idx = 0
    fps_avg, t0 = 0.0, time.time()
    WIN = "RTSP • YOLO + Depth + Analog Reader (i: inset ROI, q: quit)"
    SHOW_INSET = True

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] frame kosong; stop.")
            break
        frame_idx += 1
        frame_u = cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR) if use_undistort else frame
        frame_clean = frame_u.copy()  # ← sumber ROI pembacaan (tanpa overlay)
        H, W = frame_u.shape[:2]

        # ---------- YOLO ----------
        img_lb, r, pad = letterbox(frame_u, (IMG_SIZE, IMG_SIZE))
        det_in = img_lb[:, :, ::-1].transpose(2,0,1)
        det_in = np.ascontiguousarray(det_in, dtype=np.float16 if USE_FP16_DET else np.float32)/255.0
        det_in = det_in[None, ...]
        out = det_sess.run([det_out_name], {det_in_name: det_in})[0]
        if out.ndim==3: out = out[0]
        if out.shape[0]==85 and out.shape[1]!=85: out = out.T
        if out.dtype!=np.float32: out = out.astype(np.float32, copy=False)

        persons, clocks = [], []
        if out.shape[1] >= 6:
            boxes   = out[:, :4]; obj_cf = out[:, 4:5]; cls_vec = out[:, 5:]
            cls_ids = cls_vec.argmax(axis=1); cls_cf = cls_vec.max(axis=1, keepdims=True)
            conf    = (obj_cf * cls_cf).squeeze(1)
            mask = (conf >= CONF_THRES) & np.isin(cls_ids, list(ALLOWED_IDS))
            boxes, conf, cls_ids = boxes[mask], conf[mask], cls_ids[mask]
            if boxes.size:
                boxes_xyxy = xywh2xyxy(boxes.copy())
                keep_all = []
                for cid in np.unique(cls_ids):
                    idx = np.where(cls_ids == cid)[0]
                    k = nms(boxes_xyxy[idx], conf[idx], IOU_THRES)
                    keep_all.extend(idx[k])
                keep_all = np.array(keep_all, dtype=np.int32) if len(keep_all) else np.array([], dtype=np.int32)
                if keep_all.size:
                    boxes_xyxy = boxes_xyxy[keep_all]; conf = conf[keep_all]; cls_ids = cls_ids[keep_all]
                    boxes_xyxy = scale_coords_from_letterboxed(boxes_xyxy, r, pad, frame_u.shape)
                    for (x1,y1,x2,y2), c, cid in zip(boxes_xyxy, conf, cls_ids):
                        bb = (int(x1),int(y1),int(x2),int(y2))
                        if cid == 0: persons.append(bb)
                        elif cid == CLOCK_ID: clocks.append(bb)

        # ---------- Depth (opsional) ----------
        d_norm = None
        if Path(DEPTH_ONNX).exists() and (clocks or persons or prev_clocks):
            d_rgb = cv2.cvtColor(frame_u, cv2.COLOR_BGR2RGB)
            d_inp = cv2.resize(d_rgb, (DEPTH_W, DEPTH_H), interpolation=cv2.INTER_LINEAR)
            d_inp = d_inp.transpose(2,0,1)[None, ...].astype(np.float32)/255.0
            d_map = depth_sess.run(None, {d_in_name: d_inp})[0]
            if d_map.ndim==4: d_map = d_map[0,0]
            elif d_map.ndim==3: d_map = d_map[0]
            d_map = cv2.resize(d_map, (W, H))
            mmin, mmax = np.percentile(d_map,1), np.percentile(d_map,99)
            d_norm = np.clip((d_map-mmin)/(mmax-mmin+1e-6), 0, 1)
            if not DEPTH_CLOSER_IS_HIGHER: d_norm = 1.0 - d_norm

        # ---------- Proses clock ----------
        current_clock_records = []
        inset_y = 10
        for cb in clocks:
            x1,y1,x2,y2 = cb
            # 1) ROI dari frame_clean (tanpa overlay)
            roi = frame_clean[max(0,y1):min(H,y2), max(0,x1):min(W,x2)].copy()
            hour, minute, roi_dbg, err = read_analog_time(roi)
            time_txt = f"{hour:02d}:{minute:02d}" if err is None and hour is not None and minute is not None else "??:??"

            # 2) STATUS OCCLUSION (depth)
            status, color = "TIDAK YAKIN", (0,255,255)
            if d_norm is not None:
                ccx, ccy = (x1+x2)//2, (y1+y2)//2
                csize = max(5, int(CENTER_PATCH_FRAC * min(x2-x1, y2-y1)))
                d_clock = patch_median(d_norm, ccx, ccy, csize)
                any_overlap = False; any_front = False; any_behind = False
                for pb in persons:
                    iou, inter_area, area_c, area_p = bbox_iou(cb, pb)
                    if iou >= IOU_PAIR_THRESHOLD:
                        any_overlap = True
                        px1,py1,px2,py2 = pb
                        pcx,pcy = (px1+px2)//2, (py1+py2)//2
                        psize = max(5, int(CENTER_PATCH_FRAC * min(px2-px1, py2-py1)))
                        d_person = patch_median(d_norm, pcx, pcy, psize)
                        if np.isnan(d_person) or np.isnan(d_clock): continue
                        ix1,iy1 = max(x1,px1), max(y1,py1)
                        ix2,iy2 = min(x2,px2), min(y2,py2)
                        occ = occlusion_score(d_norm, (ix1,iy1,ix2,iy2),
                                              margin=FRONT_MARGIN, step=4,
                                              closer_is_higher=DEPTH_CLOSER_IS_HIGHER,
                                              ref_clock_depth=d_clock)
                        front  = (d_person > d_clock + FRONT_MARGIN) and (not np.isnan(occ) and occ >= OCC_FRONT_THR)
                        behind = (d_clock  > d_person + FRONT_MARGIN) and (not np.isnan(occ) and occ <= OCC_BACK_THR)
                        any_front  = any_front  or front
                        any_behind = any_behind or behind
                if any_front:
                    status, color = "TERTUTUP", (0,0,255)
                elif any_overlap and any_behind:
                    status, color = "TIDAK TERTUTUP", (0,255,0)
                elif not any_overlap:
                    status, color = "TIDAK TERTUTUP", (0,255,0)
                else:
                    status, color = "TIDAK YAKIN", (0,255,255)
                current_clock_records.append({'bbox': cb, 'depth': d_clock, 'frame': frame_idx})

            # 3) BARU GAMBARKAN OVERLAY (bbox + diagonal + teks) DI frame_u
            cv2.rectangle(frame_u, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.line(frame_u, (x1,y1), (x2,y2), (255,0,0), 1, cv2.LINE_AA)  # diagonal visual
            cv2.line(frame_u, (x1,y2), (x2,y1), (255,0,0), 1, cv2.LINE_AA)  # diagonal visual
            cv2.putText(frame_u, time_txt, (x1+6, min(H-6, y2-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,255,255), 4, cv2.LINE_AA)

            label = f"clock | {status} | {time_txt}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(y1, lh + 6)
            cv2.rectangle(frame_u, (x1, y_text - lh - 6), (x1 + lw, y_text), color, -1)
            cv2.putText(frame_u, label, (x1, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

            # 4) INSET ROI DEBUG (opsional)
            if SHOW_INSET and roi_dbg is not None:
                frame_u = place_inset(frame_u, roi_dbg, x=10, y=inset_y, max_frac=INSET_MAX_FRAC)
                inset_y += int(min(H, W) * INSET_MAX_FRAC) + 10

        # Jejak clock + full-occlusion saat jam hilang
        if current_clock_records:
            prev_clocks = current_clock_records
        else:
            if d_norm is not None and len(prev_clocks):
                for rec in prev_clocks:
                    if frame_idx - rec['frame'] > MISSING_KEEP_FRAMES: continue
                    (x1,y1,x2,y2) = rec['bbox']; d_clock_prev = rec['depth']
                    occluded_full = False
                    for pb in persons:
                        iou, inter_area, area_c, area_p = bbox_iou(rec['bbox'], pb)
                        cover = inter_area / (area_c + 1e-6)
                        if cover >= FULL_OCCL_COVER_THR:
                            ix1,iy1 = max(x1,pb[0]), max(y1,pb[1])
                            ix2,iy2 = min(x2,pb[2]), min(y2,pb[3])
                            occ = occlusion_score(d_norm, (ix1,iy1,ix2,iy2),
                                                  margin=FRONT_MARGIN, step=4,
                                                  closer_is_higher=DEPTH_CLOSER_IS_HIGHER,
                                                  ref_clock_depth=d_clock_prev)
                            if not np.isnan(occ) and occ >= OCC_FRONT_THR:
                                occluded_full = True; break
                    if occluded_full:
                        cv2.rectangle(frame_u, (x1,y1), (x2,y2), (0,0,255), 2)
                        cv2.line(frame_u, (x1,y1), (x2,y2), (0,0,255), 1, cv2.LINE_AA)
                        cv2.line(frame_u, (x1,y2), (x2,y1), (0,0,255), 1, cv2.LINE_AA)
                        txt = "JAM: TIDAK TERDETEKSI (tertutup)"
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        yt = max(y1 - 8, th + 8)
                        cv2.rectangle(frame_u, (x1, yt - th - 6), (x1 + tw, yt), (0,0,255), -1)
                        cv2.putText(frame_u, txt, (x1, yt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)
            prev_clocks = [rec for rec in prev_clocks if frame_idx - rec['frame'] <= MISSING_KEEP_FRAMES]

        # Gambar persons (terakhir)
        for (x1,y1,x2,y2) in persons:
            cv2.rectangle(frame_u, (x1,y1), (x2,y2), (0,255,0), 2)
            (tw, th), _ = cv2.getTextSize("person", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_text = max(y1, th + 4)
            cv2.rectangle(frame_u, (x1, y_text - th - 4), (x1 + tw, y_text), (0,255,0), -1)
            cv2.putText(frame_u, "person", (x1, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - t0, 1e-6)
        fps_avg = fps if fps_avg == 0 else (0.9*fps_avg + 0.1*fps)
        t0 = now
        cv2.putText(frame_u, f"FPS: {fps_avg:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(WIN, frame_u)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27): break
        if key == ord('i'): SHOW_INSET = not SHOW_INSET

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()