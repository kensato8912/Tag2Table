"""
參數驅動型 AI 肢體收割引擎
根據 data/crop_parts.json 配置，統一處理 Hands / Pose / FaceMesh 裁切。
支援資料夾或 .zip 壓縮檔來源（zip 內影像以 Byte 流 imdecode，不產生暫存檔）。
"""
import gc
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Generator

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
# 輸出路徑鎖定在專案 base_path 底下，避免噴到 StabilityMatrix 根目錄
_CROP_OUTPUT_DEFAULT = os.path.abspath(str(_PROJECT_ROOT / "data" / "crop_output"))
CROP_OUTPUT_ROOT = os.path.abspath(os.environ.get("CROP_OUTPUT_ROOT", _CROP_OUTPUT_DEFAULT))
CROP_PARTS_FILE = _PROJECT_ROOT / "data" / "crop_parts.json"
SUPPORTED_DETECTORS = {"hands", "pose", "face_mesh"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import mediapipe as mp
    _solutions = getattr(mp, "solutions", None)
    _mp_hands = getattr(_solutions, "hands", None) if _solutions else None
    _mp_pose = getattr(_solutions, "pose", None) if _solutions else None
    _mp_face_mesh = getattr(_solutions, "face_mesh", None) if _solutions else None
    _mp_drawing = getattr(mp.solutions, "drawing_utils", None)
    _mp_drawing_styles = getattr(mp.solutions, "drawing_styles", None)
    _HAS_MEDIAPIPE = _mp_hands is not None and _mp_pose is not None and _mp_face_mesh is not None
except (ImportError, AttributeError):
    _mp_hands = _mp_pose = _mp_face_mesh = _mp_drawing = _mp_drawing_styles = None
    _HAS_MEDIAPIPE = False


def load_crop_parts() -> dict:
    try:
        if CROP_PARTS_FILE.exists():
            data = json.loads(CROP_PARTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "parts" in data and data["parts"]:
                return data
    except Exception:
        pass
    return {"parts": []}


class CropEngine:
    """參數驅動型收割引擎，支援 detector 重用與釋放。"""

    def __init__(self, min_detection_confidence: float = 0.3):
        self._detectors: dict[str, Any] = {}
        self._current_detector_type: str | None = None
        self._min_confidence = max(0.01, min(1.0, min_detection_confidence))

    def _get_detector(self, detector_type: str):
        """取得或建立 detector，切換時釋放舊的以減少 VRAM。"""
        if detector_type not in SUPPORTED_DETECTORS:
            return None
        if self._current_detector_type and self._current_detector_type != detector_type:
            old = self._detectors.pop(self._current_detector_type, None)
            if old is not None:
                try:
                    del old
                except Exception:
                    pass
            self._current_detector_type = None
        if detector_type not in self._detectors:
            det = None
            conf = self._min_confidence
            if detector_type == "hands" and _mp_hands:
                det = _mp_hands.Hands(
                    static_image_mode=True, max_num_hands=2, min_detection_confidence=conf
                )
            elif detector_type == "pose" and _mp_pose:
                det = _mp_pose.Pose(
                    static_image_mode=True, model_complexity=1, min_detection_confidence=conf
                )
            elif detector_type == "face_mesh" and _mp_face_mesh:
                face_conf = min(conf, 0.2)
                det = _mp_face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=4,
                    min_detection_confidence=face_conf, refine_landmarks=True
                )
            if det is not None:
                self._detectors[detector_type] = det
                self._current_detector_type = detector_type
        return self._detectors.get(detector_type)

    def release(self):
        """釋放所有 detector，避免 VRAM 洩漏。"""
        for k in list(self._detectors.keys()):
            try:
                del self._detectors[k]
            except Exception:
                pass
        self._detectors.clear()
        self._current_detector_type = None

    def _get_landmark_points(
        self, results, detector_type: str, part_config: dict, w: int, h: int
    ) -> list[tuple[list[tuple[float, float]], int, float | None]]:
        """
        從檢測結果提取點座標，回傳 [(pts_list, idx, ankle_top_y), ...]。
        ankle_top_y 僅在 pose + y_limit_upper=ankle 時有值。
        """
        out: list[tuple[list[tuple[float, float]], int, float | None]] = []
        landmarks_filter = part_config.get("landmarks")
        exclude = set(part_config.get("exclude_points", []))
        pad = float(part_config.get("padding", 0.2))
        y_limit = part_config.get("y_limit_upper")

        def to_pts(lm_list, indices: list[int] | None = None):
            ids = indices if indices is not None else list(range(len(lm_list)))
            pts = []
            for i in ids:
                if i in exclude or i >= len(lm_list):
                    continue
                lm = lm_list[i]
                vis = getattr(lm, "visibility", 1.0)
                if vis < 0.3:
                    continue
                pts.append((lm.x * w, lm.y * h))
            return pts

        def ankle_top(lm_list, ids: list[int]):
            ys = []
            for aid in [27, 28]:
                if aid in ids and aid < len(lm_list):
                    lm = lm_list[aid]
                    if getattr(lm, "visibility", 0) > 0.3:
                        ys.append(lm.y * h)
            return min(ys) if ys else None

        if detector_type == "hands" and hasattr(results, "multi_hand_landmarks"):
            if results.multi_hand_landmarks:
                for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                    ids = landmarks_filter if landmarks_filter else list(range(len(hand_lm.landmark)))
                    pts = to_pts(hand_lm.landmark, ids)
                    if pts:
                        out.append((pts, idx, None))
        elif detector_type == "pose" and hasattr(results, "pose_landmarks"):
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                ids = landmarks_filter if landmarks_filter else list(range(len(lm)))
                pts = to_pts(lm, ids)
                if pts:
                    at = ankle_top(lm, ids) if y_limit == "ankle" else None
                    out.append((pts, 0, at))
        elif detector_type == "face_mesh" and hasattr(results, "multi_face_landmarks"):
            if results.multi_face_landmarks:
                for idx, face_lm in enumerate(results.multi_face_landmarks):
                    ids = landmarks_filter if landmarks_filter else list(range(len(face_lm.landmark)))
                    pts = to_pts(face_lm.landmark, ids)
                    if pts:
                        out.append((pts, idx, None))
        return out

    def _compute_bbox(
        self, pts: list[tuple[float, float]], part_config: dict, w: int, h: int,
        ankle_top_y: float | None = None,
    ) -> tuple[float, float, float, float] | None:
        """
        把點變成框的核心邏輯：取得極值 → 加入 Padding → 回傳 (x_min, y_min, x_max, y_max)。
        """
        if not pts:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        padding = float(part_config.get("padding", 0.2))
        pad_w = width * padding
        pad_h = height * padding
        x_min = min_x - pad_w
        x_max = max_x + pad_w
        y_min = min_y - pad_h
        y_max = max_y + pad_h
        if part_config.get("y_limit_upper") == "ankle" and ankle_top_y is not None:
            y_min = min(y_min, ankle_top_y)
        y_off = float(part_config.get("y_offset", 0))
        if y_off != 0:
            shift = height * y_off
            y_min += shift
            y_max += shift
        return x_min, y_min, x_max, y_max

    def _expand_to_square(
        self, x_min: float, y_min: float, x_max: float, y_max: float,
        part_config: dict, w: int, h: int
    ) -> tuple[int, int, int, int]:
        """依 aspect_ratio 擴展為正方形並 clamp。"""
        aspect = part_config.get("aspect_ratio", "1:1")
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        rect_w = x_max - x_min
        rect_h = y_max - y_min
        side = max(rect_w, rect_h)
        half = side / 2
        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = int(cx + half)
        y2 = int(cy + half)
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        return x1, y1, x2, y2

    def process_image(
        self,
        part_config: dict,
        output_base: Path,
        crop_size: int,
        image_path: Path | None = None,
        image_array=None,
        stem_override: str | None = None,
        suffix_override: str | None = None,
    ) -> list[tuple[Path, str]]:
        """
        處理單張圖片，回傳 [(out_path, label), ...]。
        可傳 image_path（資料夾）或 (image_array, stem_override, suffix_override)（ZIP 串流）。
        """
        if not _HAS_CV2:
            return []
        if image_array is not None and stem_override is not None:
            img = image_array
            stem = stem_override
            suffix = (suffix_override or ".png").lower()
        elif image_path is not None:
            img = cv2.imread(str(image_path))
            if img is None:
                return []
            stem = image_path.stem
            suffix = (image_path.suffix or ".png").lower()
        else:
            return []
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector_type = part_config.get("detector", "")
        detector = self._get_detector(detector_type)
        if detector is None:
            return []
        if detector_type == "hands":
            results = detector.process(rgb)
        elif detector_type == "pose":
            results = detector.process(rgb)
        elif detector_type == "face_mesh":
            results = detector.process(rgb)
        else:
            return []
        pt_groups = self._get_landmark_points(results, detector_type, part_config, w, h)
        part_id = part_config.get("id", "crop")
        folder = part_config.get("folder", "10_crop")
        tag = part_config.get("tag", f"{part_id}_focus, masterpiece")
        label = (tag.split(",")[0].strip() if isinstance(tag, str) else f"{part_id}_focus")
        output_dir = Path(os.path.abspath(os.path.join(str(output_base), folder)))
        os.makedirs(output_dir, exist_ok=True)
        results_out = []
        for pts, idx, ankle_y in pt_groups:
            bbox = self._compute_bbox(pts, part_config, w, h, ankle_top_y=ankle_y)
            if bbox is None:
                continue
            x1, y1, x2, y2 = self._expand_to_square(*bbox, part_config, w, h)
            try:
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
                out_name = f"{stem}_{part_id}_{idx}{suffix}"
                out_path = output_dir / out_name
                full_path = os.path.abspath(str(out_path))
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                cv2.imwrite(full_path, crop_resized)
                txt_path = os.path.abspath(str(output_dir / f"{stem}_{part_id}_{idx}.txt"))
                Path(txt_path).write_text(tag, encoding="utf-8")
                print(f"Saving [{part_id}] to: {full_path}")
                results_out.append((Path(full_path), label))
            except Exception:
                continue
        return results_out


ZIP_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


CONF_LEVELS = [0.5, 0.4, 0.3, 0.2, 0.15]


def _has_landmarks(results, detector_type: str) -> bool:
    """檢查 MediaPipe 結果是否包含有效 landmark。"""
    if results is None:
        return False
    if detector_type == "hands":
        return bool(results.multi_hand_landmarks)
    if detector_type == "pose":
        return bool(results.pose_landmarks)
    if detector_type == "face_mesh":
        return bool(results.multi_face_landmarks)
    return False


def _dynamic_detect(rgb, detector_type: str) -> tuple[Any | None, float]:
    """
    依序嘗試多個信心門檻，直到找到 landmark 為止。
    回傳 (results, used_confidence)，找不到則 (None, last_conf)。
    """
    for conf in CONF_LEVELS:
        engine = CropEngine(min_detection_confidence=conf)
        try:
            det = engine._get_detector(detector_type)
            if det is None:
                continue
            if detector_type == "hands":
                results = det.process(rgb)
            elif detector_type == "pose":
                results = det.process(rgb)
            elif detector_type == "face_mesh":
                results = det.process(rgb)
            else:
                continue
            if _has_landmarks(results, detector_type):
                print(f"[DynamicDetect] Success! Found landmarks at confidence={conf}")
                return results, conf
        finally:
            engine.release()
    print("[DynamicDetect] No landmarks at any confidence level.")
    return None, CONF_LEVELS[-1]


DEMO_DETECTOR_MAP = {"Face": ("face_mesh", "face"), "Hands": ("hands", "hand"), "Pose": ("pose", "feet")}


def run_detection_demo(image_input, detector_choice: str, confidence: float, resize_1024: bool = True) -> np.ndarray | None:
    """
    偵測驗證 Demo：不存檔、不裁切，只顯示 landmarks 與 bbox。
    使用 mp_drawing.draw_landmarks 畫點位，cv2.rectangle 畫 ROI。
    """
    import time
    if not _HAS_CV2 or not _HAS_MEDIAPIPE:
        return None
    mp_drawing = _mp_drawing
    if mp_drawing is None:
        return None
    mapped = DEMO_DETECTOR_MAP.get(detector_choice, ("face_mesh", "face"))
    detector_type, part_id = mapped
    cfg = load_crop_parts()
    part_by_id = {p["id"]: p for p in cfg.get("parts", []) if p.get("detector") in SUPPORTED_DETECTORS}
    part_config = part_by_id.get(part_id)
    if not part_config:
        part_config = {"padding": 0.2, "y_offset": 0, "landmarks": None}
    if image_input is None:
        return None
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = np.array(image_input)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    conf = max(0.1, min(0.5, float(confidence)))
    if resize_1024 and max(h, w) > 1024:
        scale = 1024 / max(h, w)
        tw, th = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        tw, th = w, h
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb_resized = rgb
    image = rgb.copy()
    engine = CropEngine(min_detection_confidence=conf)
    t0 = time.perf_counter()
    try:
        det = engine._get_detector(detector_type)
        if det is None:
            _draw_fail(image, w, h)
            return image
        if detector_type == "hands":
            results = det.process(rgb_resized)
        elif detector_type == "pose":
            results = det.process(rgb_resized)
        elif detector_type == "face_mesh":
            results = det.process(rgb_resized)
        else:
            return image
        elapsed = time.perf_counter() - t0
        use_w, use_h = tw, th
        if detector_type == "face_mesh" and (not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0):
            _draw_fail(image, w, h)
            return image
        landmark_lists = []
        if detector_type == "hands" and results.multi_hand_landmarks:
            landmark_lists = list(results.multi_hand_landmarks or [])
            for lm_list in results.multi_hand_landmarks or []:
                mp_drawing.draw_landmarks(image, lm_list, _mp_hands.HAND_CONNECTIONS)
        elif detector_type == "pose" and results.pose_landmarks:
            landmark_lists = [results.pose_landmarks]
            mp_drawing.draw_landmarks(image, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS)
        elif detector_type == "face_mesh" and results.multi_face_landmarks:
            landmark_lists = list(results.multi_face_landmarks or [])
            for lm_list in results.multi_face_landmarks or []:
                mp_drawing.draw_landmarks(image, lm_list, _mp_face_mesh.FACEMESH_TESSELATION)
        pt_groups = engine._get_landmark_points(results, detector_type, part_config, use_w, use_h)
        for pts, _, ankle_y in pt_groups:
            bbox = engine._compute_bbox(pts, part_config, use_w, use_h, ankle_top_y=ankle_y)
            if bbox is None:
                continue
            x1, y1, x2, y2 = engine._expand_to_square(*bbox, part_config, use_w, use_h)
            sx = w / use_w if use_w else 1
            sy = h / use_h if use_h else 1
            x1, y1 = int(x1 * sx), int(y1 * sy)
            x2, y2 = int(x2 * sx), int(y2 * sy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        success = bool(landmark_lists)
        if success:
            cv2.putText(image, f"偵測成功！信心值：{conf:.2f}，偵測耗時：{elapsed:.2f}秒", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            _draw_fail(image, w, h)
        return image
    finally:
        engine.release()


def run_precise_local_detect(
    image_input,
    detector_choice: str,
    confidence: float,
    crop_box: tuple[int, int, int, int] | None,
) -> np.ndarray | None:
    """
    針對選取區域進行精準偵測：
    1. 局部裁切 + 對比增強
    2. 二次偵測 (Face/Hands)
    3. 座標映射回原圖
    4. 畫藍框 (手動) + 紅框 (AI 修正)
    """
    import time
    if not _HAS_CV2 or not _HAS_MEDIAPIPE:
        return None
    mp_drawing = _mp_drawing
    if mp_drawing is None:
        return None
    mapped = DEMO_DETECTOR_MAP.get(detector_choice, ("face_mesh", "face"))
    detector_type, part_id = mapped
    if detector_type == "pose":
        return run_detection_demo(image_input, detector_choice, confidence, resize_1024=True)
    cfg = load_crop_parts()
    part_by_id = {p["id"]: p for p in cfg.get("parts", []) if p.get("detector") in SUPPORTED_DETECTORS}
    part_config = part_by_id.get(part_id, {"padding": 0.2, "y_offset": 0, "landmarks": None})
    if image_input is None:
        return None
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = np.array(image_input)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    x1, y1, x2, y2 = 0, 0, w, h
    if crop_box and len(crop_box) >= 4:
        x1 = max(0, int(crop_box[0]))
        y1 = max(0, int(crop_box[1]))
        x2 = min(w, int(crop_box[2]))
        y2 = min(h, int(crop_box[3]))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h
    crop_img = img[y1:y2, x1:x2]
    if crop_img.size == 0:
        return None
    crop_h, crop_w = crop_img.shape[:2]
    lab = cv2.cvtColor(crop_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    rgb_crop = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    scale = 1024 / max(crop_w, crop_h) if max(crop_w, crop_h) > 1024 else 1.0
    tw = int(crop_w * scale)
    th = int(crop_h * scale)
    rgb_resized = cv2.resize(rgb_crop, (tw, th), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else rgb_crop
    overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    conf = max(0.1, min(0.5, float(confidence)))
    engine = CropEngine(min_detection_confidence=min(conf, 0.2))
    t0 = time.perf_counter()
    try:
        det = engine._get_detector(detector_type)
        if det is None:
            return overlay
        if detector_type == "hands":
            results = det.process(rgb_resized)
        elif detector_type == "face_mesh":
            results = det.process(rgb_resized)
        else:
            return overlay
        use_w, use_h = tw, th
        if not _has_landmarks(results, detector_type):
            return overlay
        if detector_type == "hands" and results.multi_hand_landmarks:
            for lm_list in results.multi_hand_landmarks or []:
                for lm in lm_list.landmark:
                    cx = int(lm.x * crop_w) + x1
                    cy = int(lm.y * crop_h) + y1
                    cv2.circle(overlay, (cx, cy), 3, (0, 255, 0), -1)
                if mp_drawing and _mp_hands:
                    for lm_list in results.multi_hand_landmarks or []:
                        overlay_crop = overlay[y1:y2, x1:x2].copy()
                        mp_drawing.draw_landmarks(overlay_crop, lm_list, _mp_hands.HAND_CONNECTIONS)
                        overlay[y1:y2, x1:x2] = overlay_crop
        elif detector_type == "face_mesh" and results.multi_face_landmarks:
            for lm_list in results.multi_face_landmarks or []:
                for lm in lm_list.landmark:
                    cx = int(lm.x * crop_w) + x1
                    cy = int(lm.y * crop_h) + y1
                    cv2.circle(overlay, (cx, cy), 2, (0, 255, 0), -1)
                if mp_drawing and _mp_face_mesh:
                    for lm_list in results.multi_face_landmarks or []:
                        overlay_crop = overlay[y1:y2, x1:x2].copy()
                        mp_drawing.draw_landmarks(overlay_crop, lm_list, _mp_face_mesh.FACEMESH_TESSELATION)
                        overlay[y1:y2, x1:x2] = overlay_crop
        pt_groups = engine._get_landmark_points(results, detector_type, part_config, use_w, use_h)
        for pts, _, ankle_y in pt_groups:
            bbox = engine._compute_bbox(pts, part_config, use_w, use_h, ankle_top_y=ankle_y)
            if bbox is None:
                continue
            bx1, by1, bx2, by2 = engine._expand_to_square(*bbox, part_config, use_w, use_h)
            sx = crop_w / use_w if use_w else 1
            sy = crop_h / use_h if use_h else 1
            rx1 = int(bx1 * sx) + x1
            ry1 = int(by1 * sy) + y1
            rx2 = int(bx2 * sx) + x1
            ry2 = int(by2 * sy) + y1
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 0, 0), 3)
        elapsed = time.perf_counter() - t0
        cv2.putText(overlay, f"偵測成功！信心值：{conf:.2f}，耗時：{elapsed:.2f}秒", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return overlay
    finally:
        engine.release()


LIB_FOLDERS = {
    "hand": ("10_hand_crop", "hand"),
    "feet": ("10_feet_crop", "feet"),
    "face": ("10_face_crop", "face"),
}

OUT_ROOT = os.path.abspath(str(_PROJECT_ROOT / "OUT"))


def _to_square_1_1(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """
    將長方形框以中心往外擴張為 1:1 正方形，碰到底邊則停止。
    """
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    box_w = x2 - x1
    box_h = y2 - y1
    side = max(box_w, box_h)
    half = side / 2
    nx1 = int(cx - half)
    ny1 = int(cy - half)
    nx2 = int(cx + half)
    ny2 = int(cy + half)
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return x1, y1, x2, y2
    return nx1, ny1, nx2, ny2


def process_roi(
    image_input,
    crop_box: tuple[int, int, int, int],
    detector_choice: str,
    confidence: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """
    對選取區域 (藍色手動區域) 執行 AI 偵測，若偵測到 MediaPipe 點位則畫紅色 AI 修正框。
    回傳 (annotated_full_image, cropped_preview_thumbnail, raw_crop_for_save, status_message)。
    """
    if image_input is None or not crop_box or len(crop_box) < 4:
        return None, None, None, "請上傳圖片並框選區域"
    img = None
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = np.array(image_input)
        if img is not None and img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img is not None and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0:
        return None, None, None, "無效影像"
    h, w = img.shape[:2]
    x1 = max(0, int(crop_box[0]))
    y1 = max(0, int(crop_box[1]))
    x2 = min(w, int(crop_box[2]))
    y2 = min(h, int(crop_box[3]))
    if x2 <= x1 or y2 <= y1:
        return None, None, None, "無效座標"
    x1, y1, x2, y2 = _to_square_1_1(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return None, None, None, "無效座標"
    crop_box = (x1, y1, x2, y2)
    raw_crop = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    annotated = run_precise_local_detect(image_input, detector_choice, confidence, crop_box)
    if annotated is None:
        overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        annotated = overlay
    max_side = 256
    crop_h, crop_w = raw_crop.shape[:2]
    scale = max_side / max(crop_w, crop_h) if max(crop_w, crop_h) > max_side else 1.0
    tw, th = int(crop_w * scale), int(crop_h * scale)
    preview = cv2.resize(raw_crop, (tw, th), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else raw_crop
    return annotated, preview, raw_crop, "偵測完成，可確認預覽後存檔"


def _extract_image_from_editor_value(ev) -> np.ndarray | None:
    """從 ImageEditor 的 EditorValue 提取影像 (numpy)。"""
    if ev is None:
        return None
    if isinstance(ev, np.ndarray):
        return ev
    if isinstance(ev, dict):
        img = ev.get("composite")
        if img is None:
            img = ev.get("background")
        if img is not None:
            if isinstance(img, np.ndarray):
                return img
            try:
                from PIL import Image
                if isinstance(img, Image.Image):
                    return np.array(img)
                if isinstance(img, str) and os.path.isfile(img):
                    arr = cv2.imread(img)
                    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr is not None else None
            except Exception:
                pass
    return None


def run_crop_zoom_detect(
    editor_value,
    detector_choice: str,
    confidence: float,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """
    針對 ImageEditor 裁切後的影像進行局部偵測。
    回傳 (annotated_image, raw_crop_for_save, status_message)。
    - 藍框：手動框選邊界（整張圖邊框）
    - 紅框：AI 修正框（若偵測成功）
    """
    import time
    _empty = "請上傳並框選區域後執行偵測。"
    if not _HAS_CV2 or not _HAS_MEDIAPIPE:
        return None, None, "MediaPipe/CV2 不可用"
    mp_drawing = _mp_drawing
    if mp_drawing is None:
        return None, None, "繪圖模組不可用"
    img = _extract_image_from_editor_value(editor_value)
    if img is None or img.size == 0:
        return None, None, _empty
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None:
        return None, None, _empty
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = 1024 / max(w, h) if max(w, h) > 1024 else 1.0
    tw, th = int(w * scale), int(h * scale)
    rgb_resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR) if scale != 1.0 else rgb
    overlay = rgb.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)
    conf = max(0.1, min(0.5, float(confidence)))
    mapped = DEMO_DETECTOR_MAP.get(detector_choice, ("face_mesh", "face"))
    detector_type, part_id = mapped
    cfg = load_crop_parts()
    part_by_id = {p["id"]: p for p in cfg.get("parts", []) if p.get("detector") in SUPPORTED_DETECTORS}
    part_config = part_by_id.get(part_id, {"padding": 0.2, "y_offset": 0, "landmarks": None})
    engine = CropEngine(min_detection_confidence=min(conf, 0.2))
    t0 = time.perf_counter()
    try:
        det = engine._get_detector(detector_type)
        if det is None:
            return overlay, rgb, "偵測器不可用"
        if detector_type == "hands":
            results = det.process(rgb_resized)
        elif detector_type == "face_mesh":
            results = det.process(rgb_resized)
        elif detector_type == "pose":
            results = det.process(rgb_resized)
        else:
            return overlay, rgb, "不支援的偵測器"
        if not _has_landmarks(results, detector_type):
            return overlay, rgb, "偵測失敗，保留手動框（藍色）"
        use_w, use_h = tw, th
        if detector_type == "hands" and results.multi_hand_landmarks:
            for lm_list in results.multi_hand_landmarks or []:
                mp_drawing.draw_landmarks(overlay, lm_list, _mp_hands.HAND_CONNECTIONS)
        elif detector_type == "face_mesh" and results.multi_face_landmarks:
            for lm_list in results.multi_face_landmarks or []:
                mp_drawing.draw_landmarks(overlay, lm_list, _mp_face_mesh.FACEMESH_TESSELATION)
        elif detector_type == "pose" and results.pose_landmarks:
            mp_drawing.draw_landmarks(overlay, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS)
        pt_groups = engine._get_landmark_points(results, detector_type, part_config, use_w, use_h)
        for pts, _, ankle_y in pt_groups:
            bbox = engine._compute_bbox(pts, part_config, use_w, use_h, ankle_top_y=ankle_y)
            if bbox is None:
                continue
            x1, y1, x2, y2 = engine._expand_to_square(*bbox, part_config, use_w, use_h)
            sx = w / use_w if use_w else 1
            sy = h / use_h if use_h else 1
            x1, y1 = int(x1 * sx), int(y1 * sy)
            x2, y2 = int(x2 * sx), int(y2 * sy)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 3)
        elapsed = time.perf_counter() - t0
        return overlay, rgb, f"偵測成功！信心值：{conf:.2f}，耗時：{elapsed:.2f}秒（藍框=手動、紅框=AI 修正）"
    finally:
        engine.release()


def save_to_library(
    crop_image: np.ndarray | None,
    lib_type: str,
    base_filename: str | None = None,
) -> str:
    """
    將裁切影像存入資源庫。lib_type: hand | feet | face
    回傳狀態訊息。
    """
    if crop_image is None or crop_image.size == 0:
        return "[錯誤] 尚無可存檔的裁切影像，請先執行局部偵測。"
    if lib_type not in LIB_FOLDERS:
        return f"[錯誤] 不支援的資源庫類型: {lib_type}"
    folder_name, part_id = LIB_FOLDERS[lib_type]
    lib_root = os.path.abspath(str(Path(OUT_ROOT) / folder_name))
    os.makedirs(lib_root, exist_ok=True)
    cfg = load_crop_parts()
    part_by_id = {p["id"]: p for p in cfg.get("parts", [])}
    part_config = part_by_id.get(part_id, {})
    tag_str = str(part_config.get("tag", "masterpiece")).strip()
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    stem = base_filename or "crop"
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:32]
    if stem and stem != "crop":
        fname = f"{stem}_{ts}.jpg"
    else:
        fname = f"crop_{ts}.jpg"
    out_path = os.path.join(lib_root, fname)
    txt_path = os.path.splitext(out_path)[0] + ".txt"
    if crop_image.ndim == 2:
        to_save = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2BGR)
    elif crop_image.ndim == 3 and crop_image.shape[2] == 3:
        to_save = cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR)
    else:
        to_save = crop_image
    try:
        cv2.imwrite(out_path, to_save)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(tag_str)
        return f"[成功] 已存至 {folder_name}/: {fname}"
    except Exception as e:
        return f"[錯誤] 存檔失敗: {e}"


def _draw_fail(overlay, w: int, h: int):
    """在圖片中心畫紅叉並寫 DETECTION FAILED。"""
    cx, cy = w // 2, h // 2
    sz = min(w, h) // 4
    cv2.line(overlay, (cx - sz, cy - sz), (cx + sz, cy + sz), (255, 0, 0), 4)
    cv2.line(overlay, (cx - sz, cy + sz), (cx + sz, cy - sz), (255, 0, 0), 4)
    cv2.putText(overlay, "DETECTION FAILED", (cx - 100, cy + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)


def run_multi_scale_test(image_input, part_id: str) -> tuple[np.ndarray | None, str]:
    """
    在多种縮放尺寸下測試偵測，回傳 (preview_image, report_text)。
    用於找出最佳辨識尺寸。
    """
    if not _HAS_CV2 or not _HAS_MEDIAPIPE:
        return None, "MediaPipe/CV2 not available"
    cfg = load_crop_parts()
    part_by_id = {p["id"]: p for p in cfg.get("parts", []) if p.get("detector") in SUPPORTED_DETECTORS}
    part_config = part_by_id.get(part_id)
    if not part_config or image_input is None:
        return None, "No image or invalid part"
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = np.array(image_input)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0:
        return None, "Invalid image"
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector_type = part_config.get("detector", "")
    engine = CropEngine(min_detection_confidence=0.2)
    report_lines = [f"Multi-Scale Test | part={part_id} | detector={detector_type}", f"Original: {w}x{h}"]
    best_overlay = None
    best_scale_label = ""
    def _scaled_dims(max_side: int):
        s = max_side / max(w, h) if max(w, h) > max_side else 1.0
        return max(32, int(w * s)), max(32, int(h * s))
    scales = [
        ("Original", w, h),
        ("1024", *_scaled_dims(1024)),
        ("768", *_scaled_dims(768)),
        ("512", *_scaled_dims(512)),
    ]
    try:
        det = engine._get_detector(detector_type)
        if det is None:
            return rgb, "Detector not available"
        for label, tw, th in scales:
            if tw < 32 or th < 32:
                report_lines.append(f"  {label}: skip (too small)")
                continue
            if label == "Original":
                rgb_test = rgb
            else:
                rgb_test = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
            if detector_type == "hands":
                res = det.process(rgb_test)
                has = bool(res.multi_hand_landmarks)
            elif detector_type == "pose":
                res = det.process(rgb_test)
                has = bool(res.pose_landmarks)
            elif detector_type == "face_mesh":
                res = det.process(rgb_test)
                has = bool(res.multi_face_landmarks)
            else:
                has = False
            status = "OK" if has else "FAIL"
            report_lines.append(f"  {label} ({tw}x{th}): {status}")
            if has and best_overlay is None:
                overlay = rgb.copy()
                if detector_type == "hands" and res.multi_hand_landmarks and _mp_drawing and _mp_hands:
                    for hand_lm in res.multi_hand_landmarks or []:
                        _mp_drawing.draw_landmarks(overlay, hand_lm, _mp_hands.HAND_CONNECTIONS)
                elif detector_type == "pose" and res.pose_landmarks and _mp_drawing and _mp_pose:
                    _mp_drawing.draw_landmarks(overlay, res.pose_landmarks, _mp_pose.POSE_CONNECTIONS)
                elif detector_type == "face_mesh" and res.multi_face_landmarks and _mp_drawing and _mp_face_mesh:
                    for face_lm in res.multi_face_landmarks or []:
                        _mp_drawing.draw_landmarks(overlay, face_lm, _mp_face_mesh.FACEMESH_TESSELATION)
                best_overlay = overlay
                best_scale_label = label
        report = "\n".join(report_lines)
        if best_overlay is None:
            return rgb, report
        return best_overlay, report
    finally:
        engine.release()


def _log_landmark_coords(detector_type: str, landmark_lists: list, w: int, h: int):
    """輸出每個點的座標到 Terminal。座標若全為 0.0 或 None 表示模型未正常啟動。"""
    print(f"[Calibration] detector={detector_type} w={w} h={h} num_groups={len(landmark_lists)}")
    for idx, lm_obj in enumerate(landmark_lists):
        if lm_obj is None:
            print(f"  [{idx}] landmark_list=None")
            continue
        pts = getattr(lm_obj, "landmark", lm_obj)
        if not hasattr(pts, "__iter__"):
            print(f"  [{idx}] (no iterable landmarks)")
            continue
        pts_list = list(pts)
        max_show = 50 if detector_type == "face_mesh" else 999
        for i, lm in enumerate(pts_list):
            if i >= max_show:
                print(f"  [{idx}] ... ({len(pts_list)} total, showing first {max_show})")
                break
            x = getattr(lm, "x", None)
            y = getattr(lm, "y", None)
            z = getattr(lm, "z", None)
            vis = getattr(lm, "visibility", None)
            coord_str = f"x={x}, y={y}, z={z}" + (f", vis={vis}" if vis is not None else "")
            print(f"  [{idx}] #{i}: {coord_str}")


def run_calibration(
    image_input,
    part_id: str,
    manual_padding: float,
    manual_y_offset: float,
    manual_confidence: float,
) -> tuple[np.ndarray | None, str]:
    """
    校正預覽：用 mp_drawing.draw_landmarks 繪製所有偵測點，再畫紅色裁切框。
    回傳 (annotated_image, status_msg)。偵測失敗時 status_msg 顯示提示。
    """
    FAIL_MSG = "偵測失敗，請調整信心值或更換圖片"
    if not _HAS_CV2 or not _HAS_MEDIAPIPE:
        return None, FAIL_MSG
    cfg = load_crop_parts()
    part_by_id = {p["id"]: p for p in cfg.get("parts", []) if p.get("detector") in SUPPORTED_DETECTORS}
    part_config = part_by_id.get(part_id)
    if not part_config:
        return None, FAIL_MSG
    if image_input is None:
        return None, ""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        img = np.array(image_input)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if img is None or img.size == 0:
        return None, FAIL_MSG
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    overlay = rgb.copy()
    conf = max(0.01, min(1.0, float(manual_confidence)))
    override_cfg = dict(part_config)
    override_cfg["padding"] = max(0, min(2, float(manual_padding)))
    override_cfg["y_offset"] = max(-0.5, min(0.5, float(manual_y_offset)))
    engine = CropEngine(min_detection_confidence=conf)
    try:
        detector_type = part_config.get("detector", "")
        detector = engine._get_detector(detector_type)
        if detector is None:
            cv2.putText(overlay, "Detection Failed: No Landmarks Found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print("[Calibration] Detector not available")
            return overlay, FAIL_MSG
        if detector_type == "hands":
            results = detector.process(rgb)
        elif detector_type == "pose":
            results = detector.process(rgb)
        elif detector_type == "face_mesh":
            results = detector.process(rgb)
        else:
            return overlay, ""

        if not _has_landmarks(results, detector_type):
            if detector_type == "face_mesh":
                print("Warning: Face Mesh found NOTHING.")
                if max(h, w) > 1024:
                    scale = 1024 / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    results = detector.process(rgb_small)
            if not _has_landmarks(results, detector_type):
                results, _ = _dynamic_detect(rgb, detector_type)
            if not _has_landmarks(results, detector_type):
                cv2.putText(overlay, "Detection Failed: No Landmarks Found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                _log_landmark_coords(detector_type, [], w, h)
                return overlay, FAIL_MSG

        landmark_lists_for_log: list = []
        if detector_type == "hands" and hasattr(results, "multi_hand_landmarks"):
            landmark_lists_for_log = list(results.multi_hand_landmarks or [])
            if _mp_drawing and _mp_hands:
                for hand_lm in results.multi_hand_landmarks or []:
                    _mp_drawing.draw_landmarks(overlay, hand_lm, _mp_hands.HAND_CONNECTIONS)
        elif detector_type == "pose" and hasattr(results, "pose_landmarks"):
            if results.pose_landmarks:
                landmark_lists_for_log = [results.pose_landmarks]
                if _mp_drawing and _mp_pose:
                    _mp_drawing.draw_landmarks(
                        overlay, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS
                    )
        elif detector_type == "face_mesh" and hasattr(results, "multi_face_landmarks"):
            landmark_lists_for_log = list(results.multi_face_landmarks or [])
            if _mp_drawing and _mp_face_mesh:
                for face_lm in results.multi_face_landmarks or []:
                    _mp_drawing.draw_landmarks(overlay, face_lm, _mp_face_mesh.FACEMESH_TESSELATION)

        _log_landmark_coords(detector_type, landmark_lists_for_log, w, h)

        pt_groups = engine._get_landmark_points(results, detector_type, override_cfg, w, h)
        has_valid_crop = False
        for pts, idx, ankle_y in pt_groups:
            bbox = engine._compute_bbox(pts, override_cfg, w, h, ankle_top_y=ankle_y)
            if bbox is None:
                continue
            x1, y1, x2, y2 = engine._expand_to_square(*bbox, override_cfg, w, h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
            has_valid_crop = True

        num_groups = len(landmark_lists_for_log)
        status_msg = FAIL_MSG if num_groups == 0 else ""
        if not has_valid_crop and not any(landmark_lists_for_log):
            cv2.putText(overlay, "Detection Failed: No Landmarks Found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            status_msg = FAIL_MSG
        elif not has_valid_crop and landmark_lists_for_log:
            cv2.putText(overlay, "Landmarks found but no valid crop region", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
        return overlay, status_msg
    finally:
        engine.release()


def _iter_source_images(src: Path, recursive: bool) -> list[tuple[Path | None, str, str, bytes | None]]:
    """回傳 [(path_or_none, stem, suffix, bytes_or_none), ...]。含 JPG/PNG 及資料夾內 ZIP。"""
    items: list[tuple[Path | None, str, str, bytes | None]] = []
    if recursive:
        img_files = [f for ext in IMG_EXT for f in src.rglob(f"*{ext}")]
        zip_files = list(src.rglob("*.zip"))
    else:
        img_files = [f for ext in IMG_EXT for f in src.glob(f"*{ext}") if f.is_file()]
        zip_files = [f for f in src.glob("*.zip") if f.is_file()]
    for f in sorted(set(f for f in img_files if f.is_file())):
        items.append((f, f.stem, (f.suffix or ".png").lower(), None))
    for zp in sorted(set(zip_files)):
        if not zp.is_file():
            continue
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                items.extend(_iter_zip_images(zp, zf))
        except (zipfile.BadZipFile, RuntimeError):
            continue
    return items


def _iter_zip_images(zip_path: Path, zip_ref: zipfile.ZipFile) -> list[tuple[Path | None, str, str, bytes | None]]:
    """從 ZIP 取得 (path=None, stem, suffix, bytes), ...。用 os.path.basename 取純檔名避免子目錄。"""
    zip_stem = zip_path.stem
    items: list[tuple[Path | None, str, str, bytes | None]] = []
    for idx, name in enumerate(zip_ref.namelist()):
        basename = os.path.basename(name)
        p = Path(basename)
        if p.suffix.lower() not in ZIP_IMG_EXT:
            continue
        if p.name.startswith(".") or "/__MACOSX" in name.replace("\\", "/"):
            continue
        try:
            data = zip_ref.read(name)
        except (RuntimeError, zipfile.BadZipFile) as e:
            raise RuntimeError(f"ZIP 讀取失敗 [{name}]: {e}") from e
        inner_stem = p.stem
        stem = f"{zip_stem}_{idx:04d}_{inner_stem}"
        suffix = p.suffix.lower()
        items.append((None, stem, suffix, data))
    return items


def run_crop_batch(
    source_path: str,
    dest_folder: str,
    part_config: dict,
    crop_size: int = 512,
    min_resolution: int = 256,
    recursive: bool = True,
    manual_confidence: float | None = None,
) -> Generator[tuple[str, list[tuple[str, str]], int, int], None, None]:
    """
    批次收割，支援資料夾或 .zip。yield (log_line, previews, current, total)。
    """
    if not _HAS_CV2:
        yield "❌ 需要 opencv-python", [], 0, 0
        return
    if not _HAS_MEDIAPIPE:
        yield "❌ 需要 mediapipe", [], 0, 0
        return
    src = Path(source_path.strip())
    is_zip = src.suffix.lower() == ".zip"
    dest_resolved = os.path.abspath((dest_folder or "").strip() or CROP_OUTPUT_ROOT)
    dst = Path(CROP_OUTPUT_ROOT) if is_zip else Path(dest_resolved)
    if not src.exists():
        yield f"❌ 找不到來源: {src}", [], 0, 0
        return
    items: list[tuple[Path | None, str, str, bytes | None]] = []
    zip_ref = None

    if is_zip:
        print(f"[ZIP] 輸出路徑根目錄: {os.path.abspath(CROP_OUTPUT_ROOT)}")
        try:
            zip_ref = zipfile.ZipFile(src, "r")
            items = _iter_zip_images(src, zip_ref)
        except zipfile.BadZipFile as e:
            yield f"❌ ZIP 格式損壞或無法辨識: {e}", [], 0, 0
            return
        except RuntimeError as e:
            err = str(e)
            if "password" in err.lower() or "encrypted" in err.lower():
                yield "❌ ZIP 有密碼保護，目前不支援加密壓縮檔", [], 0, 0
            else:
                yield f"❌ ZIP 讀取錯誤: {e}", [], 0, 0
            return
    else:
        if not src.is_dir():
            yield f"❌ 來源必須是資料夾或 .zip 檔案: {src}", [], 0, 0
            return
        items = _iter_source_images(src, recursive)

    label_name = part_config.get("label", part_config.get("id", "部位"))
    min_conf = manual_confidence if manual_confidence is not None else 0.3
    engine = CropEngine(min_detection_confidence=min_conf)
    total = len(items)
    previews: list[tuple[str, str]] = []
    try:
        for i, (path_or_none, stem, suffix, data) in enumerate(items):
            try:
                if data is not None:
                    arr = np.frombuffer(data, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                else:
                    if path_or_none is None:
                        continue
                    if path_or_none.stat().st_size == 0:
                        continue
                    img = cv2.imread(str(path_or_none))
                if img is None:
                    continue
                ih, iw = img.shape[:2]
                if iw < min_resolution or ih < min_resolution:
                    continue
                if data is not None:
                    outs = engine.process_image(
                        part_config, dst, crop_size,
                        image_array=img, stem_override=stem, suffix_override=suffix,
                    )
                else:
                    outs = engine.process_image(part_config, dst, crop_size, image_path=path_or_none)
                if outs:
                    for out_path, lab in outs:
                        previews.append((str(out_path), lab))
                    yield f"正在收割 [{label_name}]... 已完成 {i + 1}/{total}", previews[-20:], i + 1, total
            except Exception as e:
                fn = stem or (path_or_none.name if path_or_none else "?")
                yield f"⚠️ [{fn}] {e}", previews[-20:], i + 1, total
    finally:
        if zip_ref is not None:
            try:
                zip_ref.close()
            except Exception:
                pass
        engine.release()
        if is_zip:
            gc.collect()
    yield f"✅ [{label_name}] 完成，共 {total} 張", previews, total, total
