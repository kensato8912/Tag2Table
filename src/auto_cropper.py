"""
Auto Cropper：多部位視覺主動偵測
支援手部 (Hands)、腳部 (Pose)、臉部 (Face Detection)。
部位定義存於 data/crop_parts.json，可自訂 folder、tag。
"""
import json
from pathlib import Path
from typing import Callable, Generator

_PROJECT_ROOT = Path(__file__).parent.parent
CROP_PARTS_FILE = _PROJECT_ROOT / "data" / "crop_parts.json"

# 預設部位（JSON 不存在時使用）
_DEFAULT_PARTS = {
    "parts": [
        {"id": "hand", "folder": "10_hand_crop", "tag": "hand_focus, beautiful_detailed_hand, masterpiece", "detector": "hands", "label": "手部 (Hands)"},
        {"id": "feet", "folder": "10_feet_crop", "tag": "feet_focus, beautiful_detailed_feet, masterpiece", "detector": "pose", "label": "腳部 (Feet)"},
        {"id": "face", "folder": "10_face_crop", "tag": "face_focus, beautiful_detailed_face, masterpiece", "detector": "face", "label": "臉部 (Face)"},
    ]
}


def load_crop_parts() -> dict:
    """載入 data/crop_parts.json，不存在或失敗則回傳預設"""
    try:
        # 若實檔不存在，從範例複製建立
        example_path = _PROJECT_ROOT / "data" / "crop_parts.example.json"
        if not CROP_PARTS_FILE.exists() and example_path.exists():
            import shutil
            CROP_PARTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example_path, CROP_PARTS_FILE)
        if CROP_PARTS_FILE.exists():
            data = json.loads(CROP_PARTS_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "parts" in data and data["parts"]:
                return data
    except Exception:
        pass
    return _DEFAULT_PARTS.copy()


def save_crop_parts(data: dict) -> bool:
    """儲存至 data/crop_parts.json"""
    try:
        CROP_PARTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        CROP_PARTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def get_part_config(part_id: str) -> dict | None:
    """依 id 取得部位設定"""
    cfg = load_crop_parts()
    for p in cfg.get("parts", []):
        if p.get("id") == part_id:
            return p
    return None


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
    _mp_face = getattr(_solutions, "face_detection", None) if _solutions else None
    _mp_face_mesh = getattr(_solutions, "face_mesh", None) if _solutions else None
    _HAS_MEDIAPIPE = _mp_hands is not None
    _HAS_POSE = _mp_pose is not None
    _HAS_FACE = _mp_face is not None
    _HAS_FACE_MESH = _mp_face_mesh is not None
except (ImportError, AttributeError):
    _mp_hands = _mp_pose = _mp_face = _mp_face_mesh = None
    _HAS_MEDIAPIPE = _HAS_POSE = _HAS_FACE = _HAS_FACE_MESH = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# 僅支援以下 MediaPipe 偵測器，JSON 中其他 detector 會自動過濾
SUPPORTED_DETECTORS = {"hands", "pose", "face", "face_mesh"}
# MediaPipe Pose 腳部關鍵點索引 (BlazePose 33 點)
_FOOT_LANDMARK_IDS = [27, 28, 29, 30, 31, 32]
# 腳趾區域：foot_index (31, 32) 為腳尖（BlazePose: 27,28=踝 29,30=跟 31,32=趾）
_TOE_LANDMARK_IDS = [31, 32]
# 腳踝至腳尖：僅用 27-32，忽略膝蓋以上
_ANKLE_TOE_IDS = [27, 28, 29, 30, 31, 32]
_ANKLE_IDS = [27, 28]
_CALF_HALF_MAX = 0.5  # 向上延伸不得超過小腿的一半
_ANKLE_TOE_MIN_DIST = 30  # 踝趾距離過近時視為俯視，加大 padding
_CLOSEUP_ASPECT_THRESH = 1.5  # 長寬比 > 此值視為裁切較近，加 close-up
# MediaPipe Hands 手指關鍵點（排除 0 手腕），1-20 為五指區域
_FINGER_LANDMARK_IDS = list(range(1, 21))
# 長腿模式：BlazePose 23,24=髖 25,26=膝 31,32=腳尖
_FULL_LEG_TOP_IDS = [23, 24]
_FULL_LEG_TOP_FALLBACK_IDS = [25, 26]
_FULL_LEG_BOTTOM_IDS = [31, 32]
_FULL_LEG_OUTPUT_SIZE = (512, 768)  # SD 支援的長方形
_HIP_CONFIDENCE_THRESHOLD = 0.5
# Face Mesh 精確臉部：10=額頂 152=下巴 234,454=左右邊 1=鼻尖 13,14=嘴開合
_FACE_MESH_TOP = 10
_FACE_MESH_BOTTOM = 152
_FACE_MESH_LEFT = 234
_FACE_MESH_RIGHT = 454
_FACE_MESH_NOSE = 1
_FACE_MESH_MOUTH_UPPER = 13
_FACE_MESH_MOUTH_LOWER = 14
_FACE_MESH_TOP_EXTEND = 0.15
_FACE_MESH_BOTTOM_EXTEND = 0.05
_OPEN_MOUTH_DIST_RATIO = 0.025


def _get_hands_detector():
    if not _HAS_MEDIAPIPE:
        return None
    return _mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    )


def _get_pose_detector():
    if not _HAS_POSE:
        return None
    return _mp_pose.Pose(
        static_image_mode=True, model_complexity=1, min_detection_confidence=0.5
    )


def _get_face_detector():
    if not _HAS_FACE:
        return None
    return _mp_face.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )


def _get_face_mesh_detector():
    if not _HAS_FACE_MESH:
        return None
    return _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=4,
        min_detection_confidence=0.2, refine_landmarks=True
    )


def _crop_and_save(
    img, x1: int, y1: int, x2: int, y2: int,
    output_dir: Path, stem: str, suffix: str, part_id: str, idx: int,
    crop_size: int, tag: str | None = None, extra_tags: str | None = None,
) -> Path | None:
    """根據 bbox 裁切、resize、存檔並寫入對應標籤。tag 來自 JSON，extra_tags 可追加（如 close-up）。"""
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    base_tag = tag or f"{part_id}_focus, masterpiece"
    final_tag = f"{base_tag}, {extra_tags}" if extra_tags else base_tag
    try:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_resized = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{stem}_{part_id}_{idx}{suffix}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), crop_resized)
        (output_dir / f"{stem}_{part_id}_{idx}.txt").write_text(final_tag, encoding="utf-8")
        return out_path
    except Exception:
        return None


def _crop_and_save_rect(
    img, x1: int, y1: int, x2: int, y2: int,
    output_dir: Path, stem: str, suffix: str, part_id: str, idx: int,
    out_w: int, out_h: int, tag: str | None = None,
) -> Path | None:
    """裁切並縮放到指定矩形尺寸 (如 512×768)，用於長腿模式"""
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    tag = tag or f"{part_id}_focus, masterpiece"
    try:
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        crop_resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"{stem}_{part_id}_{idx}{suffix}"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), crop_resized)
        (output_dir / f"{stem}_{part_id}_{idx}.txt").write_text(tag, encoding="utf-8")
        return out_path
    except Exception:
        return None


def _build_parts_map() -> dict:
    """從 JSON 建立 {part_id: {folder, tag, detector, label}} 映射，僅包含支援的 detector"""
    cfg = load_crop_parts()
    return {p["id"]: p for p in cfg.get("parts", []) if p.get("id") and p.get("detector") in SUPPORTED_DETECTORS}


def _detect_and_crop_hands(
    img, rgb, hands_detector, stem: str, suffix: str,
    output_dir: Path, crop_size: int, padding: float, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """偵測手部並裁切，回傳 [(path, label), ...]"""
    if not hands_detector:
        return []
    results = hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        return []
    h, w = img.shape[:2]
    tag = part_cfg.get("tag", "hand_focus, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "hand_focus")
    out_list = []
    for idx, hand_lm in enumerate(results.multi_hand_landmarks):
        xs = [lm.x * w for lm in hand_lm.landmark]
        ys = [lm.y * h for lm in hand_lm.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        side = max(x_max - x_min, y_max - y_min) * (1 + padding)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        x1, y1 = int(cx - side / 2), int(cy - side / 2)
        x2, y2 = int(x1 + side), int(y1 + side)
        p = _crop_and_save(img, x1, y1, x2, y2, output_dir, stem, suffix, "hand", idx, crop_size, tag)
        if p:
            out_list.append((p, label))
    return out_list


def _detect_and_crop_fingers(
    img, rgb, hands_detector, stem: str, suffix: str,
    output_dir: Path, crop_size: int, padding: float, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """偵測手指區域（排除手腕）並裁切，回傳 [(path, label), ...]"""
    if not hands_detector:
        return []
    results = hands_detector.process(rgb)
    if not results.multi_hand_landmarks:
        return []
    h, w = img.shape[:2]
    tag = part_cfg.get("tag", "fingers_focus, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "fingers_focus")
    out_list = []
    for idx, hand_lm in enumerate(results.multi_hand_landmarks):
        xs = [hand_lm.landmark[i].x * w for i in _FINGER_LANDMARK_IDS if i < len(hand_lm.landmark)]
        ys = [hand_lm.landmark[i].y * h for i in _FINGER_LANDMARK_IDS if i < len(hand_lm.landmark)]
        if not xs or not ys:
            continue
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        side = max(x_max - x_min, y_max - y_min) * (1 + padding)
        side = max(side, crop_size * 0.2)
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        x1, y1 = int(cx - side / 2), int(cy - side / 2)
        x2, y2 = int(x1 + side), int(y1 + side)
        p = _crop_and_save(img, x1, y1, x2, y2, output_dir, stem, suffix, "fingers", idx, crop_size, tag)
        if p:
            out_list.append((p, label))
    return out_list


def _detect_and_crop_feet(
    img, rgb, pose_detector, stem: str, suffix: str,
    output_dir: Path, crop_size: int, padding: float, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """偵測腳部並裁切"""
    if not pose_detector:
        return []
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return []
    h, w = img.shape[:2]
    lm = results.pose_landmarks.landmark
    xs, ys = [], []
    for i in _FOOT_LANDMARK_IDS:
        if i < len(lm) and lm[i].visibility > 0.3:
            xs.append(lm[i].x * w)
            ys.append(lm[i].y * h)
    if not xs or not ys:
        return []
    tag = part_cfg.get("tag", "feet_focus, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "feet_focus")
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    side = max(x_max - x_min, y_max - y_min) * (1 + padding)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x1, y1 = int(cx - side / 2), int(cy - side / 2)
    x2, y2 = int(x1 + side), int(y1 + side)
    p = _crop_and_save(img, x1, y1, x2, y2, output_dir, stem, suffix, "feet", 0, crop_size, tag)
    return [(p, label)] if p else []


def _detect_and_crop_toes(
    img, rgb, pose_detector, stem: str, suffix: str,
    output_dir: Path, crop_size: int, padding: float, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """偵測腳趾（foot_index）並裁切，回傳 [(path, label), ...]"""
    if not pose_detector:
        return []
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return []
    h, w = img.shape[:2]
    lm = results.pose_landmarks.landmark
    xs, ys = [], []
    for i in _TOE_LANDMARK_IDS:
        if i < len(lm) and lm[i].visibility > 0.3:
            xs.append(lm[i].x * w)
            ys.append(lm[i].y * h)
    if not xs or not ys:
        return []
    tag = part_cfg.get("tag", "toes_focus, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "toes_focus")
    out_list = []
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    side = max(x_max - x_min, y_max - y_min) * (1 + padding)
    side = max(side, crop_size * 0.3)  # 避免過小
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    x1, y1 = int(cx - side / 2), int(cy - side / 2)
    x2, y2 = int(x1 + side), int(y1 + side)
    p = _crop_and_save(img, x1, y1, x2, y2, output_dir, stem, suffix, "toes", 0, crop_size, tag)
    if p:
        out_list.append((p, label))
    return out_list


def _detect_and_crop_ankle_toe(
    img, rgb, pose_detector, stem: str, suffix: str,
    output_dir: Path, crop_size: int, padding: float, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """
    腳踝至腳尖：僅用 27-32，以腳心為中心擴為 1:1 正方形。
    優先向兩側延伸，嚴禁向上超過小腿一半。裁切近則加 close-up。俯視時加大 padding。
    """
    if not pose_detector:
        return []
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return []
    h, w = img.shape[:2]
    lm = results.pose_landmarks.landmark
    xs, ys = [], []
    for i in _ANKLE_TOE_IDS:
        if i < len(lm) and getattr(lm[i], "visibility", 0) > 0.3:
            xs.append(lm[i].x * w)
            ys.append(lm[i].y * h)
    if not xs or not ys:
        return []
    ankle_ys = [lm[i].y * h for i in _ANKLE_IDS if i < len(lm) and getattr(lm[i], "visibility", 0) > 0.3]
    ankle_y = min(ankle_ys) if ankle_ys else min(ys)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    rect_w = x_max - x_min
    rect_h = y_max - y_min
    ankle_toe_dist = rect_h
    pad = padding
    if ankle_toe_dist < _ANKLE_TOE_MIN_DIST:
        pad = max(pad, 0.4)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    calf_len = 0
    for ki, ai in zip([25, 26], [27, 28]):
        if ki < len(lm) and ai < len(lm):
            ky, ay = lm[ki].y * h, lm[ai].y * h
            if getattr(lm[ki], "visibility", 0) > 0.3 and getattr(lm[ai], "visibility", 0) > 0.3:
                calf_len = max(calf_len, abs(ay - ky))
    max_upward = calf_len * _CALF_HALF_MAX if calf_len > 0 else rect_h
    side = max(rect_w, rect_h) * (1 + pad)
    side = max(side, crop_size * 0.25)
    y1 = cy - side / 2
    y2 = cy + side / 2
    y1_cap = ankle_y - max_upward
    if y1 < y1_cap:
        y1 = y1_cap
        y2 = y1 + side
        cy = (y1 + y2) / 2
    x1 = int(cx - side / 2)
    y1 = int(y1)
    x2 = int(cx + side / 2)
    y2 = int(y2)
    tag = part_cfg.get("tag", "ankle_toe_focus, feet_focus, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "ankle_toe_focus")
    extra = None
    if rect_h > 0 and rect_w > 0 and (rect_h / rect_w > _CLOSEUP_ASPECT_THRESH or rect_w / rect_h > _CLOSEUP_ASPECT_THRESH):
        extra = "close-up"
    p = _crop_and_save(img, x1, y1, x2, y2, output_dir, stem, suffix, "ankle_toe", 0, crop_size, tag, extra_tags=extra)
    return [(p, label)] if p else []


def _detect_and_crop_full_leg(
    img, rgb, pose_detector, stem: str, suffix: str,
    output_dir: Path, padding: float, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """
    長腿模式：髖(23,24)→腳尖(31,32)，1:2 裁切框，輸出 512×768。
    臀部信心值低時退縮至膝(25,26)作為頂部。
    """
    if not pose_detector:
        return []
    results = pose_detector.process(rgb)
    if not results.pose_landmarks:
        return []
    h, w = img.shape[:2]
    lm = results.pose_landmarks.landmark
    out_w, out_h = _FULL_LEG_OUTPUT_SIZE

    def _get_pts(ids: list[int]):
        xs, ys, vis = [], [], []
        for i in ids:
            if i < len(lm):
                v = getattr(lm[i], "visibility", 1.0)
                xs.append(lm[i].x * w)
                ys.append(lm[i].y * h)
                vis.append(v)
        return xs, ys, vis

    top_xs, top_ys, top_vis = _get_pts(_FULL_LEG_TOP_IDS)
    bot_xs, bot_ys, _ = _get_pts(_FULL_LEG_BOTTOM_IDS)
    all_leg_ids = _FULL_LEG_TOP_IDS + [25, 26] + list(range(27, 33))
    all_xs, all_ys = [], []
    for i in all_leg_ids:
        if i < len(lm) and getattr(lm[i], "visibility", 0) > 0.3:
            all_xs.append(lm[i].x * w)
            all_ys.append(lm[i].y * h)

    if not bot_xs or not bot_ys:
        return []
    y_max = max(bot_ys)

    if top_xs and top_ys and top_vis and min(top_vis) >= _HIP_CONFIDENCE_THRESHOLD:
        y_min = min(top_ys)
    else:
        fallback_xs, fallback_ys, _ = _get_pts(_FULL_LEG_TOP_FALLBACK_IDS)
        if fallback_xs and fallback_ys:
            y_min = min(fallback_ys)
        else:
            return []

    if not all_xs or not all_ys:
        return []
    x_min_nat, x_max_nat = min(all_xs), max(all_xs)
    leg_h = (y_max - y_min) * (1 + padding)
    leg_h = max(leg_h, out_h * 0.3)
    target_ratio = out_w / out_h
    target_w = leg_h * target_ratio
    cx = (x_min_nat + x_max_nat) / 2
    half_w = target_w / 2
    x1 = int(cx - half_w)
    y1 = int(y_min - (y_max - y_min) * padding / 2)
    x2 = int(cx + half_w)
    y2 = int(y_max + (y_max - y_min) * padding / 2)

    tag = part_cfg.get("tag", "full_leg_focus, long_legs, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "full_leg_focus")
    p = _crop_and_save_rect(img, x1, y1, x2, y2, output_dir, stem, suffix, "full_leg", 0, out_w, out_h, tag)
    return [(p, label)] if p else []


def _detect_and_crop_face_mesh(
    img, rgb, face_mesh_detector, stem: str, suffix: str,
    output_dir: Path, crop_size: int, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """
    精確臉部收割：Face Mesh，10=額頂+15% 152=下巴+5%，234/454=左右，以鼻尖(1)為中心 1:1。
    依嘴部 13-14 距離自動加 open_mouth。
    """
    if not face_mesh_detector:
        return []
    results = face_mesh_detector.process(rgb)
    if not results.multi_face_landmarks:
        return []
    h, w = img.shape[:2]
    tag = part_cfg.get("tag", "face_focus, beautiful_detailed_face, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "face_focus")
    out_list = []
    for idx, face_lm in enumerate(results.multi_face_landmarks):
        lm = face_lm.landmark
        if len(lm) < 455:
            continue
        y10 = lm[_FACE_MESH_TOP].y * h
        y152 = lm[_FACE_MESH_BOTTOM].y * h
        x234 = lm[_FACE_MESH_LEFT].x * w
        x454 = lm[_FACE_MESH_RIGHT].x * w
        face_h = max(y152 - y10, 1)
        y_min = y10 - face_h * _FACE_MESH_TOP_EXTEND
        y_max = y152 + face_h * _FACE_MESH_BOTTOM_EXTEND
        x_min = min(x234, x454)
        x_max = max(x234, x454)
        rect_w = x_max - x_min
        rect_h = y_max - y_min
        side = max(rect_w, rect_h)
        nx = lm[_FACE_MESH_NOSE].x * w
        ny = lm[_FACE_MESH_NOSE].y * h
        x1 = int(nx - side / 2)
        y1 = int(ny - side / 2)
        x2 = int(nx + side / 2)
        y2 = int(ny + side / 2)
        mouth_dist = 0
        if len(lm) > max(_FACE_MESH_MOUTH_UPPER, _FACE_MESH_MOUTH_LOWER):
            m13 = lm[_FACE_MESH_MOUTH_UPPER]
            m14 = lm[_FACE_MESH_MOUTH_LOWER]
            mouth_dist = ((m13.x - m14.x) ** 2 + (m13.y - m14.y) ** 2) ** 0.5
        extra = None
        mouth_thresh = max(_OPEN_MOUTH_DIST_RATIO, (face_h / h) * 0.08)
        if mouth_dist > mouth_thresh:
            extra = "open_mouth"
        p = _crop_and_save(img, x1, y1, x2, y2, output_dir, stem, suffix, "face_mesh", idx, crop_size, tag, extra_tags=extra)
        if p:
            out_list.append((p, label))
    return out_list


def _detect_and_crop_faces(
    img, rgb, face_detector, stem: str, suffix: str,
    output_dir: Path, crop_size: int, padding: float, part_cfg: dict,
) -> list[tuple[Path, str]]:
    """偵測臉部並裁切"""
    if not face_detector:
        return []
    results = face_detector.process(rgb)
    if not results.detections:
        return []
    h, w = img.shape[:2]
    tag = part_cfg.get("tag", "face_focus, masterpiece")
    label = (tag.split(",")[0].strip() if isinstance(tag, str) else "face_focus")
    out_list = []
    for idx, det in enumerate(results.detections):
        bbox = det.location_data.relative_bounding_box
        x_min = bbox.xmin * w
        y_min = bbox.ymin * h
        bw = bbox.width * w
        bh = bbox.height * h
        side = max(bw, bh) * (1 + padding)
        cx = x_min + bw / 2
        cy = y_min + bh / 2
        x1, y1 = int(cx - side / 2), int(cy - side / 2)
        x2, y2 = int(x1 + side), int(y1 + side)
        p = _crop_and_save(img, x1, y1, x2, y2, output_dir, stem, suffix, "face", idx, crop_size, tag)
        if p:
            out_list.append((p, label))
    return out_list


def _process_image_multi(
    img_path: Path,
    dst_base: Path,
    crop_size: int,
    padding: float,
    min_resolution: int,
    detect_parts: list[str],
    parts_map: dict,
    hands_detector,
    pose_detector,
    face_detector,
    face_mesh_detector,
) -> list[tuple[Path, str]]:
    """
    對單張圖片執行多部位偵測與裁切。
    detect_parts: ["hand", "feet", "face"] 的子集
    """
    if not _HAS_CV2:
        return []
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]
    if w < min_resolution or h < min_resolution:
        return []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    stem = img_path.stem
    suffix = img_path.suffix.lower() or ".png"
    results = []
    hand_cfg = parts_map.get("hand", {"folder": "10_hand_crop", "tag": "hand_focus, masterpiece"})
    fingers_cfg = parts_map.get("fingers", {"folder": "10_fingers_crop", "tag": "fingers_focus, masterpiece"})
    feet_cfg = parts_map.get("feet", {"folder": "10_feet_crop", "tag": "feet_focus, masterpiece"})
    toes_cfg = parts_map.get("toes", {"folder": "10_toes_crop", "tag": "toes_focus, masterpiece"})
    ankle_toe_cfg = parts_map.get("ankle_toe", {"folder": "10_ankle_toe_crop", "tag": "ankle_toe_focus, masterpiece"})
    full_leg_cfg = parts_map.get("full_leg", {"folder": "10_full_leg_crop", "tag": "full_leg_focus, masterpiece"})
    face_cfg = parts_map.get("face", {"folder": "10_face_crop", "tag": "face_focus, masterpiece"})
    face_mesh_cfg = parts_map.get("face_mesh", {"folder": "10_face_mesh_crop", "tag": "face_focus, masterpiece"})

    if "hand" in detect_parts and hands_detector:
        hand_dst = dst_base / hand_cfg.get("folder", "10_hand_crop")
        results.extend(_detect_and_crop_hands(
            img, rgb, hands_detector, stem, suffix, hand_dst, crop_size, padding, hand_cfg
        ))
    if "fingers" in detect_parts and hands_detector:
        fingers_dst = dst_base / fingers_cfg.get("folder", "10_fingers_crop")
        results.extend(_detect_and_crop_fingers(
            img, rgb, hands_detector, stem, suffix, fingers_dst, crop_size, padding, fingers_cfg
        ))
    if "feet" in detect_parts and pose_detector:
        feet_dst = dst_base / feet_cfg.get("folder", "10_feet_crop")
        results.extend(_detect_and_crop_feet(
            img, rgb, pose_detector, stem, suffix, feet_dst, crop_size, padding, feet_cfg
        ))
    if "toes" in detect_parts and pose_detector:
        toes_dst = dst_base / toes_cfg.get("folder", "10_toes_crop")
        results.extend(_detect_and_crop_toes(
            img, rgb, pose_detector, stem, suffix, toes_dst, crop_size, padding, toes_cfg
        ))
    if "ankle_toe" in detect_parts and pose_detector:
        ankle_toe_dst = dst_base / ankle_toe_cfg.get("folder", "10_ankle_toe_crop")
        results.extend(_detect_and_crop_ankle_toe(
            img, rgb, pose_detector, stem, suffix, ankle_toe_dst, crop_size, padding, ankle_toe_cfg
        ))
    if "full_leg" in detect_parts and pose_detector:
        full_leg_dst = dst_base / full_leg_cfg.get("folder", "10_full_leg_crop")
        results.extend(_detect_and_crop_full_leg(
            img, rgb, pose_detector, stem, suffix, full_leg_dst, padding, full_leg_cfg
        ))
    if "face" in detect_parts and face_detector:
        face_dst = dst_base / face_cfg.get("folder", "10_face_crop")
        results.extend(_detect_and_crop_faces(
            img, rgb, face_detector, stem, suffix, face_dst, crop_size, padding, face_cfg
        ))
    if "face_mesh" in detect_parts and face_mesh_detector:
        face_mesh_dst = dst_base / face_mesh_cfg.get("folder", "10_face_mesh_crop")
        results.extend(_detect_and_crop_face_mesh(
            img, rgb, face_mesh_detector, stem, suffix, face_mesh_dst, crop_size, face_mesh_cfg
        ))

    return results


def run_smart_crop_active(
    source_folder: str,
    dest_folder: str,
    crop_size: int = 512,
    padding: float = 0.3,
    min_resolution: int = 512,
    recursive: bool = True,
    target_part: str = "",
    full_auto: bool = False,
) -> Generator[tuple[str, list[tuple[str, str]]], None, None]:
    """
    視覺主動偵測。
    full_auto=True: 全自動掃描，同時偵測手腳臉。
    full_auto=False: 依 target_part 偵測單一部位（手部/腳部）。
    """
    if not _HAS_CV2:
        yield "❌ 需要 opencv-python", []
        return
    if not (_HAS_MEDIAPIPE or _HAS_POSE or _HAS_FACE or _HAS_FACE_MESH):
        yield "❌ 需要 mediapipe。請嘗試：pip install mediapipe==0.10.9", []
        return

    if full_auto:
        detect_parts = []
        if _HAS_MEDIAPIPE:
            detect_parts.append("hand")
        if _HAS_POSE:
            detect_parts.append("feet")
        if _HAS_FACE:
            detect_parts.append("face")
        if not detect_parts:
            yield "❌ 全自動掃描需要 mediapipe hands、pose、face_detection", []
            return
        part_desc = "手、腳、臉"
    else:
        tid = str(target_part or "").strip()
        if tid == "feet" and _HAS_POSE:
            detect_parts = ["feet"]
            part_desc = "腳部"
        elif tid == "toes" and _HAS_POSE:
            detect_parts = ["toes"]
            part_desc = "腳趾"
        elif tid == "face" and _HAS_FACE:
            detect_parts = ["face"]
            part_desc = "臉部"
        elif tid == "face_mesh" and _HAS_FACE_MESH:
            detect_parts = ["face_mesh"]
            part_desc = "精確臉部"
        elif tid == "fingers" and _HAS_MEDIAPIPE:
            detect_parts = ["fingers"]
            part_desc = "手指"
        elif tid == "full_leg" and _HAS_POSE:
            detect_parts = ["full_leg"]
            part_desc = "長腿"
        elif tid == "ankle_toe" and _HAS_POSE:
            detect_parts = ["ankle_toe"]
            part_desc = "腳踝至腳尖"
        else:
            detect_parts = ["hand"] if _HAS_MEDIAPIPE else []
            part_desc = "手部"

    hands_d = _get_hands_detector() if ("hand" in detect_parts or "fingers" in detect_parts) else None
    pose_d = _get_pose_detector() if ("feet" in detect_parts or "toes" in detect_parts or "full_leg" in detect_parts or "ankle_toe" in detect_parts) else None
    face_d = _get_face_detector() if "face" in detect_parts else None
    face_mesh_d = _get_face_mesh_detector() if "face_mesh" in detect_parts else None
    parts_map = _build_parts_map()

    src = Path(source_folder)
    dst_base = Path(dest_folder)
    if not src.exists() or not src.is_dir():
        yield f"❌ 找不到來源資料夾: {src}", []
        return

    if recursive:
        img_files = []
        for ext in IMG_EXT:
            img_files.extend(src.rglob(f"*{ext}"))
    else:
        img_files = []
        for ext in IMG_EXT:
            img_files.extend([f for f in src.glob(f"*{ext}") if f.is_file()])
    img_files = sorted(set(img_files))

    yield f"🔄 視覺主動偵測（{part_desc}）：掃描 {len(img_files)} 張圖片...", []

    total_crops = 0
    previews: list[tuple[str, str]] = []

    for img_path in img_files:
        try:
            results = _process_image_multi(
                img_path, dst_base, crop_size, padding, min_resolution,
                detect_parts, parts_map, hands_d, pose_d, face_d, face_mesh_d,
            )
            if results:
                for out_path, label in results:
                    total_crops += 1
                    previews.append((str(out_path), label))
                part_names = ", ".join(set(r[1] for r in results))
                yield f"✓ [{img_path.name}] → 偵測到 {part_names}，裁切 {len(results)} 張", previews[-20:]
            else:
                yield f"⊘ [{img_path.name}] 未偵測到目標部位", previews[-20:]
        except Exception as e:
            yield f"⚠️ [{img_path.name}] 錯誤: {e}", previews[-20:]

    yield f"\n✅ 視覺主動偵測（{part_desc}）完成，共裁切 {total_crops} 張", previews


def run_smart_crop_active_sync(
    source_folder: str,
    dest_folder: str,
    crop_size: int = 512,
    padding: float = 0.3,
    min_resolution: int = 512,
    recursive: bool = True,
    target_part: str = "",
    full_auto: bool = False,
    log_callback: Callable[[str], None] | None = None,
) -> tuple[int, list[tuple[Path, str]]]:
    """同步版"""
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    previews: list[tuple[Path, str]] = []
    for log_line, pr in run_smart_crop_active(
        source_folder, dest_folder, crop_size, padding, min_resolution,
        recursive, target_part, full_auto
    ):
        log(log_line)
        if pr:
            previews = [(Path(p), label) for p, label in pr]
    return len(previews), previews
