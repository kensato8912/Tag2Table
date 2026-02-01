"""
智慧裁切：依標籤 hand_focus / feet_focus 自動裁切並輸出至訓練用資料夾。
手部：MediaPipe Hands 偵測 bounding box + 邊緣擴張。
腳部：OpenCV 邊緣偵測或圖像中心點裁切。
"""
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

try:
    import mediapipe as mp
    _solutions = getattr(mp, "solutions", None)
    _mp_hands = getattr(_solutions, "hands", None) if _solutions else None
    _HAS_MEDIAPIPE = _mp_hands is not None
except (ImportError, AttributeError):
    _mp_hands = None
    _HAS_MEDIAPIPE = False

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HAND_FOCUS = "hand_focus"
FEET_FOCUS = "feet_focus"


def _read_tags_from_txt(txt_path: Path) -> list[str]:
    """讀取 .txt 標籤檔"""
    try:
        content = txt_path.read_text(encoding="utf-8").strip()
        tags = []
        for part in content.replace("\n", ",").split(","):
            t = part.strip()
            if t:
                tags.append(t)
        return tags
    except Exception:
        return []


def _filter_tags_for_crop(tags: list[str], trigger_words: list[str], focus_tag: str) -> list[str]:
    """
    刪除角色觸發詞，開頭加入 hand_focus 或 feet_focus。
    """
    triggers_lower = {t.strip().lower() for t in trigger_words if t.strip()}
    filtered = []
    for tag in tags:
        t_lower = tag.strip().lower()
        if t_lower in triggers_lower:
            continue
        filtered.append(tag.strip())
    return [focus_tag] + filtered


def _find_hand_bbox_mediapipe(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """使用 MediaPipe Hands 偵測手部 bbox，回傳 (x1, y1, x2, y2) 或 None。img 為 RGB。"""
    if not _HAS_MEDIAPIPE:
        return None
    with _mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        h, w = img.shape[:2]
        if len(img.shape) == 2 and _HAS_CV2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img
        results = hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None
        all_x, all_y = [], []
        for lm in results.multi_hand_landmarks:
            for p in lm.landmark:
                all_x.append(int(p.x * w))
                all_y.append(int(p.y * h))
        if not all_x or not all_y:
            return None
        x1, x2 = max(0, min(all_x) - 20), min(w, max(all_x) + 20)
        y1, y2 = max(0, min(all_y) - 20), min(h, max(all_y) + 20)
        return (x1, y1, x2, y2)


def _find_best_crop_region_cv2(img: np.ndarray, crop_size: int) -> tuple[int, int]:
    """
    使用 Laplacian 邊緣偵測找出最高細節區域的中心 (x, y)。
    在網格上採樣候選點，取 Laplacian 變異數最高者。
    """
    if not _HAS_CV2:
        h, w = img.shape[:2]
        return w // 2, h // 2

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape

    step = max(crop_size // 2, 64)
    best_var = -1
    best_cx, best_cy = w // 2, h // 2

    for y in range(crop_size // 2, h - crop_size // 2, step):
        for x in range(crop_size // 2, w - crop_size // 2, step):
            x1 = max(0, x - crop_size // 2)
            y1 = max(0, y - crop_size // 2)
            x2 = min(w, x1 + crop_size)
            y2 = min(h, y1 + crop_size)
            roi = gray[y1:y2, x1:x2]
            if roi.size < crop_size * crop_size * 0.5:
                continue
            lap = cv2.Laplacian(roi, cv2.CV_64F)
            var = lap.var()
            if var > best_var:
                best_var = var
                best_cx = x
                best_cy = y

    return best_cx, best_cy


def _crop_from_bbox_pil(img: Image.Image, bbox: tuple[int, int, int, int], size: int, padding_pct: float) -> Image.Image:
    """從 bbox 裁切，加入 padding 後 resize 至 size x size"""
    x1, y1, x2, y2 = bbox
    w, h = img.size
    bw, bh = x2 - x1, y2 - y1
    pad_w = int(bw * padding_pct / 100)
    pad_h = int(bh * padding_pct / 100)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    cropped = img.crop((x1, y1, x2, y2))
    return cropped.resize((size, size), Image.Resampling.LANCZOS)


def _crop_image_pil(img: Image.Image, cx: int, cy: int, size: int) -> Image.Image:
    """以 (cx, cy) 為中心裁切 size x size"""
    w, h = img.size
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(w, x1 + size)
    y2 = min(h, y1 + size)
    if x2 - x1 < size or y2 - y1 < size:
        x1 = max(0, w - size)
        y1 = max(0, h - size)
        x2 = x1 + size
        y2 = y1 + size
        if x2 > w or y2 > h:
            return img.resize((size, size), Image.Resampling.LANCZOS)
    return img.crop((x1, y1, x2, y2))


def run_crop(
    source_folder: str,
    dest_folder: str,
    crop_size: int = 512,
    min_resolution: int = 512,
    padding_pct: float = 30,
    trigger_words: list[str] | None = None,
    use_mediapipe_hands: bool = False,
    use_edge_detection: bool = True,
    target_part: str | None = None,
    log_callback: Callable[[str], None] | None = None,
    preview_count: int = 6,
) -> tuple[int, list[tuple[Path, str]]]:
    """
    掃描來源資料夾，對含 hand_focus 或 feet_focus 的圖片執行裁切。
    回傳 (總裁切數, [(裁切圖路徑, 類別), ...]) 供預覽用。
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    triggers = trigger_words or ["Niyaniya", "Ibuki"]
    log("🔄 掃描中...")
    src = Path(source_folder)
    dst_base = Path(dest_folder)
    if not src.exists() or not src.is_dir():
        log(f"❌ 找不到來源資料夾: {src}")
        return 0, []

    hand_dst = dst_base / "10_hand_crop"
    feet_dst = dst_base / "10_feet_crop"
    hand_dst.mkdir(parents=True, exist_ok=True)
    feet_dst.mkdir(parents=True, exist_ok=True)

    txt_files = list(src.rglob("*.txt"))
    preview_results: list[tuple[Path, str]] = []
    count = 0

    for txt_path in txt_files:
        img_base = txt_path.with_suffix("")
        img_file = None
        for ext in IMG_EXT:
            cand = img_base.with_suffix(ext)
            if cand.exists():
                img_file = cand
                break
        if not img_file:
            continue

        tags = _read_tags_from_txt(txt_path)
        tags_str = " ".join(t.lower() for t in tags)

        focus_tag = None
        out_sub = None
        if HAND_FOCUS in tags_str:
            focus_tag = HAND_FOCUS
            out_sub = hand_dst
        elif FEET_FOCUS in tags_str:
            focus_tag = FEET_FOCUS
            out_sub = feet_dst

        if not focus_tag or not out_sub:
            continue
        if target_part:
            if "手部" in target_part and focus_tag != HAND_FOCUS:
                continue
            if "腳部" in target_part and focus_tag != FEET_FOCUS:
                continue

        try:
            pil_img = Image.open(img_file).convert("RGB")
        except Exception as e:
            log(f"⚠️ 無法讀取 {img_file.name}: {e}")
            continue

        w, h = pil_img.size
        if w < min_resolution or h < min_resolution:
            log(f"⚠️ 跳過 {img_file.name}（尺寸 {w}x{h} 小於 {min_resolution}）")
            continue

        img_np = np.array(pil_img)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR) if _HAS_CV2 else np.stack([img_np] * 3, axis=-1)

        cropped = None
        if focus_tag == HAND_FOCUS and use_mediapipe_hands and _HAS_MEDIAPIPE and _HAS_CV2:
            bbox = _find_hand_bbox_mediapipe(img_np)
            if bbox:
                cropped = _crop_from_bbox_pil(pil_img, bbox, crop_size, padding_pct)

        if cropped is None:
            if use_edge_detection and _HAS_CV2:
                cx, cy = _find_best_crop_region_cv2(img_np, crop_size)
            else:
                cx, cy = w // 2, h // 2
            cropped = _crop_image_pil(pil_img, cx, cy, crop_size)
        if cropped.size[0] != crop_size or cropped.size[1] != crop_size:
            cropped = cropped.resize((crop_size, crop_size), Image.Resampling.LANCZOS)

        base_name = img_file.stem
        suffix = img_file.suffix.lower()
        out_name = f"{base_name}_crop{suffix}"
        out_img = out_sub / out_name
        out_txt = out_sub / f"{base_name}_crop.txt"

        filtered_tags = _filter_tags_for_crop(tags, triggers, focus_tag)
        tag_line = ", ".join(filtered_tags)

        try:
            cropped.save(out_img, quality=95)
            out_txt.write_text(tag_line, encoding="utf-8")
            count += 1
            log(f"✓ [{focus_tag}] {img_file.name} → {out_name}")

            if len(preview_results) < preview_count:
                preview_results.append((out_img, focus_tag))
        except Exception as e:
            log(f"⚠️ 裁切儲存失敗 {img_file.name}: {e}")

    log(f"\n✅ 裁切完成，共 {count} 張")
    return count, preview_results
