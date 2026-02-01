"""
WD14 Tagger：掃描資料夾內圖片，使用 ONNX Runtime (GPU) 生成 .txt 標籤檔
第一行可自訂觸發詞（如 Niyaniya）
"""
import csv
import os
import urllib.request
from pathlib import Path

import numpy as np

from .processor import sort_tags_by_category
import onnxruntime as ort
from PIL import Image

# --- 路徑設定 ---
_PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = _PROJECT_ROOT / "data" / "wd14"
MODEL_NAME = "wd-v1-4-moat-tagger-v2"
HF_BASE = "https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2/resolve/main"

# 支援的圖片副檔名
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _ensure_model_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _download_file(url: str, dest: Path, log_callback=None):
    """下載檔案，若已存在則跳過"""
    if dest.exists():
        return True
    _ensure_model_dir()
    try:
        if log_callback:
            log_callback(f"下載中: {dest.name}")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        if log_callback:
            log_callback(f"❌ 下載失敗 {dest.name}: {e}")
        return False


def _load_model_and_tags(log_callback=None):
    """載入 ONNX 模型與標籤 CSV，若不存在則自動下載"""
    model_path = MODEL_DIR / f"{MODEL_NAME}.onnx"
    csv_path = MODEL_DIR / f"{MODEL_NAME}.csv"

    if not model_path.exists():
        # HuggingFace 上的檔名為 model.onnx
        ok = _download_file(f"{HF_BASE}/model.onnx", model_path, log_callback)
        if not ok:
            raise FileNotFoundError(
                f"無法下載模型，請手動從 HuggingFace 下載 model.onnx 並命名為 {model_path.name} 放到 {MODEL_DIR}"
            )

    if not csv_path.exists():
        ok = _download_file(f"{HF_BASE}/selected_tags.csv", csv_path, log_callback)
        if not ok:
            raise FileNotFoundError(f"無法下載 selected_tags.csv，請手動放到 {MODEL_DIR}")

    # 選擇 GPU 或 CPU
    providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    session_providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
    if log_callback:
        log_callback(f"ONNX 執行: {session_providers[0]}")

    session = ort.InferenceSession(
        str(model_path),
        providers=session_providers,
    )

    # 解析 CSV
    tags = []
    general_idx = None
    character_idx = None
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            name = row[1].replace("_", " ")
            cat = row[2]
            tags.append(name)
            if general_idx is None and cat == "0":
                general_idx = len(tags) - 1
            elif character_idx is None and cat == "4":
                character_idx = len(tags) - 1

    return session, tags, general_idx or 0, character_idx or len(tags)


def _preprocess_image(img: Image.Image, size: int = 448) -> np.ndarray:
    """縮放、白邊填滿正方形，轉 BGR float32"""
    ratio = size / max(img.size)
    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (size, size), (255, 255, 255))
    x = (size - new_size[0]) // 2
    y = (size - new_size[1]) // 2
    square.paste(img, (x, y))
    arr = np.array(square, dtype=np.float32)
    arr = arr[:, :, ::-1]  # RGB -> BGR
    return np.expand_dims(arr, axis=0)


def tag_image(
    session: ort.InferenceSession,
    tags: list,
    general_idx: int,
    character_idx: int,
    img: Image.Image,
    threshold: float = 0.35,
    character_threshold: float = 0.85,
) -> list[str]:
    """對單張圖片預測標籤，回傳排序後的標籤列表"""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    size = session.get_inputs()[0].shape[1]

    arr = _preprocess_image(img, size)
    probs = session.run([output_name], {input_name: arr})[0][0]

    result = list(zip(tags, probs))
    general = [t for t, p in result[general_idx:character_idx] if p > threshold]
    character = [t for t, p in result[character_idx:] if p > character_threshold]

    # 括號內的括號需跳脫
    def escape(tag):
        return tag.replace("(", "\\(").replace(")", "\\)")

    return [escape(t) for t in character + general]


def tag_folder(
    folder: str,
    trigger_word: str = "Niyaniya",
    threshold: float = 0.35,
    character_threshold: float = 0.85,
    sort_by_category: bool = True,
    log_callback=None,
) -> int:
    """
    掃描資料夾內圖片，生成對應 .txt 標籤檔。
    觸發詞強制置於最前面（後跟 ", "），若 WD14 已偵測到觸發詞會先移除以免重複。
    回傳處理的圖片數量。
    """
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        if log_callback:
            log_callback(f"❌ 找不到資料夾: {folder}")
        return 0

    session, tags, general_idx, character_idx = _load_model_and_tags(log_callback)

    image_files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT]
    count = 0

    for img_path in sorted(image_files):
        txt_path = img_path.with_suffix(".txt")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            if log_callback:
                log_callback(f"⚠ 無法讀取 {img_path.name}: {e}")
            continue

        tag_list = tag_image(
            session, tags, general_idx, character_idx,
            img, threshold, character_threshold,
        )
        if sort_by_category:
            tag_list = sort_tags_by_category(tag_list)

        # 先移除 WD14 可能偵測到的觸發詞，避免重複
        trigger_clean = trigger_word.strip()
        if trigger_clean:
            tag_list = [t for t in tag_list if t.strip().lower() != trigger_clean.lower()]

        # 觸發詞強制插到最前面，後面跟 ", "
        if trigger_clean:
            content = trigger_clean + ", " + ", ".join(tag_list)
        else:
            content = ", ".join(tag_list)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(content)

        count += 1
        if log_callback:
            log_callback(f"✓ {img_path.name} → {txt_path.name}")

    return count


if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    trigger = sys.argv[2] if len(sys.argv) > 2 else "Niyaniya"
    n = tag_folder(folder, trigger_word=trigger, log_callback=print)
    print(f"\n✅ 完成，共處理 {n} 張圖片")
