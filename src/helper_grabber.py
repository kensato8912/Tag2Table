"""
Helper Grabber：規則驅動的參考圖篩選，依 GRABBER_RULES 自動分類複製。
複製時 .txt 會剔除主角色觸發詞，僅保留該規則相關標籤，避免訓練污染。
品質過濾：排除負面標籤、解析度不足的圖片。
風格過濾：非動漫風格圖片移至 data/rejected_by_style，確保純 2D 訓練集。
"""
import shutil
from pathlib import Path

from PIL import Image

_PROJECT_ROOT = Path(__file__).parent.parent
REJECTED_BY_STYLE_DIR = _PROJECT_ROOT / "data" / "rejected_by_style"

# 寫實／非動漫風格關鍵字：含這些則不採用，改移至 rejected_by_style
DISCARD_TAGS = ["photorealistic", "cosplay", "3d", "real life", "realistic"]

# 負面／低品質標籤：含任一則排除
BAD_QUALITY_TAGS = [
    "bad anatomy", "bad hands", "bad fingers", "bad proportions",
    "watermark", "text", "signature", "username",
    "low quality", "worst quality", "lowres", "blurry", "jpeg artifacts",
    "out of focus", "ugly", "deformed", "mutation", "extra limbs",
]

# 最低解析度：長寬皆須大於此值（px）才採用
MIN_RESOLUTION = 512

# 規則驅動：定義 folder_name、keywords、priority 即可自動分家
# 可隨時增加新分類
GRABBER_RULES = {
    "hands": {
        "folder_name": "10_hands_ref",
        "keywords": ["hand", "hands", "finger", "fingers", "palm", "holding", "gesture", "hand_focus", "reaching"],
        "priority": 10,
    },
    "feet": {
        "folder_name": "10_feet_ref",
        "keywords": ["foot", "feet", "toe", "toes", "sole", "soles", "feet_focus", "barefoot"],
        "priority": 10,
    },
    "eyes": {
        "folder_name": "5_eyes_ref",
        "keywords": ["eye_focus", "beautiful_detailed_eyes", "heterochromia", "eyes"],
        "priority": 5,
    },
    "clothing_details": {
        "folder_name": "8_clothes_ref",
        "keywords": ["frills", "lace", "ribbon", "buttons_detail", "sleeve", "collar"],
        "priority": 8,
    },
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _read_tags_from_txt(txt_path: Path) -> list[str]:
    """讀取 .txt 標籤檔，回傳標籤列表（含跨行合併後以逗號分割）"""
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


def is_anime_style(tags: list[str]) -> bool:
    """若標籤含有寫實／非動漫關鍵字則回傳 False（不採用，改移至 rejected_by_style）"""
    tags_str = " ".join(t.lower() for t in tags)
    if any(dt in tags_str for dt in DISCARD_TAGS):
        return False
    return True


def has_bad_quality_tags(tags: list[str]) -> bool:
    """若標籤含有負面／低品質關鍵字則回傳 True（不採用）"""
    tags_str = " ".join(t.lower() for t in tags)
    return any(bq in tags_str for bq in BAD_QUALITY_TAGS)


def _meets_resolution(img_path: Path, min_size: int = MIN_RESOLUTION) -> bool:
    """檢查圖片長寬是否皆大於 min_size（px）"""
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            return w > min_size and h > min_size
    except Exception:
        return False


def _rule_matches(tags: list[str], rule: dict) -> bool:
    """檢查標籤是否符合該規則的關鍵字"""
    tags_str = " ".join(t.lower() for t in tags)
    keywords = rule.get("keywords", [])
    return any(kw in tags_str for kw in keywords)


def _filter_tags_for_rule(
    tags: list[str],
    rule: dict,
    trigger_words: list[str],
) -> list[str]:
    """
    剔除觸發詞，僅保留該規則關鍵字相關的標籤，避免訓練污染。
    """
    triggers_lower = {t.strip().lower() for t in trigger_words if t.strip()}
    keywords_lower = {kw.lower() for kw in rule.get("keywords", [])}
    filtered = []
    for tag in tags:
        t_lower = tag.strip().lower()
        if t_lower in triggers_lower:
            continue
        if any(kw in t_lower for kw in keywords_lower):
            filtered.append(tag)
    return filtered


def grab_hand_feet_refs(
    source_folder: str,
    dest_folder: str,
    trigger_words: list[str] | None = None,
    recursive: bool = True,
    rules: dict | None = None,
    log_callback=None,
) -> int:
    """
    依規則掃描來源資料夾，若標籤符合某規則且為動漫風格，
    則複製圖片與過濾後的 .txt 到該規則對應的子資料夾。
    回傳複製的圖片總數量（同一張圖若符合多規則會計多次）。
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    src = Path(source_folder)
    dst_base = Path(dest_folder)
    if not src.exists() or not src.is_dir():
        log(f"❌ 找不到來源資料夾: {src}")
        return 0

    triggers = trigger_words or ["Niyaniya", "Ibuki", "niyaniya", "ibuki"]
    grabber_rules = rules if rules is not None else GRABBER_RULES

    # 依 priority 排序（愈小愈先處理）
    sorted_rules = sorted(
        grabber_rules.items(),
        key=lambda x: x[1].get("priority", 999),
    )

    if recursive:
        txt_files = list(src.rglob("*.txt"))
    else:
        txt_files = [f for f in src.glob("*.txt") if f.is_file()]

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
        if not is_anime_style(tags):
            REJECTED_BY_STYLE_DIR.mkdir(parents=True, exist_ok=True)
            dest_rej_img = REJECTED_BY_STYLE_DIR / img_file.name
            dest_rej_txt = REJECTED_BY_STYLE_DIR / txt_path.name
            try:
                shutil.copy2(img_file, dest_rej_img)
                shutil.copy2(txt_path, dest_rej_txt)
                log(f"⚠️ 跳過非動漫風格圖片: {img_file.name}")
            except Exception as e:
                log(f"⚠️ 跳過非動漫風格圖片: {img_file.name} (搬移至 rejected 失敗: {e})")
            continue
        if has_bad_quality_tags(tags):
            continue
        if not _meets_resolution(img_file):
            continue

        for rule_id, rule in sorted_rules:
            if not _rule_matches(tags, rule):
                continue

            folder_name = rule.get("folder_name", rule_id)
            rule_dst = dst_base / folder_name
            rule_dst.mkdir(parents=True, exist_ok=True)

            filtered_tags = _filter_tags_for_rule(tags, rule, triggers)
            filtered_line = ", ".join(filtered_tags) if filtered_tags else ""

            dest_img = rule_dst / img_file.name
            dest_txt = rule_dst / txt_path.name

            try:
                shutil.copy2(img_file, dest_img)
                dest_txt.write_text(filtered_line, encoding="utf-8")
                count += 1
                log(f"✓ [{folder_name}] {img_file.name}")
            except Exception as e:
                log(f"⚠ 複製失敗 {img_file.name} → {folder_name}: {e}")

    return count
