"""
資料庫模組：處理 JSON 讀取、寫入、合併
包含：config、categories、tag_map、標籤資料庫、角色套裝
"""
import os
import json
import shutil
from pathlib import Path
from collections import Counter
from datetime import datetime

# 專案根目錄（src 的上一層）
_PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
TXT_OUTPUT_DIR = _PROJECT_ROOT / "txt"

def _ensure_data_dir():
    """確保 data 資料夾存在"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def _ensure_txt_dir():
    """確保 txt 輸出資料夾存在"""
    TXT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _migrate_if_needed(old_path, new_path):
    """若新位置無檔案但舊位置有，則搬遷（向後相容）"""
    if not new_path.exists() and old_path.exists():
        try:
            _ensure_data_dir()
            shutil.copy2(old_path, new_path)
        except Exception:
            pass


def _init_from_example_if_needed(real_path, example_name):
    """若實檔不存在但 .example.json 存在，則從範例複製建立"""
    example_path = DATA_DIR / example_name
    if not real_path.exists() and example_path.exists():
        try:
            _ensure_data_dir()
            shutil.copy2(example_path, real_path)
        except Exception:
            pass

# JSON 路徑（優先使用 data/，啟動時會檢查舊位置並遷移）
CONFIG_FILE = DATA_DIR / "config.json"
TAGS_DB_FILE = DATA_DIR / "tags_db.json"
DB_FILE = DATA_DIR / "all_characters_tags.json"
CATEGORIES_FILE = DATA_DIR / "categories.json"
TAG_MAP_FILE = DATA_DIR / "tag_map.json"
PROMPT_PRESETS_FILE = DATA_DIR / "prompt_presets.json"

# 啟動時執行一次遷移（舊路徑 → data/）
for _old, _new in [
    (_PROJECT_ROOT / "config.json", CONFIG_FILE),
    (_PROJECT_ROOT / "tags_db.json", TAGS_DB_FILE),
    (_PROJECT_ROOT / "all_characters_tags.json", DB_FILE),
    (_PROJECT_ROOT / "categories.json", CATEGORIES_FILE),
    (_PROJECT_ROOT / "tag_map.json", TAG_MAP_FILE),
    (_PROJECT_ROOT / "prompt_presets.json", PROMPT_PRESETS_FILE),
]:
    _migrate_if_needed(_old, _new)

# 若實檔不存在，從 .example.json 複製建立（首次 clone 後自動初始化）
for _real, _example in [
    (CONFIG_FILE, "config.example.json"),
    (TAG_MAP_FILE, "tag_map.example.json"),
    (CATEGORIES_FILE, "categories.example.json"),
    (PROMPT_PRESETS_FILE, "prompt_presets.example.json"),
    (TAGS_DB_FILE, "tags_db.example.json"),
    (DB_FILE, "all_characters_tags.example.json"),
]:
    _init_from_example_if_needed(_real, _example)

# 預設分類（若 JSON 不存在時使用）
DEFAULT_CATEGORIES = {
    "身體特徵 (Body)": ["hair", "eyes", "blush", "body", "thigh", "face", "skin", "lips", "breast",
                        "hand", "finger", "foot", "arm", "leg", "neck", "ear", "nose"],
    "衣服配件 (Clothing)": ["shirt", "skirt", "jacket", "hat", "gloves", "pantyhose", "uniform",
                            "shoes", "bow", "clothes", "dress", "sleeve", "collar", "ribbon", "cane", "beret"],
    "姿態動作 (Pose)": ["standing", "sitting", "lying", "holding", "looking", "smile", "open",
                        "spread", "raised", "bent", "crossed", "arms", "legs"],
    "背景環境 (Background)": ["background", "indoors", "outdoors", "scenery", "room", "white",
                              "simple", "sky", "cloud", "tree", "flower", "wall", "floor"],
    "光線風格 (Style)": ["light", "dark", "shadow", "glow", "sunlight", "sunset", "night",
                         "realistic", "anime", "solo", "masterpiece"],
    "角色與作品 (Character)": ["ibuki", "niyaniya", "professor", "blue archive", "blue_archive", "halo", "1girl", "2girls", "3girls"],
}


def load_categories():
    """從 categories.json 載入分類，不存在則建立預設檔並回傳"""
    try:
        if CATEGORIES_FILE.exists():
            with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
        save_categories(DEFAULT_CATEGORIES)
        return DEFAULT_CATEGORIES
    except Exception:
        return DEFAULT_CATEGORIES.copy()


def save_categories(categories):
    """將分類儲存至 categories.json"""
    try:
        _ensure_data_dir()
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# 載入分類（可隨 categories.json 更新）
CATEGORIES = load_categories()


def load_tag_map():
    """從 tag_map.json 載入基礎翻譯對照，不存在則建立預設檔並回傳"""
    default = {
        "niyaniya": "尼亞尼亞", "ibuki": "伊吹", "1girl": "1名女孩", "solo": "單人",
        "blue archive": "蔚藍檔案", "blue_archive": "蔚藍檔案", "Blue Archive": "蔚藍檔案",
    }
    try:
        if TAG_MAP_FILE.exists():
            with open(TAG_MAP_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        save_tag_map(default)
        return default
    except Exception:
        return default.copy()


def save_tag_map(tag_map_dict):
    """將基礎翻譯對照儲存至 tag_map.json"""
    try:
        _ensure_data_dir()
        with open(TAG_MAP_FILE, 'w', encoding='utf-8') as f:
            json.dump(tag_map_dict, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# 載入基礎翻譯（可隨 tag_map.json 更新）
tag_map = load_tag_map()


def load_config():
    """從 config.json 載入設定"""
    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_config(config):
    """儲存設定至 config.json"""
    try:
        _ensure_data_dir()
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_existing_translations(json_path=None):
    """讀取 JSON 中已有的翻譯，下次執行可跳過已翻譯的"""
    path = str(json_path or DB_FILE)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return {item['en_tag']: item['zh_tag'] for item in data}
    except Exception:
        pass
    return {}


def load_tag_database(db_path=None):
    """載入完整標籤資料庫，回傳 {en_tag: {zh_tag, category, count, characters, ...}}"""
    path = str(db_path or DB_FILE)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return {item['en_tag']: item for item in json.load(f)}
    except Exception:
        pass
    return {}


def get_category(tag):
    """根據標籤關鍵字回傳所屬分類（含 \\( \\) 括號格式偵測）"""
    tag_lower = tag.lower()
    if "\\(" in tag or "\\)" in tag:
        return "角色與作品 (Character)"
    char_keywords = CATEGORIES.get("角色與作品 (Character)", [])
    if any(kw in tag_lower for kw in char_keywords):
        return "角色與作品 (Character)"
    for cat, keywords in CATEGORIES.items():
        if cat == "角色與作品 (Character)":
            continue
        if any(kw in tag_lower for kw in keywords):
            return cat
    return "其他 (General)"


def save_tags_to_json(tag_counts, tag_map_dict, output_path=None, log_callback=None, category_getter=None):
    """將標籤、翻譯、次數、分類存成 JSON（單次覆寫，智慧分類）"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    get_cat = category_getter or get_category
    path = str(output_path or TAGS_DB_FILE)
    try:
        final_data = []
        for tag, count in tag_counts.items():
            category = get_cat(tag)
            translation = tag_map_dict.get(tag, "未翻譯")
            final_data.append({
                "en_tag": tag,
                "zh_tag": translation,
                "count": count,
                "category": category
            })
        _ensure_data_dir()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=4)
        log(f"✅ 標籤資料庫已儲存至: {path}")
    except Exception as e:
        log(f"⚠️ 儲存 JSON 失敗: {e}")


def update_tag_database(new_tags_list, db_path=None, character=None, log_callback=None):
    """
    合併更新標籤資料庫（累加次數、更新翻譯、記錄角色）
    new_tags_list 格式: [{"en_tag": "...", "zh_tag": "...", "count": 10, "category": "..."}, ...]
    character: 此次處理的角色/資料夾名稱，會寫入 characters 清單
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    path = str(db_path or DB_FILE)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                db = {item['en_tag']: item for item in json.load(f)}
        else:
            db = {}
        for item in new_tags_list:
            en = item['en_tag']
            if en in db:
                db[en]['count'] += item['count']
                db[en]['zh_tag'] = item['zh_tag']
                db[en]['category'] = item.get('category', db[en].get('category', '其他 (General)'))
                if character:
                    chars = db[en].setdefault('characters', [])
                    if character not in chars:
                        chars.append(character)
            else:
                entry = {k: v for k, v in item.items() if k != 'character'}
                entry['characters'] = [character] if character else []
                db[en] = entry
        _ensure_data_dir()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(list(db.values()), f, ensure_ascii=False, indent=4)
        log(f"✅ 資料庫已更新，目前共有 {len(db)} 個不重複標籤。")
        show_db_status(path, log_callback=log_callback)
    except Exception as e:
        log(f"⚠️ 更新資料庫失敗: {e}")


def show_db_status(db_file, log_callback=None):
    """顯示標籤資料庫狀態報告（檔案大小、總標籤數、分類統計）"""
    def out(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    db_path = str(db_file)
    if not os.path.exists(db_path):
        out("❌ 資料庫檔案尚未建立。")
        return
    try:
        file_size = os.path.getsize(db_path) / (1024 * 1024)
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        total_tags = len(data)
        cat_stats = Counter([item.get('category', '未分類') for item in data])
        out("\n" + "=" * 40)
        out("📊 【標籤資料庫狀態報告】")
        out(f"📂 檔案路徑: {os.path.abspath(db_path)}")
        out(f"💾 佔用容量: {file_size:.4f} MB")
        out(f"🏷️ 總標籤數: {total_tags} 條")
        out("-" * 40)
        out("🗂️ 分類統計:")
        for cat, count in sorted(cat_stats.items(), key=lambda x: -x[1]):
            out(f"   ● {str(cat).ljust(20)} : {count} 條")
        out("=" * 40 + "\n")
    except Exception as e:
        out(f"⚠️ 狀態報告失敗: {e}")


def scan_txt_files(root_folder, log_callback=None):
    """遞迴掃描資料夾內所有的 .txt 標籤檔"""
    txt_files = []
    root_path = Path(root_folder)
    if not root_path.exists():
        error_msg = f"❌ 錯誤：資料夾不存在 - {root_folder}"
        if log_callback:
            log_callback(error_msg)
        else:
            print(error_msg)
        return txt_files
    for txt_file in root_path.rglob('*.txt'):
        txt_files.append(txt_file)
    msg = f"📁 找到 {len(txt_files)} 個 .txt 標籤檔"
    if log_callback:
        log_callback(msg)
    else:
        print(msg)
    return txt_files


def extract_tags_from_file(file_path):
    """從單一 .txt 檔案中提取標籤（支援逗號分隔）"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            tags = [t.strip() for t in content.split(',') if t.strip()]
            return tags
    except Exception as e:
        print(f"⚠️ 讀取檔案失敗 {file_path}: {e}")
        return []


def generate_folder_report(folder_path, report_file=None, db_path=None, log_callback=None, open_after=True):
    """針對特定訓練資料夾生成報告（Notepad++ 格式）"""
    def out(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    path = Path(folder_path)
    out("📂 重新載入 JSON 資料庫以確保為最新...")
    db = load_tag_database(db_path)
    if not path.exists():
        out(f"❌ 找不到路徑: {folder_path}")
        return False
    txt_files = list(path.rglob('*.txt'))
    all_tags = []
    for f in txt_files:
        all_tags.extend(extract_tags_from_file(f))
    tag_counts = Counter(all_tags)
    if not tag_counts:
        out("❌ 此資料夾沒有找到任何標籤")
        return False
    categorized_data = {}
    for tag, count in tag_counts.items():
        info = db.get(tag, {"zh_tag": "未翻譯", "category": get_category(tag)})
        cat = info.get('category', '其他 (General)')
        if cat not in categorized_data:
            categorized_data[cat] = []
        categorized_data[cat].append({
            "en": tag, "zh": info.get('zh_tag', '未翻譯'), "count": count
        })
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_path = str(report_file or TXT_OUTPUT_DIR / "Current_Folder_Tag_Report.txt")
    report = [
        "╔════════════════════════════════════════════════════════════════╗",
        "║             當前訓練資料夾標籤分析報告 (Notepad++)             ║",
        "╠════════════════════════════════════════════════════════════════╣",
        f"  資料夾: {path.resolve()}",
        f"  分析時間: {now}",
        f"  標籤檔數: {len(txt_files)} 張",
        "╚════════════════════════════════════════════════════════════════╝\n"
    ]
    cat_order = ["角色與作品 (Character)"] + [c for c in CATEGORIES.keys() if c != "角色與作品 (Character)"] + ["其他 (General)"]
    for cat in cat_order:
        if cat not in categorized_data:
            continue
        report.append(f"● {cat}")
        report.append("-" * 75)
        report.append(f"{'英文標籤 (Tag)'.ljust(35)} | {'次數'.ljust(6)} | {'中文翻譯'}")
        report.append("-" * 75)
        items = sorted(categorized_data[cat], key=lambda x: x['count'], reverse=True)
        for item in items:
            report.append(f"{item['en'].ljust(35)} | {str(item['count']).ljust(6)} | {item['zh']}")
        report.append("")
    try:
        _ensure_txt_dir()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        out(f"✅ 資料夾報告已生成：{out_path}")
        if open_after and os.name == 'nt':
            os.startfile(out_path)
        return True
    except Exception as e:
        out(f"⚠️ 寫入報告失敗: {e}")
        return False


def generate_file_report(folder_path, report_file=None, db_path=None, log_callback=None, open_after=True):
    """逐檔標籤分類報告：針對每個 .txt 檔案，列出其標籤並按分類排序"""
    def out(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    path = Path(folder_path)
    out("📂 重新載入 JSON 資料庫以確保為最新...")
    db = load_tag_database(db_path)
    if not path.exists():
        out(f"❌ 找不到路徑: {folder_path}")
        return False
    txt_files = sorted(path.rglob('*.txt'), key=lambda p: p.name)
    if not txt_files:
        out("❌ 此資料夾沒有找到任何 .txt 檔")
        return False
    cat_order = ["角色與作品 (Character)"] + [c for c in CATEGORIES.keys() if c != "角色與作品 (Character)"] + ["其他 (General)"]
    report = [
        "╔════════════════════════════════════════════════════════════════╗",
        "║             逐檔標籤分類報告 (按分類排序)                      ║",
        "╠════════════════════════════════════════════════════════════════╣",
        f"  分析目錄: {path.resolve()}",
        f"  分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "╚════════════════════════════════════════════════════════════════╝\n"
    ]
    for txt_path in txt_files:
        f_name = txt_path.name
        tags = extract_tags_from_file(txt_path)
        file_categorized = {cat: [] for cat in cat_order}
        for tag in tags:
            info = db.get(tag, {})
            cat = info.get('category', get_category(tag))
            if cat not in file_categorized:
                cat = "其他 (General)"
            zh = info.get('zh_tag', '未翻譯')
            file_categorized[cat].append(f"{tag}({zh})")
        report.append(f"📄 檔名: [{f_name}]")
        report.append("-" * 70)
        for cat in cat_order:
            if file_categorized[cat]:
                tag_line = ", ".join(file_categorized[cat])
                report.append(f"  ● {cat.ljust(22)}: {tag_line}")
        report.append("\n" + "." * 70 + "\n")
    out_path = str(report_file or TXT_OUTPUT_DIR / "File_Based_Tag_Report.txt")
    try:
        _ensure_txt_dir()
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        out(f"✅ 逐檔報告已生成：{out_path} ({len(txt_files)} 個檔案)")
        if open_after and os.name == 'nt':
            os.startfile(out_path)
        return True
    except Exception as e:
        out(f"⚠️ 寫入報告失敗: {e}")
        return False


def load_tag_data_for_prompt(db_path=None):
    """讀取 all_characters_tags.json 並依分類組織成 {分類: [(en, zh), ...]}"""
    path = str(db_path or DB_FILE)
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    structured = {}
    for item in data:
        cat = item.get('category', '其他 (General)')
        en = item.get('en_tag', '')
        zh = item.get('zh_tag', '')
        if en:
            structured.setdefault(cat, []).append((en, zh))
    return structured


def load_prompt_presets():
    """從 prompt_presets.json 載入角色快速套裝"""
    default = {
        "Key 基礎款": ["key (blue archive)", "1girl", "long silver hair", "red eyes", "mechanical halo", "bangs"],
        "尼亞尼亞 基礎款": ["niyaniya (blue archive)", "1girl", "grey hair", "red eyes", "halo", "black coat", "beret"],
    }
    try:
        if PROMPT_PRESETS_FILE.exists():
            with open(PROMPT_PRESETS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        save_prompt_presets(default)
        return default
    except Exception:
        return default.copy()


def save_prompt_presets(presets):
    """儲存角色快速套裝至 prompt_presets.json"""
    try:
        _ensure_data_dir()
        with open(PROMPT_PRESETS_FILE, 'w', encoding='utf-8') as f:
            json.dump(presets, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
