"""
處理模組：整合資料庫與翻譯 API，執行標籤掃描、翻譯、分類、輸出
"""
from pathlib import Path
from collections import Counter

from database_manager import (
    load_tag_database, load_tag_map, save_tags_to_json, update_tag_database,
    get_category, scan_txt_files, extract_tags_from_file, tag_map,
    CATEGORIES, TAGS_DB_FILE, DB_FILE,
)
from ollama_client import (
    init_gemini, get_ollama_client, get_ai_translation, get_gemini_translation,
    batch_translate_and_classify_gemini, batch_translate_and_classify_ollama,
)


def process_with_ai(folder_path, ollama_url, model_name, output_file, enable_classification=False,
                    translation_mode="ollama", gemini_api_key="",
                    log_callback=None, progress_callback=None):
    """
    主要處理函數
    參數:
        translation_mode: "ollama" 或 "gemini"
        gemini_api_key: Gemini API Key（僅 gemini 模式需要）
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)

    def update_progress(value, max_value):
        if progress_callback:
            progress_callback(value, max_value)

    use_gemini = (translation_mode == "gemini")

    if use_gemini:
        if not gemini_api_key:
            log("❌ 請輸入 Gemini API Key")
            return False
        log("🔧 正在初始化 Gemini API（新版 google-genai SDK）...")
        gemini_client, actual_model = init_gemini(gemini_api_key, log_callback=log)
        if gemini_client is None:
            log("❌ Gemini 初始化失敗，已終止")
            return False
        client = None
    else:
        log("🔧 正在連接到 Ollama 服務...")
        client, actual_model = get_ollama_client(ollama_url, model_name, log_callback=log)
        if client is None:
            log("❌ 無法連接到 Ollama 服務")
            log("💡 提示：請確認 Ollama 服務正在運行（執行: ollama serve）")
            return False

    log("=" * 70)
    log("🚀 開始掃描標籤檔...")
    log("=" * 70)

    txt_files = scan_txt_files(folder_path, log_callback=log)
    if not txt_files:
        log("❌ 沒有找到任何 .txt 標籤檔！")
        return False

    log(f"\n📊 正在讀取並統計標籤...")
    all_tags = []
    for i, txt_file in enumerate(txt_files):
        tags = extract_tags_from_file(txt_file)
        all_tags.extend(tags)
        if (i + 1) % 50 == 0:
            update_progress(i + 1, len(txt_files))

    if not all_tags:
        log("❌ 沒有找到任何標籤！")
        return False

    tag_counts = Counter(all_tags)
    total_unique_tags = len(tag_counts)
    log(f"✅ 共找到 {total_unique_tags} 個不重複的標籤")

    db = load_tag_database()
    tag_map_local = dict(tag_map)
    category_map = {}
    if db:
        for en, item in db.items():
            tag_map_local[en] = item.get('zh_tag', '')
            if item.get('category'):
                category_map[en] = item['category']
        log(f"📂 已載入 {len(db)} 筆既有資料，智慧跳過重複並更新「其他」類")

    OTHER_LIKE = (None, '', '其他 (General)', '其他')
    VALID_TRANSLATION = lambda t: t and t not in ('', '未翻譯', '（未翻譯）')
    sorted_tags = [tag for tag, _ in tag_counts.most_common()]

    tags_need_translate = [t for t in sorted_tags
                          if t not in db or not VALID_TRANSLATION(db[t].get('zh_tag'))]
    tags_reclassify_only = [t for t in sorted_tags
                           if t in db
                           and VALID_TRANSLATION(db[t].get('zh_tag'))
                           and db[t].get('category') in OTHER_LIKE]

    engine = "Gemini" if use_gemini else f"Ollama ({actual_model})"
    log(f"\n🌐 正在使用 {engine} 翻譯標籤...")
    log("=" * 70)

    if tags_need_translate:
        log(f"📦 需要翻譯的標籤: {len(tags_need_translate)} 個")
        batch_size = 80
        for i in range(0, len(tags_need_translate), batch_size):
            chunk = tags_need_translate[i:i + batch_size]
            if use_gemini:
                batch_result = batch_translate_and_classify_gemini(chunk, gemini_api_key, actual_model, log_callback=log)
            else:
                batch_result = batch_translate_and_classify_ollama(client, actual_model, chunk, log_callback=log)
            for en, data in batch_result.items():
                tag_map_local[en] = data.get('zh_tag', '（未翻譯）')
                if data.get('category'):
                    category_map[en] = data['category']

    if tags_reclassify_only:
        log(f"🔄 保留翻譯、重新分類的標籤: {len(tags_reclassify_only)} 個")

    tag_map.update(tag_map_local)

    def resolve_category(tag):
        kw_cat = get_category(tag)
        if kw_cat != "其他 (General)":
            return kw_cat
        return category_map.get(tag) or kw_cat

    def get_local_translation(tag):
        if tag not in tag_map:
            if use_gemini:
                tag_map[tag] = get_gemini_translation(tag, gemini_client, actual_model, log_callback=log)
            else:
                tag_map[tag] = get_ai_translation(tag, client, actual_model, log_callback=log)
        return tag_map[tag]

    try:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as out:
            if enable_classification:
                classified = {cat: [] for cat in CATEGORIES.keys()}
                classified["其他 (General)"] = []
                for tag, count in tag_counts.most_common():
                    category = resolve_category(tag)
                    chinese = get_local_translation(tag)
                    line = f"{tag.ljust(30)} | {str(count).ljust(5)} | {chinese}"
                    classified[category].append(line)
                    log(f"[{category}] {tag} -> {chinese}")
                cat_order = ["角色與作品 (Character)"] + [c for c in CATEGORIES.keys() if c != "角色與作品 (Character)"] + ["其他 (General)"]
                for cat in cat_order:
                    lines = classified.get(cat, [])
                    if lines:
                        out.write(f"\n● {cat}\n")
                        out.write("-" * 60 + "\n")
                        out.write("\n".join(lines) + "\n")
            else:
                for tag, count in tag_counts.most_common():
                    chinese = get_local_translation(tag)
                    line = f"{tag.ljust(30)} | {str(count).ljust(5)} | {chinese}\n"
                    out.write(line)
                    log(f"已翻譯: {tag} -> {chinese}")

        log("=" * 70)
        log(f"✨ 完成！結果已儲存至：{output_file}")
        log(f"📝 總共處理了 {len(txt_files)} 個檔案，{total_unique_tags} 個不重複標籤")
        character = Path(folder_path).resolve().parent.name
        new_tags_list = [
            {"en_tag": tag, "zh_tag": tag_map.get(tag, "未翻譯"), "count": count, "category": resolve_category(tag)}
            for tag, count in tag_counts.items()
        ]
        save_tags_to_json(tag_counts, tag_map, TAGS_DB_FILE, log_callback=log, category_getter=resolve_category)
        update_tag_database(new_tags_list, DB_FILE, character=character, log_callback=log)
        log("=" * 70)
        return True

    except RuntimeError as e:
        log(f"❌ 已終止: {e}")
        return False
    except Exception as e:
        log(f"❌ 寫入檔案失敗: {e}")
        return False
