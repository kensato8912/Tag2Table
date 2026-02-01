"""
網路模組：負責連線到 Ollama / Mac / 雲端 Gemini 翻譯標籤
支援本地 Ollama、遠端 Ollama（如 Mac）、Google Gemini API
"""
import json
import time
import urllib.request
import urllib.error
from openai import OpenAI

from .database_manager import CATEGORIES, get_category

# 統一翻譯 Prompt（Ollama / Gemini 共用）
TRANSLATE_PROMPT_BASE = (
    "你是一個蔚藍檔案(Blue Archive)與動漫專家。"
    "'Blue Archive' 譯為 '蔚藍檔案'，'ibuki' 譯為 '伊吹'，'niyaniya' 譯為 '尼亞尼亞'。"
)


def build_translate_prompt(tag=None, tags_text=None):
    """建立翻譯用 prompt，單一標籤或批次共用"""
    if tag is not None:
        return f"{TRANSLATE_PROMPT_BASE}請將 SD 繪圖標籤 '{tag}' 翻譯成簡短繁體中文。只回傳翻譯結果。"
    return f"{TRANSLATE_PROMPT_BASE}請翻譯以下標籤清單，一行對應一個，順序不變，只回傳翻譯結果。\n\n{tags_text}"


# 分類選項（與 AI 約定，需與 CATEGORIES 對應）
CLASSIFY_CATEGORIES = list(CATEGORIES.keys()) + ["其他 (General)"]


def build_translate_and_classify_prompt(tags_text, pre_sorted_hint=False):
    """建立「翻譯＋分類」合併 prompt，一次 API 回傳兩者"""
    cat_list = "、".join(CLASSIFY_CATEGORIES)
    order_hint = (
        "\n【重要】下列標籤已按「人物→衣裝→背景」排好序，請保持此順序進行通順的繁體中文翻譯。\n"
        if pre_sorted_hint
        else ""
    )
    return (
        f"{TRANSLATE_PROMPT_BASE}請翻譯以下 SD 繪圖標籤並分類。\n"
        f"{order_hint}"
        f"分類限定為：[{cat_list}]\n"
        "請嚴格依照格式回傳，一行一筆： 英文標籤 | 分類 | 中文翻譯\n"
        "例如：bangs | 身體特徵 (Body) | 劉海\n"
        f"待處理清單：\n{tags_text}"
    )


def _parse_translate_classify_response(text, tags_list):
    """解析 AI 回傳的 英文標籤 | 分類 | 中文翻譯 格式"""
    result = {}
    for line in (text or "").strip().split('\n'):
        if '|' not in line:
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 3:
            en, cat, zh = parts[0], parts[1], parts[2]
            if en in tags_list:
                if cat not in CLASSIFY_CATEGORIES:
                    for c in CLASSIFY_CATEGORIES:
                        if cat in c or c.startswith(cat):
                            cat = c
                            break
                    else:
                        cat = "其他 (General)"
                result[en] = {"zh_tag": zh, "category": cat}
    return result


# --- Ollama 連線 ---

def get_ollama_models(base_url, log_callback=None):
    """自動抓取 Ollama 已安裝的模型列表"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    model_list = []
    api_base = base_url.rstrip('/').replace('/v1', '')
    tags_url = f"{api_base}/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=5) as response:
            data = json.loads(response.read().decode())
            models = data.get('models', [])
            for m in models:
                name = m.get('name') or m.get('model', '')
                if name:
                    model_list.append(name)
    except Exception as e:
        log(f"⚠️ 無法取得模型列表: {e}")
    return model_list


def unload_model(base_url, model_name, log_callback=None):
    """將 Ollama 模型從顯存釋放"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    api_base = base_url.rstrip('/').replace('/v1', '')
    url = f"{api_base}/api/generate"
    data = json.dumps({"model": model_name, "prompt": ".", "stream": False, "keep_alive": 0}).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='POST',
                                 headers={'Content-Type': 'application/json'})
    try:
        urllib.request.urlopen(req, timeout=10)
        log(f"✅ 模型 {model_name} 已成功從顯存釋放！")
        return True
    except urllib.error.HTTPError as e:
        log(f"❌ 釋放失敗 (HTTP {e.code}): {e.read().decode()[:100]}")
        return False
    except Exception as e:
        log(f"❌ 釋放失敗: {e}")
        return False


def get_ollama_client(base_url, model_name, log_callback=None):
    """初始化 Ollama OpenAI 相容客戶端（支援本地 / 遠端 Mac）"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    try:
        client = OpenAI(
            base_url=base_url,
            api_key="ollama"
        )
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=10
            )
            if response and response.choices:
                log(f"✅ 成功連接到 Ollama，使用模型: {model_name}")
                return client, model_name
        except Exception as e:
            log(f"⚠️ 模型測試失敗: {str(e)[:100]}")
            log("💡 提示：請確認 Ollama 服務正在運行，且模型名稱正確")
            return None, None
    except Exception as e:
        log(f"❌ Ollama 客戶端初始化失敗: {e}")
        log("💡 提示：請確認 Ollama 服務地址正確（預設: http://localhost:11434/v1）")
        return None, None


def get_ai_translation(tag, client, model_name, log_callback=None):
    """使用 Ollama OpenAI 相容介面翻譯未知的 AI 繪圖標籤"""
    clean_tag = tag.replace("\\(", "(").replace("\\)", ")")
    prompt = build_translate_prompt(tag=clean_tag)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.3
        )
        if response and response.choices and len(response.choices) > 0:
            translation = response.choices[0].message.content.strip()
            refusal_keywords = (
                "cannot fulfill", "i cannot", "i can't", "my purpose is",
                "refuse", "cannot translate", " inappropriate", "explicit",
                "我無法提供", "無法提供此類內容", "我的目的是提供安全", "合乎道德",
                "安全且合乎道德", "cannot provide", "不提供"
            )
            trans_lower = translation.lower()
            if (len(translation) > 80 or
                    any(kw in trans_lower or kw in translation for kw in refusal_keywords)):
                return "（未翻譯）"
            return translation
        else:
            return "（未翻譯）"
    except Exception as e:
        error_msg = f"⚠️ 翻譯失敗 {tag}: {e}"
        if log_callback:
            log_callback(error_msg)
        return "（未翻譯）"


# --- Gemini 雲端 API ---

def init_gemini(api_key, log_callback=None):
    """測試 Gemini API 連線，回傳 (client, model_name) 或 (None, None)"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        for model_name in ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']:
            try:
                response = client.models.generate_content(model=model_name, contents="hi")
                if response and response.text:
                    log(f"✅ Gemini 使用模型: {model_name}")
                    return client, model_name
            except Exception as e:
                err_str = str(e)
                if "404" in err_str or "not found" in err_str.lower():
                    log(f"⚠️ 模型 {model_name} 不存在，嘗試下一個...")
                else:
                    log(f"⚠️ 模型 {model_name} 測試失敗: {err_str[:80]}")
                continue
        log("❌ 無法找到可用的 Gemini 模型，請檢查 API 或改用 Ollama")
        return None, None
    except ImportError:
        log("❌ 請安裝 google-genai: pip install google-genai")
        return None, None
    except Exception as e:
        log(f"❌ Gemini 初始化失敗: {e}")
        return None, None


def get_gemini_translation(tag, client, model_name, log_callback=None):
    """使用 Google Gemini 雲端 API 翻譯標籤"""
    try:
        time.sleep(2)
        clean_tag = tag.replace("\\(", "(").replace("\\)", ")")
        prompt = build_translate_prompt(tag=clean_tag)
        response = client.models.generate_content(model=model_name, contents=prompt)
        translation = (response.text or "").strip()
        refusal_keywords = (
            "cannot fulfill", "i cannot", "我無法提供", "無法提供此類內容",
            "我的目的是提供安全", "合乎道德", "cannot provide", "不提供"
        )
        if len(translation) > 80 or any(kw in translation.lower() or kw in translation for kw in refusal_keywords):
            return "（未翻譯）"
        return translation
    except Exception as e:
        err_str = str(e)
        if log_callback:
            log_callback(f"⚠️ Gemini 翻譯失敗 {tag}: {err_str[:100]}")
        if "404" in err_str or "not found" in err_str.lower():
            raise RuntimeError(f"Gemini 模型不可用: {err_str}") from e
        return "（未翻譯）"


# --- 批次翻譯 ---

def batch_translate_ollama(client, model_name, tags_list, log_callback=None):
    """Ollama 批次翻譯"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    if not tags_list:
        return {}
    try:
        tags_text = "\n".join(t.strip().replace("\\(", "(").replace("\\)", ")") for t in tags_list)
        prompt = build_translate_prompt(tags_text=tags_text)
        log(f"📦 批次翻譯 {len(tags_list)} 個標籤...")
        time.sleep(2)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.3
        )
        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        translated_list = [line.strip() for line in text.split('\n') if line.strip()]
        result = {}
        for i, tag in enumerate(tags_list):
            result[tag] = translated_list[i] if i < len(translated_list) else "（未翻譯）"
        return result
    except Exception as e:
        log(f"❌ 批次翻譯失敗: {e}")
        return {tag: "（未翻譯）" for tag in tags_list}


def batch_translate_gemini(tags_list, api_key, model_name='gemini-1.5-flash', log_callback=None):
    """Gemini 批次翻譯"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    if not tags_list:
        return {}
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        tags_text = "\n".join(t.strip().replace("\\(", "(").replace("\\)", ")") for t in tags_list)
        prompt = build_translate_prompt(tags_text=tags_text)
        log(f"📦 批次翻譯 {len(tags_list)} 個標籤...")
        time.sleep(2)
        response = client.models.generate_content(model=model_name, contents=prompt)
        translated_list = [line.strip() for line in (response.text or "").strip().split('\n') if line.strip()]
        result = {}
        for i, tag in enumerate(tags_list):
            result[tag] = translated_list[i] if i < len(translated_list) else "（未翻譯）"
        return result
    except Exception as e:
        log(f"❌ 批次翻譯失敗: {e}")
        return {tag: "（未翻譯）" for tag in tags_list}


# --- 批次翻譯＋分類 ---

def batch_translate_and_classify_gemini(tags_list, api_key, model_name, log_callback=None, pre_sorted_hint=False):
    """一次讓 AI 翻譯並分類，回傳 {tag: {zh_tag, category}}"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    if not tags_list:
        return {}
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        tags_text = "\n".join(t.strip().replace("\\(", "(").replace("\\)", ")") for t in tags_list)
        prompt = build_translate_and_classify_prompt(tags_text, pre_sorted_hint=pre_sorted_hint)
        log(f"🧠 批次翻譯＋分類 {len(tags_list)} 個標籤（智慧更新其他類）...")
        time.sleep(2)
        response = client.models.generate_content(model=model_name, contents=prompt)
        parsed = _parse_translate_classify_response(response.text or "", tags_list)
        for t in tags_list:
            if t not in parsed:
                parsed[t] = {"zh_tag": "（未翻譯）", "category": "其他 (General)"}
        return parsed
    except Exception as e:
        log(f"❌ 批次翻譯＋分類失敗: {e}")
        return {t: {"zh_tag": "（未翻譯）", "category": "其他 (General)"} for t in tags_list}


def batch_translate_and_classify_ollama(client, model_name, tags_list, log_callback=None, pre_sorted_hint=False):
    """Ollama 版：一次翻譯並分類"""
    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    if not tags_list:
        return {}
    try:
        tags_text = "\n".join(t.strip().replace("\\(", "(").replace("\\)", ")") for t in tags_list)
        prompt = build_translate_and_classify_prompt(tags_text, pre_sorted_hint=pre_sorted_hint)
        log(f"🧠 批次翻譯＋分類 {len(tags_list)} 個標籤（智慧更新其他類）...")
        time.sleep(2)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
            temperature=0.3
        )
        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        parsed = _parse_translate_classify_response(text, tags_list)
        for t in tags_list:
            if t not in parsed:
                parsed[t] = {"zh_tag": "（未翻譯）", "category": "其他 (General)"}
        return parsed
    except Exception as e:
        log(f"❌ 批次翻譯＋分類失敗: {e}")
        return {t: {"zh_tag": "（未翻譯）", "category": "其他 (General)"} for t in tags_list}
