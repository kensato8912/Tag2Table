"""素材自動採集分頁：整合 gallery-dl"""
import re
import subprocess
import threading
import shutil
from pathlib import Path
from urllib.parse import quote_plus

import gradio as gr

from . import shared

_PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
DEFAULT_CRAWL_DIR = DATA_DIR / "temp_raw"
DANBOORU_BASE = "https://danbooru.donmai.us/posts?tags="

# 數量選項：(顯示文字, 實際 range 值)
RANGE_OPTIONS = [
    ("50 張", "1-50"),
    ("50-100 張", "1-100"),
    ("100-200 張", "1-200"),
    ("200-300 張", "1-300"),
    ("500 張", "1-500"),
]
RANGE_MAP = {label: val for label, val in RANGE_OPTIONS}

# 允許下載的圖片格式
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
# 排除的格式（影片、動圖等）
EXCLUDED_EXTENSIONS = ("mp4", "webm", "gif", "avi", "mov", "mkv")
_SKIP_PATTERN = re.compile(
    r"[^/\\]+\.(" + "|".join(re.escape(e) for e in EXCLUDED_EXTENSIONS) + r")(?:\?|$|\s|\))",
    re.IGNORECASE,
)

# 供停止按鈕使用
_crawl_proc = None
_crawl_stopped = False


def stop_crawl(current_log: str = "") -> str:
    """停止當前抓取任務，producer 會偵測並寫入日誌"""
    global _crawl_proc, _crawl_stopped
    _crawl_stopped = True
    if _crawl_proc and _crawl_proc.poll() is None:
        try:
            _crawl_proc.terminate()
        except Exception:
            pass
    return current_log or ""


def check_gallery_dl() -> tuple[bool, str]:
    """檢查 gallery-dl 是否已安裝，回傳 (是否可用, 提示訊息)"""
    exe = shutil.which("gallery-dl")
    if exe:
        return True, f"✅ gallery-dl 已安裝: {exe}"
    try:
        import gallery_dl  # noqa: F401
        return True, "✅ gallery-dl 已安裝 (python -m gallery_dl)"
    except ImportError:
        pass
    return False, (
        "⚠️ 未偵測到 gallery-dl，請先安裝： pip install gallery-dl\n"
        "https://github.com/mikf/gallery-dl#installation"
    )


def _run_crawler_stream(tags, range_val, output_dir, sleep_sec, chain_wd14, wd14_trigger, sort_by,
                        helper_ref_dest, helper_triggers):
    """執行 gallery-dl 下載，可選連鎖 WD14 + 素材篩選"""
    global _crawl_proc, _crawl_stopped
    if not tags or not tags.strip():
        yield "❌ 請輸入關鍵字 (Tags)"
        return

    ok, msg = check_gallery_dl()
    if not ok:
        yield msg
        return

    output_path = Path(output_dir or str(DEFAULT_CRAWL_DIR)).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    tags_encoded = quote_plus(tags.strip())
    url = DANBOORU_BASE + tags_encoded

    try:
        sleep_val = max(0, float(sleep_sec)) if sleep_sec is not None else 1.0
    except (TypeError, ValueError):
        sleep_val = 1.0

    # 優先使用 PATH 中的 gallery-dl，否則用 python -m gallery_dl
    if shutil.which("gallery-dl"):
        cmd_base = ["gallery-dl"]
    else:
        cmd_base = ["python", "-m", "gallery_dl"]

    # 僅下載 jpg/jpeg/png/webp，排除 mp4/webm/gif 等
    ext_filter = f"extension.lower() in {ALLOWED_EXTENSIONS}"
    size_filter = "image_width > 512 and image_height > 512"
    combined_filter = f"{ext_filter} and {size_filter}"

    cmd = cmd_base + [
        "--range", range_val or "1-100",
        "--directory", str(output_path),
        "--sleep", str(sleep_val),
        "--filter", combined_filter,
        "--verbose",
        url,
    ]

    def producer(log_cb):
        global _crawl_proc, _crawl_stopped
        _crawl_stopped = False
        _crawl_proc = None

        log_cb(f"🚀 執行: {' '.join(cmd)}")
        log_cb("📷 僅下載 jpg/jpeg/png/webp，排除 mp4/webm/gif 等影片與動圖")
        log_cb("")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        _crawl_proc = proc

        def _read():
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    continue
                line = line.rstrip()
                m = _SKIP_PATTERN.search(line)
                if m and ("skip" in line.lower() or "filter" in line.lower() or "filtered" in line.lower()):
                    log_cb(f"已跳過非圖片資源: {m.group(0)}")
                else:
                    log_cb(line)
            proc.stdout.close()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        proc.wait()
        _crawl_proc = None

        if _crawl_stopped:
            log_cb("\n⏹ 已手動停止抓取")
            return
        if proc.returncode != 0:
            log_cb(f"\n⚠️ gallery-dl 結束碼: {proc.returncode}")
            return

        log_cb("\n✅ 下載完成")

        if not chain_wd14:
            return

        # 連鎖：WD14 標註
        log_cb("\n" + "=" * 50)
        log_cb("🔄 自動執行 WD14 標註...")
        from src.tagger_wd14 import tag_folder as wd14_tag_folder
        try:
            n = wd14_tag_folder(
                str(output_path),
                trigger_word=wd14_trigger or "Niyaniya",
                sort_by_category=sort_by,
                log_callback=log_cb,
            )
            log_cb(f"\n✅ WD14 標註完成，處理 {n} 張圖片")
        except Exception as e:
            log_cb(f"\n❌ WD14 標註失敗: {e}")
            return

        # 連鎖：素材篩選 (helper_grabber)
        log_cb("\n" + "=" * 50)
        log_cb("🔄 自動執行素材篩選...")
        from src.helper_grabber import grab_hand_feet_refs
        try:
            dest = helper_ref_dest or str(output_path.parent / "crawler_filtered")
            triggers = [t.strip() for t in (helper_triggers or "").split(",") if t.strip()] or ["Niyaniya", "Ibuki"]
            n = grab_hand_feet_refs(
                str(output_path),
                dest,
                trigger_words=triggers,
                recursive=True,
                log_callback=log_cb,
            )
            log_cb(f"\n✅ 素材篩選完成，複製 {n} 張圖片至 {dest}")
        except Exception as e:
            log_cb(f"\n❌ 素材篩選失敗: {e}")

    for text in shared.stream_from_log_callback(producer):
        yield text


def render(defaults: dict, wd14_comps: dict, train_comps: dict):
    """
    建立素材自動抓取分頁。
    需從 WD14、Train 分頁取得：sort_tags_by_category, wd14_trigger_word, helper_ref_dest, helper_trigger_words
    """
    ok, env_msg = check_gallery_dl()
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(f"**環境檢查**: {env_msg}")
            tags_input = gr.Textbox(
                label="關鍵字 (Tags)",
                value=defaults.get("crawler_tags", "hand_focus rating:g score:>20"),
                placeholder="Danbooru 標籤，如 hand_focus rating:g score:>20",
                lines=2,
            )
            range_dropdown = gr.Dropdown(
                label="數量",
                choices=[label for label, _ in RANGE_OPTIONS],
                value="50-100 張",
            )
            sleep_input = gr.Slider(
                label="請求延遲 (秒) — 避免請求太頻繁被圖庫站封鎖",
                minimum=0,
                maximum=15,
                step=0.5,
                value=defaults.get("crawler_sleep", 1.0),
            )
            with gr.Row():
                output_dir = gr.Textbox(
                    label="輸出路徑",
                    value=defaults.get("crawler_output_dir", str(DEFAULT_CRAWL_DIR)),
                    placeholder="下載至...",
                    scale=9,
                )
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_folder(x),
                    inputs=[output_dir],
                    outputs=[output_dir],
                )
            chain_check = gr.Checkbox(
                label="下載完後自動執行 WD14 標註與素材篩選",
                value=defaults.get("crawler_chain_wd14", False),
            )
            with gr.Row():
                crawl_btn = gr.Button("開始下載", variant="primary")
                stop_btn = gr.Button("停止抓取")

    # 依賴其他分頁的元件（用於連鎖時取得參數）
    wd14_trigger = wd14_comps.get("wd14_trigger_word")
    sort_by = wd14_comps.get("sort_tags_by_category")
    helper_ref_dest = train_comps.get("helper_ref_dest")
    helper_triggers = train_comps.get("helper_trigger_words")

    def _run(tags, range_label, out_dir, sleep_sec, chain, trig, sb, ref_dest, triggers):
        range_val = RANGE_MAP.get(range_label, "1-100")
        for text in _run_crawler_stream(
            tags, range_val, out_dir, sleep_sec, chain,
            trig, sb, ref_dest, triggers
        ):
            yield text

    bindings = [
        (crawl_btn, _run, [
            tags_input, range_dropdown, output_dir, sleep_input, chain_check,
            wd14_trigger, sort_by, helper_ref_dest, helper_triggers,
        ]),
    ]

    comps = {
        "stop_crawl_btn": stop_btn,
        "stop_crawl_fn": stop_crawl,
        "crawler_tags": tags_input,
        "crawler_output_dir": output_dir,
        "crawler_sleep": sleep_input,
        "crawler_chain_wd14": chain_check,
    }
    return comps, bindings
