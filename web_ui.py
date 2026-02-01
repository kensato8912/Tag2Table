"""
Tag2Table Gradio Web UI — 主入口
啟動後透過 http://localhost:7870 訪問

檔案佈局：
  web_ui.py    — 主入口，建立 Tabs 與共用日誌
  tabs/        — 各功能分頁：translate_tab, train_tab, wd14_tab, crawler_tab, cropper_tab
  tabs/shared.py — 共用：stream_from_log_callback, load_defaults, 檔案對話框
  src/         — 核心邏輯：processor, trainer, tagger_wd14, helper_grabber, cropper 等

全域設定：data/config.json 儲存 API Key、路徑、常用關鍵字（由 database_manager 讀寫）

依賴：gradio, mediapipe, gallery-dl, opencv-python（見 requirements.txt）
"""
import os
import sys
from pathlib import Path

# 鎖定當前執行路徑為腳本所在目錄，避免噴到 StabilityMatrix 根目錄
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_PATH)
if BASE_PATH not in sys.path:
    sys.path.insert(0, BASE_PATH)

_PROJECT_ROOT = Path(BASE_PATH)

import gradio as gr

from src.database_manager import save_config, _ensure_txt_dir


def _check_core_deps():
    """確認核心依賴已安裝，缺少時在 console 提示"""
    missing = []
    try:
        import gradio  # noqa: F401
    except ImportError:
        missing.append("gradio")
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        missing.append("mediapipe")
    try:
        import cv2  # noqa: F401
    except ImportError:
        missing.append("opencv-python")
    try:
        import gallery_dl  # noqa: F401
    except ImportError:
        missing.append("gallery-dl")
    if missing:
        print(f"[WARN] Missing deps: {', '.join(missing)}. Run: pip install -r requirements.txt")


from tabs.shared import load_defaults, get_ollama_url
from tabs.translate_tab import render as render_translate_tab
from tabs.train_tab import render as render_train_tab
from tabs.wd14_tab import render as render_wd14_tab
from tabs.crawler_tab import render as render_crawler_tab
from tabs.cropper_tab import render as render_cropper_tab, render_detection_demo_tab, render_crop_lib_demo_tab
from tabs.memo_tab import render as render_memo_tab
from tabs.lora_demo_tab import render as render_lora_demo_tab


def _save_config_cfg(gemini_api_key, folder_path, output_file, ollama_host_mode, ollama_remote_ip,
                     base_model_path, train_data_dir, lora_output_name, sort_tags_by_category,
                     helper_ref_dest, helper_trigger_words, wd14_source_dir, wd14_trigger_word,
                     crawler_tags, crawler_output_dir, crawler_sleep, crawler_chain_wd14):
    config = {
        "gemini_api_key": gemini_api_key or "",
        "folder_path": folder_path or "",
        "output_file": output_file or "",
        "ollama_url": get_ollama_url(ollama_host_mode, ollama_remote_ip or ""),
        "ollama_host_mode": ollama_host_mode or "local",
        "ollama_remote_ip": ollama_remote_ip or "",
        "base_model_path": base_model_path or "",
        "train_data_dir": train_data_dir or "",
        "lora_output_name": lora_output_name or "Ken_Ansha_LoRA",
        "sort_tags_by_category": sort_tags_by_category,
        "helper_ref_dest": helper_ref_dest or "",
        "helper_trigger_words": helper_trigger_words or "Niyaniya, Ibuki",
        "wd14_source_dir": wd14_source_dir or "",
        "wd14_trigger_word": wd14_trigger_word or "Niyaniya",
        "crawler_tags": crawler_tags or "",
        "crawler_output_dir": crawler_output_dir or "",
        "crawler_sleep": crawler_sleep if crawler_sleep is not None else 1.0,
        "crawler_chain_wd14": crawler_chain_wd14,
    }
    save_config(config)
    return "✅ 設定已儲存"


LOG_CSS = """
#log_display textarea {
    height: 300px !important;
    overflow-y: scroll !important;
    font-family: 'Courier New', monospace;
    background-color: #1a1a1a;
    color: #00ff00;
}
"""


def create_ui():
    _check_core_deps()
    defaults = load_defaults()
    _ensure_txt_dir()

    with gr.Blocks(title="Tag2Table", css=LOG_CSS) as demo:
        gr.Markdown("# Tag2Table — AI 標籤統計與翻譯工具")

        with gr.Tabs():
            with gr.Tab("標籤翻譯管理"):
                comps_translate, bind_translate = render_translate_tab(defaults)

            with gr.Tab("LoRA 一鍵訓練"):
                comps_train, bind_train = render_train_tab(defaults)

            with gr.Tab("WD14 標籤"):
                comps_wd14, bind_wd14 = render_wd14_tab(defaults, comps_train["train_data_dir"])

            with gr.Tab("素材自動抓取"):
                comps_crawler, bind_crawler = render_crawler_tab(defaults, comps_wd14, comps_train)

            with gr.Tab("智慧裁切"):
                comps_cropper, bind_cropper = render_cropper_tab(defaults, comps_train)

            with gr.Tab("偵測驗證 Demo"):
                comps_demo, bind_demo = render_detection_demo_tab()

            with gr.Tab("手動框選 + 存檔"):
                comps_crop_lib, bind_crop_lib = render_crop_lib_demo_tab()

            with gr.Tab("Memo 轉換"):
                comps_memo, bind_memo = render_memo_tab(defaults)

            with gr.Tab("LoRA 一鍵驗收"):
                comps_lora_demo, bind_lora_demo = render_lora_demo_tab(defaults)

        # 共用日誌（頁面最下方，所有分頁共用，固定高度防無限延伸）
        with gr.Column():
            log_output = gr.Textbox(
                label="執行日誌",
                lines=12,
                max_lines=15,
                interactive=False,
                placeholder="操作日誌將顯示於此...",
                elem_id="log_display",
            )

        # 綁定各分頁的按鈕
        for item in bind_translate + bind_train + bind_wd14 + bind_crawler + bind_cropper + bind_demo + bind_crop_lib + bind_memo + bind_lora_demo:
            if len(item) == 4:
                btn, fn, inputs, extra_out = item
                btn.click(fn=fn, inputs=inputs, outputs=[log_output, extra_out])
            else:
                btn, fn, inputs = item
                btn.click(fn=fn, inputs=inputs, outputs=log_output)

        # 停止抓取按鈕（需傳入當前 log 以 append）
        if comps_crawler.get("stop_crawl_btn"):
            comps_crawler["stop_crawl_btn"].click(
                fn=comps_crawler["stop_crawl_fn"],
                inputs=[log_output],
                outputs=log_output,
            )

        # 儲存設定按鈕
        save_config_btn = comps_translate.get("save_config_btn")
        if save_config_btn:
            save_config_btn.click(
                fn=_save_config_cfg,
                inputs=[
                    comps_translate["gemini_api_key"],
                    comps_translate["folder_path"],
                    comps_translate["output_file"],
                    comps_translate["ollama_host_mode"],
                    comps_translate["ollama_remote_ip"],
                    comps_train["base_model_path"],
                    comps_train["train_data_dir"],
                    comps_train["lora_output_name"],
                    comps_wd14["sort_tags_by_category"],
                    comps_train["helper_ref_dest"],
                    comps_train["helper_trigger_words"],
                    comps_wd14["wd14_source_dir"],
                    comps_wd14["wd14_trigger_word"],
                    comps_crawler["crawler_tags"],
                    comps_crawler["crawler_output_dir"],
                    comps_crawler["crawler_sleep"],
                    comps_crawler["crawler_chain_wd14"],
                ],
                outputs=log_output,
            )

    return demo


def _collect_allowed_paths():
    """從 config 收集使用者選擇的路徑，加入 Gradio 白名單（動態讀取）"""
    seen = set()
    paths = []

    def _add(p: Path, also_parent: bool = False):
        if not p or not str(p).strip():
            return
        dir_path = p if p.is_dir() else p.parent
        if not dir_path.exists():
            return
        norm = str(dir_path).replace("\\", "/")
        if norm not in seen:
            seen.add(norm)
            paths.append(norm)
        if also_parent and dir_path.parent != dir_path:
            parent_norm = str(dir_path.parent).replace("\\", "/")
            if parent_norm not in seen:
                seen.add(parent_norm)
                paths.append(parent_norm)

    paths.append(str(_PROJECT_ROOT).replace("\\", "/"))
    seen.add(str(_PROJECT_ROOT).replace("\\", "/"))

    try:
        defaults = load_defaults()
        for key in ["folder_path", "train_data_dir", "helper_ref_dest", "wd14_source_dir",
                    "crawler_output_dir", "cropper_source", "cropper_dest"]:
            v = defaults.get(key)
            if v and isinstance(v, str) and v.strip():
                _add(Path(v.strip()), also_parent=(key == "train_data_dir"))
        out_file = defaults.get("output_file")
        if out_file and isinstance(out_file, str) and out_file.strip():
            _add(Path(out_file.strip()).parent)
        base_model = defaults.get("base_model_path")
        if base_model and isinstance(base_model, str) and base_model.strip():
            _add(Path(base_model.strip()).parent)
    except Exception:
        pass
    return paths


def launch(share=False, server_port=7870):
    demo = create_ui()
    allowed_paths = _collect_allowed_paths()
    for port in range(server_port, server_port + 10):
        try:
            demo.launch(share=share, server_name="127.0.0.1", server_port=port, allowed_paths=allowed_paths)
            return
        except OSError as e:
            if "Cannot find empty port" in str(e) or "address already in use" in str(e).lower():
                print(f"Port {port} in use, trying {port + 1}...")
                continue
            raise
    raise OSError(f"All ports {server_port}-{server_port + 9} are in use.")


if __name__ == "__main__":
    base_port = 7870
    print(f"Tag2Table Gradio UI - Starting (port {base_port} or next available)...")
    launch(share=False, server_port=base_port)
