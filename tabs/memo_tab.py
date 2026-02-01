"""Memo 轉換分頁：1. 正常拆解括號  2. 純英文轉 帶括號+中文"""
import os
import re
from pathlib import Path

import gradio as gr

from . import shared
from src.database_manager import (
    load_tag_database,
    load_prompt_presets,
    save_prompt_presets,
    DB_FILE,
    tag_map,
)


def _convert_normal(text: str) -> tuple[str, str]:
    """正常模式：拆解括號，輸出 1=移除括號內文字，2=只顯示括號內文字"""
    if not text or not text.strip():
        return "", ""
    text = text.strip()
    ptrn = r'(?<!\\)\(([^)]*)(?<!\\)\)'
    out1 = re.sub(ptrn, "", text)
    out1 = re.sub(r',\s*,', ',', out1).strip(' ,')
    matches = re.findall(ptrn, text)
    out2 = ", ".join(matches) if matches else ""
    return out1, out2


def _convert_en2zh(text: str) -> tuple[str, str]:
    """純英文轉帶括號+中文：依 tag_map 翻譯，輸出 1=英文(中文)，2=僅中文"""
    if not text or not text.strip():
        return "", ""
    text = text.strip()
    text_clean = re.sub(r'(?<!\\)\([^)]*(?<!\\)\)', "", text)
    tags = [t.strip() for t in text_clean.replace("\n", ",").split(",") if t.strip()]
    db = load_tag_database(DB_FILE)
    for k, v in tag_map.items():
        db.setdefault(k, {})["zh_tag"] = v
    result_with_bracket = []
    result_zh_only = []
    for tag in tags:
        zh = db.get(tag, {}).get("zh_tag", "未翻譯")
        result_with_bracket.append(f"{tag}({zh})")
        result_zh_only.append(zh)
    out1 = ", ".join(result_with_bracket)
    out2 = ", ".join(result_zh_only)
    return out1, out2


def _convert(mode: str, text: str) -> tuple[str, str]:
    if mode == "en2zh":
        return _convert_en2zh(text)
    return _convert_normal(text)


def _save_as_preset(text: str, preset_name: str) -> str:
    if not text or not text.strip():
        return "❌ Memo 輸入為空，請先輸入或載入標籤"
    text_clean = re.sub(r'(?<!\\)\([^)]*(?<!\\)\)', "", text)
    tags = [t.strip() for t in text_clean.replace("\n", ",").split(",") if t.strip()]
    if not tags:
        return "❌ 無法解析標籤，請確認格式"
    name = (preset_name or "").strip() or "自訂套裝"
    presets = load_prompt_presets()
    presets[name] = tags
    save_prompt_presets(presets)
    return f"✅ 已儲存為「{name}」，題詞組合器下次開啟時可選用"


def _load_image_for_preview(path: str):
    """載入圖片供預覽，回傳 RGB numpy array 或 None"""
    if not path or not os.path.isfile(path):
        return None
    try:
        import cv2
        img = cv2.imread(path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _find_same_name_image(txt_path: str) -> tuple[str | None, str]:
    """依 TXT 檔名找同名 JPG/PNG，回傳 (圖片路徑, 檔名)"""
    if not txt_path or not os.path.isfile(txt_path):
        return None, ""
    base = Path(txt_path).with_suffix("")
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = base.with_suffix(ext)
        if p.exists():
            return str(p), p.name
    return None, base.name


def _read_txt(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="cp950") as f:
                return f.read().strip()
        except Exception:
            return ""
    except Exception:
        return ""


def render(defaults: dict = None):
    """建立 Memo 轉換分頁。回傳 (comps_dict, bindings)。"""
    gr.Markdown("### Memo 轉換\n1. **正常拆解括號**：輸出移除括號內文字 / 只顯示括號內文字  \n2. **純英文轉帶括號+中文**：依 tag_map 翻譯成 英文(中文)")
    with gr.Row():
        with gr.Column(scale=1):
            memo_input = gr.Textbox(
                label="Memo 輸入",
                placeholder="long hair(長髮), blush(臉紅), bangs(劉海), blonde hair(金髮)",
                lines=8,
            )
            with gr.Row():
                mode_radio = gr.Radio(
                    choices=[
                        ("1. 正常拆解括號", "normal"),
                        ("2. 純英文轉 帶括號+中文", "en2zh"),
                    ],
                    value="normal",
                    label="模式",
                )
            with gr.Row():
                btn_convert = gr.Button("轉換", variant="primary")
                btn_clear = gr.Button("清除")
        with gr.Column(scale=1):
            memo_out1 = gr.Textbox(
                label="輸出1：移除 ( ) 內文字",
                lines=6,
                interactive=False,
            )
            memo_out2 = gr.Textbox(
                label="輸出2：只顯示 ( ) 當中文字",
                lines=6,
                interactive=False,
            )
    with gr.Row():
        preset_name = gr.Textbox(label="套裝名稱（存成角色快速套裝）", placeholder="自訂套裝", scale=3)
        btn_save_preset = gr.Button("存成角色快速套裝", variant="secondary")
    preset_status = gr.Textbox(label="狀態", interactive=False, visible=True)

    gr.Markdown("---\n### 檔案瀏覽（*.txt）")
    with gr.Row():
        memo_folder = gr.Textbox(label="資料夾", placeholder="選擇包含 TXT 的資料夾")
        btn_browse = gr.Button("瀏覽", size="sm")
        btn_load = gr.Button("載入 TXT 清單", variant="primary", size="sm")
    with gr.Row():
        btn_prev = gr.Button("⬅ 上一張")
        memo_counter = gr.Textbox(label="進度", value="0 / 0", interactive=False, scale=0)
        btn_next = gr.Button("下一張 ➡")
    with gr.Row():
        with gr.Column(scale=1):
            memo_preview_img = gr.Image(label="同名圖片預覽 (JPG/PNG)", type="numpy", height=240)
            memo_filename = gr.Textbox(label="檔名", value="", interactive=False)
        with gr.Column(scale=1):
            gr.Markdown("載入 TXT 時會自動填入上方 Memo 輸入並執行「純英文轉帶括號+中文」翻譯")

    txt_list_state = gr.State(value=[])
    txt_index_state = gr.State(value=0)

    def do_convert(mode, text):
        o1, o2 = _convert(mode, text)
        return o1, o2

    def do_clear():
        return "", "", ""

    def _load_folder(folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            return [], 0, "", "", "", None, "", "請選擇有效資料夾", "0 / 0"
        txt_paths = sorted(Path(folder_path).rglob("*.txt"), key=lambda p: p.name)
        txt_paths = [str(p) for p in txt_paths]
        if not txt_paths:
            return [], 0, "", "", "", None, "", "資料夾內無 *.txt 檔", "0 / 0"
        return _load_txt_at(txt_paths, 0)

    def _load_txt_at(txt_paths, idx):
        if not txt_paths:
            return [], 0, "", "", "", None, "", "", "0 / 0"
        idx = max(0, min(idx, len(txt_paths) - 1))
        path = txt_paths[idx]
        text = _read_txt(path)
        o1, o2 = _convert_en2zh(text)
        img_path, img_fname = _find_same_name_image(path)
        img = _load_image_for_preview(img_path) if img_path else None
        display_name = img_fname if img_path else (Path(path).name if path else "")
        return (
            txt_paths, idx,
            text, o1, o2,
            img, display_name,
            f"已載入 {len(txt_paths)} 個 TXT", f"{idx + 1} / {len(txt_paths)}",
        )

    def _go_prev(txt_list, idx):
        if not txt_list:
            return [], 0, "", "", "", None, "", "", "0 / 0"
        return _load_txt_at(txt_list, idx - 1)

    def _go_next(txt_list, idx):
        if not txt_list:
            return [], 0, "", "", "", None, "", "", "0 / 0"
        return _load_txt_at(txt_list, idx + 1)

    btn_browse.click(fn=lambda x: shared.browse_folder(x), inputs=[memo_folder], outputs=[memo_folder])
    btn_load.click(
        fn=_load_folder,
        inputs=[memo_folder],
        outputs=[
            txt_list_state, txt_index_state,
            memo_input, memo_out1, memo_out2,
            memo_preview_img, memo_filename,
            preset_status, memo_counter,
        ],
    )
    btn_prev.click(
        fn=_go_prev,
        inputs=[txt_list_state, txt_index_state],
        outputs=[
            txt_list_state, txt_index_state,
            memo_input, memo_out1, memo_out2,
            memo_preview_img, memo_filename,
            preset_status, memo_counter,
        ],
    )
    btn_next.click(
        fn=_go_next,
        inputs=[txt_list_state, txt_index_state],
        outputs=[
            txt_list_state, txt_index_state,
            memo_input, memo_out1, memo_out2,
            memo_preview_img, memo_filename,
            preset_status, memo_counter,
        ],
    )

    btn_convert.click(
        fn=do_convert,
        inputs=[mode_radio, memo_input],
        outputs=[memo_out1, memo_out2],
    )
    btn_clear.click(
        fn=do_clear,
        inputs=[],
        outputs=[memo_input, memo_out1, memo_out2],
    )
    btn_save_preset.click(
        fn=_save_as_preset,
        inputs=[memo_input, preset_name],
        outputs=[preset_status],
    )

    comps = {
        "memo_input": memo_input,
        "memo_out1": memo_out1,
        "memo_out2": memo_out2,
    }
    return comps, []
