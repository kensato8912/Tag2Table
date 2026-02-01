"""LoRA 一鍵訓練分頁"""
import re
import gradio as gr

from . import shared
from src.trainer import start_ken_lora_train
from src.helper_grabber import grab_hand_feet_refs
from pathlib import Path


def _train_log_stream(model_path, data_dir, output_name):
    def producer(log_cb):
        proc = start_ken_lora_train(
            model_path, data_dir, output_name or "Ken_Ansha_LoRA",
            log_callback=lambda m, replace_last=False, **kwargs: log_cb(m, replace_last)
        )
        if proc:
            proc.wait()

    for text in shared.stream_from_log_callback(producer):
        yield text


def _run_helper_grabber_stream(source, dest, triggers_raw):
    if not source:
        yield "❌ 請先選擇圖片資料夾（來源）"
        return
    dest = dest or str(Path(source).parent / "hands_feet_ref")
    triggers = [t.strip() for t in (triggers_raw or "").split(",") if t.strip()] or ["Niyaniya", "Ibuki"]

    def producer(log_cb):
        n = grab_hand_feet_refs(source, dest, trigger_words=triggers, recursive=True, log_callback=log_cb)
        log_cb(f"\n\n✅ 自訂篩選完成，共複製 {n} 張圖片")

    for text in shared.stream_from_log_callback(producer):
        yield text


def _run_custom_classify_stream(source, dest, triggers_raw, kw1, fd1, kw2, fd2, kw3, fd3, kw4, fd4):
    if not source:
        yield "❌ 請先選擇圖片資料夾（來源）"
        return
    dest = dest or str(Path(source).parent / "custom_ref")
    triggers = [t.strip() for t in (triggers_raw or "").split(",") if t.strip()] or ["Niyaniya", "Ibuki"]
    rules = {}
    for i, (kw, fd) in enumerate([(kw1, fd1), (kw2, fd2), (kw3, fd3), (kw4, fd4)]):
        if not (kw and fd):
            continue
        kws = [k.strip() for k in kw.split(",") if k.strip()]
        if not kws:
            continue
        priority = 10
        if fd and fd[0].isdigit():
            m = re.match(r"^(\d+)_", fd)
            if m:
                priority = int(m.group(1))
        rules[f"rule_{i}"] = {"folder_name": fd, "keywords": kws, "priority": priority}
    if not rules:
        yield "❌ 請至少填寫一列：關鍵字與目標資料夾"
        return

    def prod(log_cb):
        n = grab_hand_feet_refs(source, dest, trigger_words=triggers, recursive=True,
                                rules=rules, log_callback=log_cb)
        log_cb(f"\n\n✅ 自訂分類完成，共複製 {n} 張圖片")

    for text in shared.stream_from_log_callback(prod):
        yield text


def render(defaults: dict):
    """建立 LoRA 訓練分頁。回傳 (comps_dict, bindings)。"""
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                base_model_path = gr.Textbox(label="底模", value=defaults["base_model_path"],
                                             placeholder=".safetensors 或 .ckpt 路徑", scale=9)
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_file_open(x, "選擇底模"),
                    inputs=[base_model_path], outputs=[base_model_path])
            with gr.Row():
                train_data_dir = gr.Textbox(label="圖片資料夾", value=defaults["train_data_dir"],
                                            placeholder="訓練用圖片與 .txt 所在資料夾", scale=9)
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_folder(x),
                    inputs=[train_data_dir], outputs=[train_data_dir])
            lora_output_name = gr.Textbox(label="輸出名稱", value=defaults["lora_output_name"],
                                          placeholder="Ken_Ansha_LoRA")

            gr.Markdown("### 自訂篩選")
            with gr.Row():
                helper_ref_dest = gr.Textbox(label="參考資料夾", value=defaults["helper_ref_dest"],
                                             placeholder="輸出位置", scale=9)
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_folder(x),
                    inputs=[helper_ref_dest], outputs=[helper_ref_dest])
            helper_trigger_words = gr.Textbox(label="觸發詞排除", value=defaults["helper_trigger_words"],
                                              placeholder="Niyaniya, Ibuki")

            gr.Markdown("### 自訂分類監控")
            with gr.Row():
                kw1, fd1 = gr.Textbox(label="關鍵字1", placeholder="weapon, sword"), gr.Textbox(label="目標資料夾1", placeholder="10_weapon_ref")
            with gr.Row():
                kw2, fd2 = gr.Textbox(label="關鍵字2"), gr.Textbox(label="目標資料夾2")
            with gr.Row():
                kw3, fd3 = gr.Textbox(label="關鍵字3"), gr.Textbox(label="目標資料夾3")
            with gr.Row():
                kw4, fd4 = gr.Textbox(label="關鍵字4"), gr.Textbox(label="目標資料夾4")

            with gr.Row():
                train_btn = gr.Button("一鍵訓練", variant="primary")
                helper_btn = gr.Button("篩選手腳參考圖")
                custom_btn = gr.Button("執行分類監控")

    def _train_stream(model_path, data_dir, output_name):
        if not model_path:
            yield "❌ 請選擇底模"
            return
        if not data_dir:
            yield "❌ 請選擇圖片資料夾"
            return
        for text in _train_log_stream(model_path, data_dir, output_name or "Ken_Ansha_LoRA"):
            yield text

    bindings = [
        (train_btn, _train_stream, [base_model_path, train_data_dir, lora_output_name]),
        (helper_btn, _run_helper_grabber_stream, [train_data_dir, helper_ref_dest, helper_trigger_words]),
        (custom_btn, _run_custom_classify_stream,
         [train_data_dir, helper_ref_dest, helper_trigger_words, kw1, fd1, kw2, fd2, kw3, fd3, kw4, fd4]),
    ]

    comps = {
        "base_model_path": base_model_path,
        "train_data_dir": train_data_dir,
        "lora_output_name": lora_output_name,
        "helper_ref_dest": helper_ref_dest,
        "helper_trigger_words": helper_trigger_words,
    }
    return comps, bindings
