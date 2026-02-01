"""WD14 標籤分頁"""
import gradio as gr

from . import shared
from src.tagger_wd14 import tag_folder as wd14_tag_folder


def _run_wd14_stream(wd14_source_dir, train_data_dir, trigger, sort_by):
    src = (wd14_source_dir or train_data_dir or "").strip()
    if not src:
        yield "❌ 請選擇 WD14 輸入資料夾或圖片資料夾"
        return

    def producer(log_cb):
        n = wd14_tag_folder(src, trigger_word=trigger or "Niyaniya",
                            sort_by_category=sort_by, log_callback=log_cb)
        log_cb(f"\n\n✅ WD14 標籤完成，共處理 {n} 張圖片")

    for text in shared.stream_from_log_callback(producer):
        yield text


def render(log_output: gr.Textbox, defaults: dict, train_data_dir: gr.Textbox):
    """
    建立 WD14 標籤分頁。
    train_data_dir 來自 LoRA 分頁，作為輸入資料夾的備用。
    回傳供 save_config 使用的元件 dict。
    """
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                wd14_source_dir = gr.Textbox(label="輸入資料夾", value=defaults["wd14_source_dir"],
                                             placeholder="包含圖片的資料夾路徑", scale=9)
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_folder(x),
                    inputs=[wd14_source_dir], outputs=[wd14_source_dir])
            wd14_trigger_word = gr.Textbox(label="前置詞", value=defaults["wd14_trigger_word"],
                                           placeholder="Niyaniya")
            sort_tags_by_category = gr.Checkbox(label="自動按類別排序標籤",
                                                value=defaults["sort_tags_by_category"])
            wd14_btn = gr.Button("執行 WD14 標籤", variant="primary")

    wd14_btn.click(
        fn=_run_wd14_stream,
        inputs=[wd14_source_dir, train_data_dir, wd14_trigger_word, sort_tags_by_category],
        outputs=log_output,
    )

    return {
        "wd14_source_dir": wd14_source_dir,
        "wd14_trigger_word": wd14_trigger_word,
        "sort_tags_by_category": sort_tags_by_category,
    }
