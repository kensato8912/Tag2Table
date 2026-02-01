"""標籤翻譯管理分頁"""
import gradio as gr

from . import shared
from src.database_manager import DB_FILE, generate_folder_report, generate_file_report
from src.processor import process_with_ai


def _run_translate_stream(folder_path, output_file, enable_classification, translation_mode,
                          gemini_api_key, ollama_host_mode, ollama_remote_ip, model_name):
    if not folder_path:
        yield "❌ 請選擇訓練資料夾"
        return
    if not output_file:
        yield "❌ 請輸入輸出檔名"
        return
    if translation_mode == "gemini" and not gemini_api_key:
        yield "❌ 請輸入 Gemini API Key"
        return
    if translation_mode == "ollama":
        url = shared.get_ollama_url(ollama_host_mode, ollama_remote_ip)
        if not url:
            yield "❌ 請填入遠端 IP 或確認 Ollama 已啟動"
            return
        if not model_name:
            yield "❌ 請選擇模型"
            return

    def producer(log_cb):
        process_with_ai(
            folder_path=folder_path,
            ollama_url=shared.get_ollama_url(ollama_host_mode, ollama_remote_ip),
            model_name=model_name,
            output_file=output_file,
            enable_classification=enable_classification,
            translation_mode=translation_mode,
            gemini_api_key=gemini_api_key,
            log_callback=log_cb,
            progress_callback=lambda v, m: None,
        )

    for text in shared.stream_from_log_callback(producer):
        yield text


def _run_folder_report_stream(folder_path):
    if not folder_path:
        yield "❌ 請先選擇訓練資料夾"
        return

    def producer(log_cb):
        ok = generate_folder_report(folder_path, report_file=None, db_path=DB_FILE,
                                    log_callback=log_cb, open_after=False)
        log_cb("\n\n✅ 報告已生成" if ok else "\n\n❌ 產生失敗")

    for text in shared.stream_from_log_callback(producer):
        yield text


def _run_file_report_stream(folder_path):
    if not folder_path:
        yield "❌ 請先選擇訓練資料夾"
        return

    def producer(log_cb):
        ok = generate_file_report(folder_path, report_file=None, db_path=DB_FILE,
                                  log_callback=log_cb, open_after=False)
        log_cb("\n\n✅ 報告已生成" if ok else "\n\n❌ 產生失敗")

    for text in shared.stream_from_log_callback(producer):
        yield text


def render(log_output: gr.Textbox, defaults: dict):
    """建立標籤翻譯管理分頁，回傳供 save_config 使用的元件 dict"""
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                folder_path = gr.Textbox(label="訓練資料夾", value=defaults["folder_path"],
                                         placeholder="選擇包含 .txt 的資料夾路徑", scale=9)
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_folder(x),
                    inputs=[folder_path], outputs=[folder_path])
            with gr.Row():
                output_file = gr.Textbox(label="存檔路徑", value=defaults["output_file"],
                                         placeholder="輸出 .txt 路徑", scale=9)
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_file_save(x),
                    inputs=[output_file], outputs=[output_file])
            enable_classification = gr.Checkbox(label="啟用分類輸出", value=defaults["enable_classification"])
            translation_mode = gr.Radio(["ollama", "gemini"], value=defaults["translation_mode"],
                                        label="翻譯引擎")

            with gr.Accordion("Gemini 設定", open=False):
                gemini_api_key = gr.Textbox(label="API Key", value=defaults["gemini_api_key"],
                                            type="password", placeholder="輸入 Gemini API Key")

            with gr.Accordion("Ollama 設定", open=True):
                ollama_host_mode = gr.Radio(["local", "remote"], value=defaults["ollama_host_mode"],
                                            label="連線目標")
                ollama_remote_ip = gr.Textbox(label="遠端 IP", value=defaults["ollama_remote_ip"],
                                              placeholder="同 Wi‑Fi 下另一台電腦的 IP")
                model_name = gr.Textbox(label="模型", value=defaults["model_name"],
                                        placeholder="gemma2:2b")

            with gr.Row():
                process_btn = gr.Button("開始處理", variant="primary")
                report_folder_btn = gr.Button("產生資料夾報告")
                report_file_btn = gr.Button("逐檔報告")
                save_config_btn = gr.Button("儲存設定")

    process_btn.click(
        fn=_run_translate_stream,
        inputs=[folder_path, output_file, enable_classification, translation_mode,
                gemini_api_key, ollama_host_mode, ollama_remote_ip, model_name],
        outputs=log_output,
    )
    report_folder_btn.click(
        fn=_run_folder_report_stream,
        inputs=[folder_path],
        outputs=log_output,
    )
    report_file_btn.click(
        fn=_run_file_report_stream,
        inputs=[folder_path],
        outputs=log_output,
    )

    return {
        "folder_path": folder_path,
        "output_file": output_file,
        "enable_classification": enable_classification,
        "translation_mode": translation_mode,
        "gemini_api_key": gemini_api_key,
        "ollama_host_mode": ollama_host_mode,
        "ollama_remote_ip": ollama_remote_ip,
        "model_name": model_name,
        "save_config_btn": save_config_btn,
    }
