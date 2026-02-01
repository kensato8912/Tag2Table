"""LoRA 一鍵驗收 Demo：選擇 LoRA、組合 Prompt、呼叫 SD WebUI API 產圖"""
import base64
from pathlib import Path

import gradio as gr

from . import shared

# LoRA 資料夾路徑（StabilityMatrix）
DEFAULT_LORA_DIR = r"E:\Docker\StabilityMatrix\Data\Packages\reforge\models\Lora"
DEFAULT_SD_PORT = 7860  # WebUI 預設 7860，可改 7870 等
DEFAULT_PROMPT = "(masterpiece:1.2), best quality, 1girl, annsha, face closeup"


def _list_lora_files(lora_dir: str) -> list[str]:
    """列出資料夾內所有 .safetensors（含子資料夾）"""
    path = Path(lora_dir or DEFAULT_LORA_DIR)
    if not path.is_dir():
        return ["（請選擇有效路徑）"]
    files = sorted(path.rglob("*.safetensors"), key=lambda p: str(p).lower())
    return [str(f.relative_to(path)) if path != f.parent else f.name for f in files] if files else ["（無 .safetensors）"]


def _refresh_and_generate(
    lora_choice: str,
    prompt: str,
    weight: float,
    port: int,
) -> tuple[str | None, str]:
    """
    1. 呼叫 refresh-loras 更新列表
    2. 組合 prompt + <lora:檔名:權重>
    3. POST txt2img，回傳 (圖片 base64 解碼後的 numpy, 錯誤訊息)
    """
    import json
    import urllib.request
    import urllib.error

    try:
        import numpy as np
    except ImportError:
        return None, "需要 numpy"

    base_url = f"http://127.0.0.1:{int(port) if port else DEFAULT_SD_PORT}"

    # 1. refresh-loras
    try:
        req = urllib.request.Request(
            f"{base_url}/sdapi/v1/refresh-loras",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b"{}",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass
    except urllib.error.URLError as e:
        return None, f"❌ refresh-loras 失敗：{e}\n請確認 SD WebUI 已啟動且 port {port} 可連"
    except Exception as e:
        return None, f"❌ refresh-loras 錯誤：{e}"

    # 2. 組合 prompt
    lora_name = (lora_choice or "").strip()
    if not lora_name or lora_name.startswith("（"):
        return None, "❌ 請選擇有效的 LoRA 檔"
    # 只取檔名（不含子資料夾路徑），去掉 .safetensors
    base_name = Path(lora_name).stem
    lora_tag = f"<lora:{base_name}:{weight}>"
    full_prompt = f"{lora_tag}, {prompt.strip()}" if prompt.strip() else lora_tag

    # 3. txt2img
    payload = {
        "prompt": full_prompt,
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error",
        "steps": 20,
        "cfg_scale": 7,
        "width": 512,
        "height": 768,
        "seed": -1,
    }
    body = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(
            f"{base_url}/sdapi/v1/txt2img",
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        return None, f"❌ txt2img 失敗：{e}"
    except Exception as e:
        return None, f"❌ txt2img 錯誤：{e}"

    images = data.get("images")
    if not images:
        return None, "❌ API 未回傳圖片"
    b64 = images[0]
    try:
        raw = base64.b64decode(b64)
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(raw))
        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return arr, f"✅ 產圖完成 | Prompt: {full_prompt[:80]}..."
    except Exception as e:
        return None, f"❌ 解析圖片失敗：{e}"


def render(defaults: dict = None):
    """建立 LoRA 一鍵驗收 Demo 分頁。"""
    lora_dir = (defaults or {}).get("lora_demo_dir") or DEFAULT_LORA_DIR
    choices = _list_lora_files(lora_dir)

    sd_port = (defaults or {}).get("sd_api_port") or DEFAULT_SD_PORT
    gr.Markdown("### LoRA 一鍵驗收 Demo\n選擇 LoRA、調整權重與 Prompt，點擊產圖。請先啟動 SD WebUI 並開啟 `--api`。")
    with gr.Row():
        with gr.Column(scale=1):
            lora_dropdown = gr.Dropdown(
                choices=choices,
                value=choices[0] if choices else "",
                label="LoRA 檔 (.safetensors)",
                allow_custom_value=False,
            )
            with gr.Row():
                lora_dir_input = gr.Textbox(
                    label="LoRA 資料夾",
                    value=lora_dir,
                    placeholder=DEFAULT_LORA_DIR,
                    scale=4,
                )
                btn_browse = gr.Button("瀏覽", size="sm", scale=0)
            btn_refresh_list = gr.Button("🔄 重新掃描 LoRA 清單")
            port_input = gr.Number(
                label="SD WebUI Port",
                value=sd_port,
                minimum=1,
                maximum=65535,
                precision=0,
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                value=DEFAULT_PROMPT,
                lines=4,
                placeholder="(masterpiece:1.2), best quality, 1girl...",
            )
            weight_slider = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                value=1.0,
                step=0.05,
                label="LoRA 權重",
            )
            btn_generate = gr.Button("🎨 開始產圖驗收", variant="primary")
            status_text = gr.Textbox(label="狀態", interactive=False, lines=2)
        with gr.Column(scale=1):
            output_image = gr.Image(label="產出圖片", type="numpy", height=400)

    def _refresh_list(folder):
        ch = _list_lora_files(folder)
        val = ch[0] if ch else ""
        return gr.Dropdown(choices=ch, value=val)

    def _run_generate(lora, prompt, weight, port):
        img, msg = _refresh_and_generate(lora, prompt, weight, port)
        return img, msg

    btn_browse.click(fn=lambda x: shared.browse_folder(x), inputs=[lora_dir_input], outputs=[lora_dir_input])
    btn_refresh_list.click(
        fn=_refresh_list,
        inputs=[lora_dir_input],
        outputs=[lora_dropdown],
    )
    btn_generate.click(
        fn=_run_generate,
        inputs=[lora_dropdown, prompt_input, weight_slider, port_input],
        outputs=[output_image, status_text],
    )

    comps = {
        "lora_dropdown": lora_dropdown,
        "lora_dir_input": lora_dir_input,
        "output_image": output_image,
    }
    return comps, []
