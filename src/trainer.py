import os
import subprocess
import threading

# --- 路徑設定 ---
# 請再次確認 E:\Docker\StabilityMatrix\Data\Packages\kohya_ss 是否正確
KOHYA_ROOT = r"E:\Docker\StabilityMatrix\Data\Packages\kohya_ss"
TRAIN_SCRIPT = os.path.join(KOHYA_ROOT, "sd-scripts", "train_network.py")
PYTHON_EXE = os.path.join(KOHYA_ROOT, "venv", "Scripts", "python.exe")

OUTPUT_DIR = r"E:\Docker\my-python-app\output\models"
LOGGING_DIR = r"E:\Docker\my-python-app\output\logs"


def _stream_output(pipe, log_callback):
    """
    即時讀取子流程輸出，支援 tqdm 的 \\r 進度條。
    log_callback(msg, replace_last=False): replace_last=True 時覆寫前一條（用於 tqdm 更新）。
    """
    buf = ""
    try:
        while True:
            chunk = pipe.read(1024)
            if not chunk:
                break
            buf += chunk
            # 依 \n 與 \r 分割；\r 分隔的視為 tqdm 進度條
            while "\n" in buf or "\r" in buf:
                i = buf.find("\n")
                j = buf.find("\r")
                sep = "\n" if (i >= 0 and (j < 0 or i <= j)) else "\r"
                line, _, buf = buf.partition(sep)
                line = line.strip()
                if not line:
                    continue
                is_tqdm = sep == "\r"
                if log_callback:
                    log_callback(line, replace_last=is_tqdm)
                else:
                    print(f"\r{line}" if is_tqdm else line, end="" if is_tqdm else None, flush=True)
        if buf.strip():
            line = buf.strip()
            if log_callback:
                log_callback(line, replace_last=False)
            else:
                print(line)
    except (BrokenPipeError, ValueError, OSError):
        pass


def start_ken_lora_train(model_path, data_dir, output_name, log_callback=None):
    """
    啟動 LoRA 訓練，透過 log_callback 即時串流輸出。
    log_callback(msg, replace_last=False): replace_last=True 表示 tqdm 進度條，可覆寫前一條。
    """
    if not os.path.exists(TRAIN_SCRIPT):
        msg = f"❌ 找不到訓練腳本：{TRAIN_SCRIPT}"
        if log_callback:
            log_callback(msg, replace_last=False)
        else:
            print(msg)
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOGGING_DIR, exist_ok=True)

    cmd = [
        PYTHON_EXE, TRAIN_SCRIPT,
        "--pretrained_model_name_or_path", model_path,
        "--train_data_dir", data_dir,
        "--output_name", output_name,
        "--resolution", "512,512",
        "--mixed_precision", "fp16",
        "--save_precision", "fp16",
        "--network_module", "networks.lora",
        "--network_dim", "64",
        "--network_alpha", "32",
        "--optimizer_type", "AdamW8bit",
        "--max_train_epochs", "10",
        "--lr_scheduler", "cosine",
        "--learning_rate", "0.0001",
        "--sdpa",
        "--clip_skip", "2",
        "--gradient_checkpointing",
        "--enable_bucket",
        "--min_bucket_reso", "256",
        "--max_bucket_reso", "1024",
        "--output_dir", OUTPUT_DIR,
        "--logging_dir", LOGGING_DIR,
        "--log_with", "tensorboard",
    ]

    if log_callback:
        log_callback("🚀 指令已準備就緒，正在啟動 Stability Matrix 訓練環境...", replace_last=False)
    else:
        print("🚀 指令已準備就緒，正在啟動 Stability Matrix 訓練環境...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
        cwd=os.path.dirname(TRAIN_SCRIPT) or None,
    )

    def _log(line, replace_last=False):
        if log_callback:
            log_callback(line, replace_last=replace_last)
        else:
            print(line)

    def _reader():
        _stream_output(proc.stdout, log_callback)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return proc

# 測試用 (你可以先註解掉，等要跑的時候再打開)
# start_ken_lora_train(r"C:\models\AnythingV5.safetensors", r"D:\train_img", "Ken_Ansha_LoRA")