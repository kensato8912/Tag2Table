"""
共用工具：日誌串流、檔案對話框、設定載入
"""
import queue
import sys
from pathlib import Path

# 專案根目錄加入 path（供 src 匯入）
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from tkinter import filedialog
    _HAS_FILEDIALOG = True
except ImportError:
    _HAS_FILEDIALOG = False

# 全域 log 佇列（供即時串流）
_log_queue = queue.Queue()
_log_buffer = []


def log_callback(msg: str, replace_last: bool = False):
    _log_queue.put(("append", msg, replace_last))


def log_clear():
    _log_buffer.clear()
    _log_queue.put(("clear", None, None))


def get_ollama_url(host_mode: str, remote_ip: str) -> str:
    if host_mode == "remote" and remote_ip.strip():
        return f"http://{remote_ip.strip()}:11434/v1"
    return "http://localhost:11434/v1"


def browse_folder(current: str) -> str:
    if not _HAS_FILEDIALOG:
        return current or ""
    try:
        root = __import__("tkinter").Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="選擇資料夾", initialdir=current or None)
        root.destroy()
        return path if path else (current or "")
    except Exception:
        return current or ""


def browse_file_save(current: str, default_name: str = "AI_Tag_Reference.txt") -> str:
    if not _HAS_FILEDIALOG:
        return current or ""
    try:
        root = __import__("tkinter").Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        init_dir = str(Path(current).parent) if current else None
        init_file = Path(current).name if current else default_name
        path = filedialog.asksaveasfilename(
            title="另存為",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
            initialdir=init_dir,
            initialfile=init_file,
        )
        root.destroy()
        return path if path else (current or "")
    except Exception:
        return current or ""


def browse_file_open(current: str, title: str = "選擇檔案", filetypes=None) -> str:
    if not _HAS_FILEDIALOG:
        return current or ""
    if filetypes is None:
        filetypes = [("safetensors", "*.safetensors"), ("ckpt", "*.ckpt"), ("所有檔案", "*.*")]
    try:
        root = __import__("tkinter").Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        init_dir = str(Path(current).parent) if current else None
        path = filedialog.askopenfilename(title=title, initialdir=init_dir, filetypes=filetypes)
        root.destroy()
        return path if path else (current or "")
    except Exception:
        return current or ""


def stream_from_log_callback(log_producer):
    """通用日誌串流 generator"""
    log_clear()
    _log_buffer.clear()

    def run():
        def log_cb(m, replace_last=False):
            _log_queue.put(("append", m, replace_last))
        try:
            log_producer(log_cb)
        except Exception as e:
            _log_queue.put(("append", f"❌ 錯誤: {e}", False))
        _log_queue.put(("done", None, None))

    import threading
    t = threading.Thread(target=run, daemon=True)
    t.start()

    while True:
        try:
            cmd, msg, replace_last = _log_queue.get(timeout=0.05)
            if cmd == "clear":
                _log_buffer.clear()
            elif cmd == "append":
                if replace_last and _log_buffer:
                    _log_buffer[-1] = msg
                else:
                    _log_buffer.append(msg)
            elif cmd == "done":
                break
        except queue.Empty:
            pass
        yield "\n".join(_log_buffer)
    yield "\n".join(_log_buffer)


def load_defaults():
    from src.database_manager import load_config, TXT_OUTPUT_DIR
    cfg = load_config()
    return {
        "folder_path": cfg.get("folder_path", ""),
        "output_file": cfg.get("output_file", str(TXT_OUTPUT_DIR / "AI_Tag_Reference.txt")),
        "enable_classification": cfg.get("enable_classification", False),
        "translation_mode": cfg.get("translation_mode", "ollama"),
        "gemini_api_key": cfg.get("gemini_api_key", ""),
        "ollama_host_mode": cfg.get("ollama_host_mode", "local"),
        "ollama_remote_ip": cfg.get("ollama_remote_ip", ""),
        "model_name": cfg.get("model_name", "gemma2:2b"),
        "base_model_path": cfg.get("base_model_path", ""),
        "train_data_dir": cfg.get("train_data_dir", ""),
        "lora_output_name": cfg.get("lora_output_name", "Ken_Ansha_LoRA"),
        "sort_tags_by_category": cfg.get("sort_tags_by_category", True),
        "helper_ref_dest": cfg.get("helper_ref_dest", ""),
        "helper_trigger_words": cfg.get("helper_trigger_words", "Niyaniya, Ibuki"),
        "wd14_source_dir": cfg.get("wd14_source_dir", ""),
        "wd14_trigger_word": cfg.get("wd14_trigger_word", "Niyaniya"),
    }
