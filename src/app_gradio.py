"""
Gradio Web UI：向後相容入口
實際實作已移至專案根目錄 web_ui.py，分頁拆至 tabs/ 資料夾
"""
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from web_ui import create_ui, launch

__all__ = ["create_ui", "launch"]

if __name__ == "__main__":
    print("Tag2Table Gradio UI - http://localhost:7860")
    launch(share=False, server_port=7860)
