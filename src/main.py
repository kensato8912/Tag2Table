"""
主程式：啟動視窗、整合各模組
支援 Tkinter 與 Gradio 兩種介面
"""
import sys


def main_tk():
    """Tkinter 桌面版"""
    import tkinter as tk
    from .ui_components import TagProcessorGUI
    root = tk.Tk()
    app = TagProcessorGUI(root)
    root.mainloop()


def main_gradio():
    """Gradio Web 版 (http://localhost:7860)"""
    from .app_gradio import launch
    launch(share=False, server_port=7860)


def main():
    """依參數選擇介面：預設 Tkinter，--web 或 -w 啟動 Gradio"""
    if "--web" in sys.argv or "-w" in sys.argv:
        main_gradio()
    else:
        main_tk()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"GUI 啟動失敗: {e}")
        print("請確認已安裝依賴：pip install -r requirements.txt")
