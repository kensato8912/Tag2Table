"""
主程式：啟動視窗、整合各模組
"""
import tkinter as tk

from ui_components import TagProcessorGUI


def main():
    """主函數"""
    root = tk.Tk()
    app = TagProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"GUI 啟動失敗: {e}")
        print("請確認已安裝依賴：pip install -r requirements.txt")
