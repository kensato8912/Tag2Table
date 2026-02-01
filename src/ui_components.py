"""
介面模組：定義按鈕、分頁、群組包樣式
包含主視窗 TagProcessorGUI、Memo 轉換、題詞組合器
"""
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, simpledialog
from pathlib import Path
import threading

from .database_manager import (
    load_config, save_config, load_tag_database, load_tag_map, load_prompt_presets,
    save_prompt_presets, load_tag_data_for_prompt, generate_folder_report, generate_file_report,
    DB_FILE, tag_map, TXT_OUTPUT_DIR, _ensure_txt_dir,
)
from .ollama_client import get_ollama_models, unload_model
from .processor import process_with_ai
from .trainer import start_ken_lora_train
from .tagger_wd14 import tag_folder as wd14_tag_folder
from .helper_grabber import grab_hand_feet_refs


# --- Memo 轉換視窗 ---

class MemoConvertWindow:
    """Memo 轉換視窗：1. 正常拆解括號  2. 純英文轉 帶括號+中文"""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Memo 轉換")
        self.win.geometry("950x560")
        self.win.minsize(900, 480)
        self.win.resizable(True, True)
        self.win.columnconfigure(0, weight=1)
        self.win.rowconfigure(0, weight=1)

        main = ttk.Frame(self.win, padding="10")
        main.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="資料夾", padding="5")
        left.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        ttk.Button(left, text="瀏覽資料夾...", command=self._browse_folder).grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        list_frame = ttk.Frame(left)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        scroll = ttk.Scrollbar(list_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_listbox = tk.Listbox(list_frame, width=20, yscrollcommand=scroll.set,
                                      font=("Consolas", 9), selectmode=tk.SINGLE)
        self.txt_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.txt_listbox.yview)
        self.txt_listbox.bind("<<ListboxSelect>>", self._on_txt_select)
        self.current_folder = None
        self.txt_paths = []

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        right.columnconfigure(0, weight=1)
        right.columnconfigure(1, weight=0, minsize=220)
        right.rowconfigure(2, weight=1)
        right.rowconfigure(5, weight=1)
        right.rowconfigure(7, weight=1)

        content = ttk.Frame(right)
        content.grid(row=0, column=0, rowspan=9, sticky=(tk.W, tk.E, tk.N, tk.S))
        content.columnconfigure(0, weight=1)
        content.rowconfigure(3, weight=1)
        content.rowconfigure(6, weight=1)
        content.rowconfigure(8, weight=1)

        top_btn_frame = ttk.Frame(content)
        top_btn_frame.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))
        ttk.Button(top_btn_frame, text="存成角色快速套裝", command=self._save_as_preset).pack(side=tk.LEFT, padx=3)

        mode_frame = ttk.Frame(content)
        mode_frame.grid(row=1, column=0, sticky=tk.W, pady=(0, 8))
        ttk.Label(mode_frame, text="模式：").pack(side=tk.LEFT, padx=(0, 8))
        self.mode_var = tk.StringVar(value="normal")
        ttk.Radiobutton(mode_frame, text="1. 正常功能（拆解括號）", variable=self.mode_var, value="normal").pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(mode_frame, text="2. 純英文轉 帶括號+中文", variable=self.mode_var, value="en2zh").pack(side=tk.LEFT)

        ttk.Label(content, text="Memo1 輸入：").grid(row=2, column=0, sticky=tk.W, pady=(0, 3))
        self.input_text = scrolledtext.ScrolledText(content, height=8, width=50, wrap=tk.WORD)
        self.input_text.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        self.input_text.insert(tk.END, "long hair(長髮), blush(臉紅), bangs(劉海), blonde hair(金髮)")

        btn_frame = ttk.Frame(content)
        btn_frame.grid(row=4, column=0, pady=8)
        ttk.Button(btn_frame, text="轉換", command=self._convert).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="清除", command=self._clear).pack(side=tk.LEFT, padx=3)
        ttk.Button(btn_frame, text="開啟 TXT", command=self._open_txt).pack(side=tk.LEFT, padx=3)

        ttk.Label(content, text="視窗1：移除 ( ) 內文字 → ").grid(row=5, column=0, sticky=tk.W, pady=(10, 3))
        self.output1_text = scrolledtext.ScrolledText(content, height=8, width=50, wrap=tk.WORD)
        self.output1_text.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))

        ttk.Label(content, text="視窗2：只顯示 ( ) 當中的文字 → ").grid(row=7, column=0, sticky=tk.W, pady=(10, 3))
        self.output2_text = scrolledtext.ScrolledText(content, height=8, width=50, wrap=tk.WORD)
        self.output2_text.grid(row=8, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))

        preview_frame = ttk.LabelFrame(right, text="預覽 (JPG/PNG)", padding="5")
        preview_frame.grid(row=0, column=1, rowspan=8, sticky=(tk.N, tk.S, tk.E), padx=(10, 0))
        preview_frame.columnconfigure(0, weight=1, minsize=200)
        preview_frame.rowconfigure(0, weight=1, minsize=200)
        self.preview_label = tk.Label(preview_frame, text="選 TXT 載入同檔名圖片", bg="#f0f0f0",
                                      relief=tk.SUNKEN, anchor=tk.CENTER)
        self.preview_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self._preview_photo = None

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="選擇資料夾（列出所有 TXT）")
        if not folder:
            return
        self.current_folder = Path(folder)
        self.txt_paths = sorted(self.current_folder.rglob("*.txt"), key=lambda p: p.name)
        self.txt_listbox.delete(0, tk.END)
        for p in self.txt_paths:
            self.txt_listbox.insert(tk.END, p.name)

    def _on_txt_select(self, event):
        sel = self.txt_listbox.curselection()
        if not sel or not self.txt_paths:
            return
        idx = sel[0]
        if idx >= len(self.txt_paths):
            return
        path = self.txt_paths[idx]
        self._load_and_convert_txt(str(path))
        self._load_preview(path)

    def _load_preview(self, txt_path):
        base = Path(txt_path).with_suffix("")
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
            p = base.with_suffix(ext)
            if p.exists():
                img_path = p
                break
        self.preview_label.config(text="無對應圖片" if not img_path else "")
        self._preview_photo = None
        if not img_path:
            return
        try:
            from PIL import Image, ImageTk
            img = Image.open(img_path)
            w, h = img.size
            max_size = 200
            if max(w, h) > max_size:
                img = img.copy()
                try:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                except AttributeError:
                    img.thumbnail((max_size, max_size), Image.LANCZOS)
            self._preview_photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=self._preview_photo, text="")
        except ImportError:
            self.preview_label.config(text="需安裝 Pillow")
        except Exception:
            self.preview_label.config(text="載入失敗")

    def _convert(self):
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            return
        if self.mode_var.get() == "en2zh":
            self._convert_en2zh(text)
        else:
            self._convert_normal(text)

    def _convert_normal(self, text):
        ptrn = r'(?<!\\)\(([^)]*)(?<!\\)\)'
        out1 = re.sub(ptrn, '', text)
        out1 = re.sub(r',\s*,', ',', out1).strip(' ,')
        matches = re.findall(ptrn, text)
        out2 = ', '.join(matches) if matches else ''
        self._set_output(out1, out2)

    def _convert_en2zh(self, text):
        db = load_tag_database(DB_FILE)
        for k, v in tag_map.items():
            db.setdefault(k, {})['zh_tag'] = v
        text_clean = re.sub(r'(?<!\\)\([^)]*(?<!\\)\)', '', text)
        tags = [t.strip() for t in text_clean.replace('\n', ',').split(',') if t.strip()]
        result_with_bracket = []
        result_zh_only = []
        for tag in tags:
            zh = db.get(tag, {}).get('zh_tag', '未翻譯')
            result_with_bracket.append(f"{tag}({zh})")
            result_zh_only.append(zh)
        out1 = ", ".join(result_with_bracket)
        out2 = ", ".join(result_zh_only)
        self._set_output(out1, out2)

    def _set_output(self, out1, out2):
        self.output1_text.config(state=tk.NORMAL)
        self.output1_text.delete(1.0, tk.END)
        self.output1_text.insert(tk.END, out1)
        self.output2_text.config(state=tk.NORMAL)
        self.output2_text.delete(1.0, tk.END)
        self.output2_text.insert(tk.END, out2)

    def _clear(self):
        self.input_text.delete(1.0, tk.END)
        self.output1_text.config(state=tk.NORMAL)
        self.output1_text.delete(1.0, tk.END)
        self.output2_text.config(state=tk.NORMAL)
        self.output2_text.delete(1.0, tk.END)

    def _load_and_convert_txt(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            try:
                with open(path, 'r', encoding='cp950') as f:
                    text = f.read().strip()
            except Exception as e:
                messagebox.showerror("錯誤", f"讀取失敗: {e}")
                return
        except Exception as e:
            messagebox.showerror("錯誤", f"讀取失敗: {e}")
            return
        if not text:
            messagebox.showinfo("提示", "檔案為空")
            return
        self.input_text.delete(1.0, tk.END)
        self.input_text.insert(tk.END, text)
        self.mode_var.set("en2zh")
        self._convert()

    def _save_as_preset(self):
        text = self.input_text.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "Memo1 為空，請先輸入或載入標籤")
            return
        text_clean = re.sub(r'(?<!\\)\([^)]*(?<!\\)\)', '', text)
        tags = [t.strip() for t in text_clean.replace('\n', ',').split(',') if t.strip()]
        if not tags:
            messagebox.showwarning("提示", "無法解析標籤，請確認格式")
            return
        name = simpledialog.askstring("存成角色快速套裝", "請輸入套裝名稱：", initialvalue="自訂套裝")
        if not name or not name.strip():
            return
        name = name.strip()
        presets = load_prompt_presets()
        presets[name] = tags
        save_prompt_presets(presets)
        messagebox.showinfo("完成", f"已儲存為「{name}」\n題詞組合器下次開啟時可選用")

    def _open_txt(self):
        path = filedialog.askopenfilename(
            title="選擇 TXT 檔（英文逗號分隔）",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")]
        )
        if not path:
            return
        self._load_and_convert_txt(path)
        self._load_preview(Path(path))
        messagebox.showinfo("完成", "已載入並轉換為原文＋帶括號翻譯")


# --- 題詞組合器 ---

class AutoPromptMixerWindow:
    """智慧題詞組合器：分組標籤頁 + 角色快速套裝"""

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Ken's AI 標籤群組管家")
        self.win.geometry("870x780")
        self.win.minsize(750, 550)
        self.win.columnconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=1)

        self.selected_tags = []
        self.tag_data = load_tag_data_for_prompt()

        preset_frame = ttk.LabelFrame(self.win, text=" ⚡ 角色快速套裝 (一鍵載入外觀) ")
        preset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=8)
        preset_frame.columnconfigure(0, weight=1)
        preset_canvas = tk.Canvas(preset_frame, highlightthickness=0, height=55)
        preset_scroll = ttk.Scrollbar(preset_frame, orient="horizontal", command=preset_canvas.xview)
        self.preset_inner = ttk.Frame(preset_canvas)
        self.preset_inner.bind("<Configure>", lambda e: preset_canvas.configure(scrollregion=preset_canvas.bbox("all")))
        preset_canvas.create_window((0, 0), window=self.preset_inner, anchor="nw")
        preset_canvas.configure(xscrollcommand=preset_scroll.set)
        preset_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        preset_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        preset_canvas.bind("<MouseWheel>", lambda e: preset_canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"))
        self.preset_canvas = preset_canvas
        self._build_preset_buttons()

        self.notebook = ttk.Notebook(self.win)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        self.notebook.rowconfigure(0, weight=1)
        self.win.rowconfigure(1, weight=1)
        self.win.rowconfigure(2, weight=1)

        self._build_notebook_tabs()

        footer = ttk.LabelFrame(self.win, text="已選題詞", padding=8)
        footer.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=8)
        footer.columnconfigure(0, weight=1)
        footer.rowconfigure(0, weight=1)
        self.text_area = scrolledtext.ScrolledText(footer, height=6, font=("Consolas", 10), wrap=tk.WORD)
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        btn_row = ttk.Frame(footer)
        btn_row.grid(row=1, column=0, sticky=tk.W)
        ttk.Button(btn_row, text="清空", command=self._clear_tags).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="複製提示詞", command=self._copy_result).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="重新更新", command=self._reload_data).pack(side=tk.LEFT, padx=5)

    def _build_preset_buttons(self):
        """建立或重建角色快速套裝按鈕"""
        for w in self.preset_inner.winfo_children():
            w.destroy()
        for name, tags in load_prompt_presets().items():
            ttk.Button(self.preset_inner, text=name, command=lambda t=tags: self._load_preset(t)).pack(side=tk.LEFT, padx=5, pady=3)

    def _build_notebook_tabs(self):
        """建立或重建 Notebook 分頁"""
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)
        if not self.tag_data:
            ttk.Label(self.notebook, text="❌ 找不到 all_characters_tags.json", foreground="red").pack(pady=30, padx=20)
        else:
            sort_order = ["角色與作品 (Character)", "衣服配件 (Clothing)", "姿態動作 (Pose)",
                          "身體特徵 (Body)", "背景環境 (Background)", "光線風格 (Style)", "其他 (General)"]
            short_names = {"角色與作品 (Character)": "角色", "衣服配件 (Clothing)": "衣服",
                          "姿態動作 (Pose)": "姿態", "身體特徵 (Body)": "身體",
                          "背景環境 (Background)": "背景", "光線風格 (Style)": "光線", "其他 (General)": "其他"}
            for cat in sort_order:
                if cat in self.tag_data:
                    tab = ttk.Frame(self.notebook)
                    self.notebook.add(tab, text=f" {short_names.get(cat, cat.split(' ')[0])} ")
                    self._create_tag_buttons(tab, self.tag_data[cat])

    def _reload_data(self):
        """重新載入資料庫、角色套裝並更新分頁"""
        self.tag_data = load_tag_data_for_prompt()
        self._build_preset_buttons()
        self._build_notebook_tabs()
        messagebox.showinfo("完成", "已重新載入 all_characters_tags.json 與 prompt_presets.json")

    def _create_tag_buttons(self, parent_frame, tags):
        canvas = tk.Canvas(parent_frame)
        scrollbar = ttk.Scrollbar(parent_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        for i, (en, zh) in enumerate(tags):
            btn = ttk.Button(scrollable, text=f"{en}\n({zh})", width=24,
                             command=lambda e=en: self._add_tag(e))
            btn.grid(row=i // 4, column=i % 4, padx=5, pady=4, sticky=(tk.W, tk.E))
        for c in range(4):
            scrollable.columnconfigure(c, weight=1)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    def _load_preset(self, tags):
        self.selected_tags = list(dict.fromkeys(self.selected_tags + tags))
        self._update_display()

    def _add_tag(self, tag):
        if tag not in self.selected_tags:
            self.selected_tags.append(tag)
            self._update_display()

    def _update_display(self):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        self.text_area.insert(tk.END, ", ".join(self.selected_tags))

    def _clear_tags(self):
        self.selected_tags = []
        self._update_display()

    def _copy_result(self):
        final = ", ".join(self.selected_tags)
        if not final:
            messagebox.showinfo("提示", "請先選擇標籤")
            return
        try:
            import pyperclip
            pyperclip.copy(final)
            messagebox.showinfo("成功", "組合完成！已複製到剪貼簿。")
        except ImportError:
            self.win.clipboard_clear()
            self.win.clipboard_append(final)
            self.text_area.tag_add(tk.SEL, "1.0", "end")
            self.text_area.mark_set(tk.INSERT, "1.0")
            self.text_area.see(tk.INSERT)
            messagebox.showinfo("完成", "題詞已複製（請 Ctrl+C 貼上）\n或執行 pip install pyperclip")


# --- 標籤翻譯分頁 ---

class TagTranslatePage(ttk.Frame):
    """標籤翻譯管理分頁：資料夾選取、Gemini/Ollama 設定、進度條"""

    def __init__(self, parent, gui):
        super().__init__(parent)
        self.gui = gui
        self._build()

    def _build(self):
        self.columnconfigure(1, weight=1)
        P = {"padx": 10, "pady": 5}

        data_frame = ttk.LabelFrame(self, text="資料來源與輸出", padding=10)
        data_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), **P)
        data_frame.columnconfigure(1, weight=1)
        ttk.Label(data_frame, text="訓練資料夾：").grid(row=0, column=0, sticky=tk.W, **P)
        ttk.Entry(data_frame, textvariable=self.gui.folder_path, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), **P)
        ttk.Button(data_frame, text="瀏覽...", command=self.gui.browse_folder).grid(row=0, column=2, **P)
        ttk.Label(data_frame, text="存檔路徑：").grid(row=1, column=0, sticky=tk.W, **P)
        ttk.Entry(data_frame, textvariable=self.gui.output_file, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), **P)
        ttk.Button(data_frame, text="另存為...", command=self.gui.browse_output_file).grid(row=1, column=2, **P)
        ttk.Checkbutton(data_frame, text="啟用分類輸出（依身體、衣服、姿態等分組）",
                       variable=self.gui.enable_classification).grid(row=2, column=0, columnspan=2, sticky=tk.W, **P)

        engine_frame = ttk.LabelFrame(self, text="翻譯引擎", padding=10)
        engine_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), **P)
        engine_frame.columnconfigure(1, weight=1)
        mode_frame = ttk.Frame(engine_frame)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky=tk.W, **P)
        ttk.Radiobutton(mode_frame, text="雲端 Gemini（精度高，不佔顯存）",
                       variable=self.gui.translation_mode, value="gemini",
                       command=self.gui.toggle_translation_mode).pack(side=tk.LEFT, **P)
        ttk.Radiobutton(mode_frame, text="本地 Ollama（完全隱私，需啟動 Ollama）",
                       variable=self.gui.translation_mode, value="ollama",
                       command=self.gui.toggle_translation_mode).pack(side=tk.LEFT, **P)

        self.gemini_label = ttk.Label(engine_frame, text="Gemini API Key：")
        self.gemini_label.grid(row=1, column=0, sticky=tk.W, **P)
        self.gemini_row = ttk.Frame(engine_frame)
        self.gemini_row.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), **P)
        self.gemini_row.columnconfigure(0, weight=1)
        self.gemini_entry = ttk.Entry(self.gemini_row, textvariable=self.gui.gemini_api_key, width=50, show="●")
        self.gemini_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10), pady=5)
        ttk.Button(self.gemini_row, text="儲存", command=self.gui.save_config_clicked).grid(row=0, column=1, **P)

        self.ollama_frame = ttk.LabelFrame(engine_frame, text="Ollama 設定", padding=10)
        self.ollama_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), **P)
        self.ollama_frame.columnconfigure(2, weight=1)
        host_frame = ttk.Frame(self.ollama_frame)
        host_frame.grid(row=0, column=0, columnspan=4, sticky=tk.W, **P)
        ttk.Label(host_frame, text="連線目標：").pack(side=tk.LEFT, padx=(0, 10), pady=5)
        ttk.Radiobutton(host_frame, text="本地", variable=self.gui.ollama_host_mode, value="local",
                        command=self.gui._sync_ollama_mode).pack(side=tk.LEFT, **P)
        ttk.Radiobutton(host_frame, text="遠端", variable=self.gui.ollama_host_mode, value="remote",
                        command=self.gui._sync_ollama_mode).pack(side=tk.LEFT, **P)
        ttk.Button(host_frame, text="載入模型", command=self.gui.load_models).pack(side=tk.LEFT, **P)
        ttk.Button(host_frame, text="卸載模型", command=self.gui.unload_model_clicked).pack(side=tk.LEFT, **P)
        self.ollama_remote_row = ttk.Frame(self.ollama_frame)
        self.ollama_remote_row.grid(row=1, column=0, columnspan=4, sticky=tk.W, **P)
        ttk.Label(self.ollama_remote_row, text="遠端 IP：").pack(side=tk.LEFT, padx=(0, 10), pady=5)
        ttk.Entry(self.ollama_remote_row, textvariable=self.gui.ollama_remote_ip, width=20).pack(
            side=tk.LEFT, padx=(0, 10), pady=5)
        ttk.Label(self.ollama_remote_row, text="（同 Wi‑Fi 下 Mac/另一台電腦的 IP）").pack(side=tk.LEFT, **P)
        ttk.Label(self.ollama_frame, text="模型：").grid(row=2, column=0, sticky=tk.W, **P)
        model_frame = ttk.Frame(self.ollama_frame)
        model_frame.grid(row=2, column=1, columnspan=3, sticky=(tk.W, tk.E), **P)
        model_frame.columnconfigure(0, weight=1)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.gui.model_name, width=45)
        self.model_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10), pady=5)

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=15)
        self.process_button = ttk.Button(btn_frame, text="開始處理", command=self.gui.start_processing)
        self.process_button.pack(side=tk.LEFT, **P)
        ttk.Button(btn_frame, text="產生資料夾報告", command=self.gui.generate_report_clicked).pack(side=tk.LEFT, **P)
        ttk.Button(btn_frame, text="逐檔報告", command=self.gui.generate_file_report_clicked).pack(side=tk.LEFT, **P)
        ttk.Button(btn_frame, text="Memo 轉換", command=self.gui.open_memo_convert).pack(side=tk.LEFT, **P)
        ttk.Button(btn_frame, text="題詞組合器", command=self.gui.open_prompt_mixer).pack(side=tk.LEFT, **P)

        self.progress = ttk.Progressbar(self, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), **P)


# --- LoRA 訓練分頁 ---

class LoraTrainPage(ttk.Frame):
    """LoRA 一鍵訓練分頁：底模、圖片資料夾、輸出名稱"""

    def __init__(self, parent, gui):
        super().__init__(parent)
        self.gui = gui
        self._build()

    def _build(self):
        self.columnconfigure(0, weight=1)
        P = {"padx": 10, "pady": 5}

        train_frame = ttk.LabelFrame(self, text="LoRA 訓練參數", padding=15)
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), **P)
        train_frame.columnconfigure(1, weight=1)

        ttk.Label(train_frame, text="底模：").grid(row=0, column=0, sticky=tk.W, **P)
        ttk.Entry(train_frame, textvariable=self.gui.base_model_path, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), **P)
        ttk.Button(train_frame, text="瀏覽...", command=self._browse_base_model).grid(row=0, column=2, **P)

        ttk.Label(train_frame, text="圖片資料夾：").grid(row=1, column=0, sticky=tk.W, **P)
        ttk.Entry(train_frame, textvariable=self.gui.train_data_dir, width=50).grid(
            row=1, column=1, sticky=(tk.W, tk.E), **P)
        ttk.Button(train_frame, text="瀏覽...", command=self._browse_train_data_dir).grid(row=1, column=2, **P)

        ttk.Label(train_frame, text="輸出名稱：").grid(row=2, column=0, sticky=tk.W, **P)
        ttk.Entry(train_frame, textvariable=self.gui.lora_output_name, width=50).grid(
            row=2, column=1, sticky=(tk.W, tk.E), **P)

        ttk.Checkbutton(train_frame, text="自動按類別排序標籤", variable=self.gui.sort_tags_by_category).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, **P)

        ttk.Label(train_frame, text="WD14 輸入資料夾：").grid(row=4, column=0, sticky=tk.W, **P)
        ttk.Entry(train_frame, textvariable=self.gui.wd14_source_dir, width=50).grid(
            row=4, column=1, sticky=(tk.W, tk.E), **P)
        ttk.Button(train_frame, text="瀏覽...", command=self._browse_wd14_source).grid(row=4, column=2, **P)

        ttk.Label(train_frame, text="WD14 前置詞：").grid(row=5, column=0, sticky=tk.W, **P)
        ttk.Entry(train_frame, textvariable=self.gui.wd14_trigger_word, width=50).grid(
            row=5, column=1, sticky=(tk.W, tk.E), **P)

        helper_frame = ttk.LabelFrame(self, text="自訂篩選", padding=10)
        helper_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), **P)
        helper_frame.columnconfigure(1, weight=1)
        ttk.Label(helper_frame, text="參考資料夾：").grid(row=0, column=0, sticky=tk.W, **P)
        ttk.Entry(helper_frame, textvariable=self.gui.helper_ref_dest, width=50).grid(
            row=0, column=1, sticky=(tk.W, tk.E), **P)
        ttk.Button(helper_frame, text="瀏覽...", command=self._browse_helper_ref).grid(row=0, column=2, **P)
        ttk.Label(helper_frame, text="觸發詞排除：").grid(row=1, column=0, sticky=tk.W, **P)
        ttk.Entry(helper_frame, textvariable=self.gui.helper_trigger_words, width=50).grid(
            row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), **P)
        ttk.Button(helper_frame, text="篩選手腳參考圖", command=self._run_helper_grabber).grid(
            row=2, column=0, columnspan=2, pady=(5, 0), sticky=tk.W)

        monitor_frame = ttk.LabelFrame(self, text="自訂分類監控", padding=10)
        monitor_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), **P)
        monitor_frame.columnconfigure(0, weight=1)
        monitor_frame.columnconfigure(1, weight=1)
        ttk.Label(monitor_frame, text="關鍵字（逗號分隔）").grid(row=0, column=0, sticky=tk.W, **P)
        ttk.Label(monitor_frame, text="目標資料夾與倍率").grid(row=0, column=1, sticky=tk.W, **P)
        self.monitor_rows = []
        for i in range(4):
            kw_var = tk.StringVar()
            fd_var = tk.StringVar()
            ttk.Entry(monitor_frame, textvariable=kw_var, width=35).grid(row=i + 1, column=0, sticky=(tk.W, tk.E), **P)
            ttk.Entry(monitor_frame, textvariable=fd_var, width=25).grid(row=i + 1, column=1, sticky=(tk.W, tk.E), **P)
            self.monitor_rows.append((kw_var, fd_var))
        mon_btn_row = ttk.Frame(monitor_frame)
        mon_btn_row.grid(row=5, column=0, columnspan=2, sticky=tk.W, **P)
        ttk.Button(mon_btn_row, text="執行分類監控", command=self._run_custom_classify).pack(side=tk.LEFT, padx=(0, 10), pady=5)

        action_frame = ttk.LabelFrame(self, text="執行", padding=10)
        action_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), **P)
        action_frame.columnconfigure(0, weight=1)
        btn_row = ttk.Frame(action_frame)
        btn_row.pack(padx=10, pady=5)
        self.train_button = ttk.Button(btn_row, text="一鍵訓練", command=self._start_lora_train)
        self.train_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        self.stop_train_button = ttk.Button(btn_row, text="停止訓練", command=self._stop_lora_train, state=tk.DISABLED)
        self.stop_train_button.pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Button(btn_row, text="WD14 標籤", command=self._run_wd14_tagger).pack(side=tk.LEFT, padx=10, pady=5)

    def _browse_base_model(self):
        path = filedialog.askopenfilename(
            title="選擇底模",
            filetypes=[
                ("safetensors", "*.safetensors"),
                ("ckpt", "*.ckpt"),
                ("所有檔案", "*.*")
            ]
        )
        if path:
            self.gui.base_model_path.set(path)

    def _browse_train_data_dir(self):
        folder = filedialog.askdirectory(title="選擇圖片資料夾")
        if folder:
            self.gui.train_data_dir.set(folder)

    def _browse_wd14_source(self):
        folder = filedialog.askdirectory(title="選擇 WD14 輸入資料夾")
        if folder:
            self.gui.wd14_source_dir.set(folder)

    def _browse_helper_ref(self):
        folder = filedialog.askdirectory(title="選擇參考資料夾（輸出位置）")
        if folder:
            self.gui.helper_ref_dest.set(folder)

    def _run_helper_grabber(self):
        source = self.gui.train_data_dir.get().strip()
        dest = self.gui.helper_ref_dest.get().strip()
        if not source:
            messagebox.showwarning("提示", "請先選擇圖片資料夾（來源）")
            return
        if not dest:
            dest_default = str(Path(source).parent / "hands_feet_ref")
            self.gui.helper_ref_dest.set(dest_default)
            dest = dest_default
        if self.gui.is_processing:
            messagebox.showwarning("警告", "處理中，請稍候...")
            return
        triggers_raw = self.gui.helper_trigger_words.get().strip()
        triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()] if triggers_raw else ["Niyaniya", "Ibuki"]
        self.gui.is_processing = True
        self.train_button.config(state=tk.DISABLED)
        root = self.gui.root

        def run_grab():
            try:
                n = grab_hand_feet_refs(
                    source, dest, trigger_words=triggers, recursive=True,
                    log_callback=lambda m: root.after(0, lambda _m=m: self.gui.log(_m))
                )
                root.after(0, lambda: messagebox.showinfo("完成", f"自訂篩選完成，共複製 {n} 張圖片"))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("錯誤", str(e)))
                self.gui.log(f"❌ 自訂篩選失敗: {e}")
            finally:
                root.after(0, lambda: setattr(self.gui, 'is_processing', False))
                root.after(0, lambda: self.train_button.config(state=tk.NORMAL))

        threading.Thread(target=run_grab, daemon=True).start()

    def _run_custom_classify(self):
        source = self.gui.train_data_dir.get().strip()
        dest = self.gui.helper_ref_dest.get().strip()
        if not source:
            messagebox.showwarning("提示", "請先選擇圖片資料夾（來源）")
            return
        if not dest:
            dest_default = str(Path(source).parent / "custom_ref")
            self.gui.helper_ref_dest.set(dest_default)
            dest = dest_default
        if self.gui.is_processing:
            messagebox.showwarning("警告", "處理中，請稍候...")
            return
        rules = {}
        for i, (kw_var, fd_var) in enumerate(self.monitor_rows):
            kw = kw_var.get().strip()
            fd = fd_var.get().strip()
            if not kw or not fd:
                continue
            keywords = [k.strip() for k in kw.split(",") if k.strip()]
            if not keywords:
                continue
            rule_id = f"rule_{i}"
            priority = 10
            if fd and fd[0].isdigit():
                m = re.match(r"^(\d+)_", fd)
                if m:
                    priority = int(m.group(1))
            rules[rule_id] = {
                "folder_name": fd,
                "keywords": keywords,
                "priority": priority,
            }
        if not rules:
            messagebox.showwarning("提示", "請至少填寫一列：關鍵字與目標資料夾")
            return
        triggers_raw = self.gui.helper_trigger_words.get().strip()
        triggers = [t.strip() for t in triggers_raw.split(",") if t.strip()] if triggers_raw else ["Niyaniya", "Ibuki"]
        self.gui.is_processing = True
        self.train_button.config(state=tk.DISABLED)
        root = self.gui.root

        def run_custom():
            try:
                n = grab_hand_feet_refs(
                    source, dest, trigger_words=triggers, recursive=True, rules=rules,
                    log_callback=lambda m: root.after(0, lambda _m=m: self.gui.log(_m))
                )
                root.after(0, lambda: messagebox.showinfo("完成", f"自訂分類完成，共複製 {n} 張圖片"))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("錯誤", str(e)))
                self.gui.log(f"❌ 自訂分類失敗: {e}")
            finally:
                root.after(0, lambda: setattr(self.gui, "is_processing", False))
                root.after(0, lambda: self.train_button.config(state=tk.NORMAL))

        threading.Thread(target=run_custom, daemon=True).start()

    def _start_lora_train(self):
        model_path = self.gui.base_model_path.get().strip()
        data_dir = self.gui.train_data_dir.get().strip()
        output_name = self.gui.lora_output_name.get().strip() or "Ken_Ansha_LoRA"
        if not model_path:
            messagebox.showwarning("提示", "請選擇底模")
            return
        if not data_dir:
            messagebox.showwarning("提示", "請選擇圖片資料夾")
            return
        if self.gui.is_processing:
            messagebox.showwarning("警告", "處理中，請稍候...")
            return
        self.gui.is_processing = True
        self.train_button.config(state=tk.DISABLED)
        if hasattr(self.gui, 'translate_page') and self.gui.translate_page:
            self.gui.translate_page.process_button.config(state=tk.DISABLED)
        self.gui.log("🚀 啟動 LoRA 訓練...")
        self.gui.log(f"  底模: {model_path}")
        self.gui.log(f"  圖片: {data_dir}")
        self.gui.log(f"  輸出: {output_name}")
        root = self.gui.root

        def log_stream(msg, replace_last=False):
            root.after(0, lambda m=msg, r=replace_last: self.gui.log(m, replace_last=r))

        def run_train():
            try:
                proc = start_ken_lora_train(model_path, data_dir, output_name, log_callback=log_stream)
                if proc is None:
                    root.after(0, self._restore_buttons)
                    return
                self.gui.current_train_process = proc
                root.after(0, lambda: self.stop_train_button.config(state=tk.NORMAL))
                proc.wait()
                self.gui.current_train_process = None
                if getattr(self.gui, 'train_was_stopped', False):
                    root.after(0, lambda: messagebox.showinfo("已停止", "LoRA 訓練已手動停止"))
                else:
                    root.after(0, lambda: messagebox.showinfo("完成", "LoRA 訓練已結束"))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("錯誤", str(e)))
                self.gui.log(f"❌ 訓練失敗: {e}")
            finally:
                root.after(0, self._restore_buttons)

        threading.Thread(target=run_train, daemon=True).start()

    def _restore_buttons(self):
        self.gui.is_processing = False
        self.gui.train_was_stopped = False
        self.gui.current_train_process = None
        self.train_button.config(state=tk.NORMAL)
        self.stop_train_button.config(state=tk.DISABLED)
        if hasattr(self.gui, 'translate_page') and self.gui.translate_page:
            self.gui.translate_page.process_button.config(state=tk.NORMAL)

    def _run_wd14_tagger(self):
        data_dir = self.gui.wd14_source_dir.get().strip() or self.gui.train_data_dir.get().strip()
        if not data_dir:
            messagebox.showwarning("提示", "請選擇 WD14 輸入資料夾或圖片資料夾")
            return
        if self.gui.is_processing:
            messagebox.showwarning("警告", "處理中，請稍候...")
            return
        trigger = self.gui.wd14_trigger_word.get().strip() or "Niyaniya"
        sort_by = self.gui.sort_tags_by_category.get()
        self.gui.is_processing = True
        self.train_button.config(state=tk.DISABLED)
        root = self.gui.root

        def run_wd14():
            try:
                n = wd14_tag_folder(
                    data_dir, trigger_word=trigger, sort_by_category=sort_by,
                    log_callback=lambda m: root.after(0, lambda _m=m: self.gui.log(_m))
                )
                root.after(0, lambda: messagebox.showinfo("完成", f"WD14 標籤完成，共處理 {n} 張圖片"))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("錯誤", str(e)))
                self.gui.log(f"❌ WD14 標籤失敗: {e}")
            finally:
                root.after(0, lambda: setattr(self.gui, 'is_processing', False))
                root.after(0, lambda: self.train_button.config(state=tk.NORMAL))

        threading.Thread(target=run_wd14, daemon=True).start()

    def _stop_lora_train(self):
        proc = getattr(self.gui, 'current_train_process', None)
        if proc is None or proc.poll() is not None:
            return
        self.gui.train_was_stopped = True
        try:
            proc.terminate()
            self.gui.log("⏹ 訓練已手動停止")
        except Exception as e:
            self.gui.log(f"❌ 停止失敗: {e}")


# --- 主視窗 ---

class TagProcessorGUI:
    """主視窗：分頁（標籤翻譯 / LoRA 訓練）+ 共用日誌"""

    def __init__(self, root):
        self.root = root
        self.root.title("Tag2Table — AI 標籤統計與翻譯工具")
        self.root.geometry("750x820")
        self.root.resizable(True, True)

        self.folder_path = tk.StringVar()
        self.ollama_url = tk.StringVar(value="http://localhost:11434/v1")
        self.ollama_host_mode = tk.StringVar(value="local")
        self.ollama_remote_ip = tk.StringVar(value="")
        self.model_name = tk.StringVar(value="gemma2:2b")
        self.output_file = tk.StringVar(value=str(TXT_OUTPUT_DIR / "AI_Tag_Reference.txt"))
        self.enable_classification = tk.BooleanVar(value=False)
        self.translation_mode = tk.StringVar(value="ollama")
        self.gemini_api_key = tk.StringVar()
        self.is_processing = False
        self.base_model_path = tk.StringVar()
        self.train_data_dir = tk.StringVar()
        self.lora_output_name = tk.StringVar(value="Ken_Ansha_LoRA")
        self.sort_tags_by_category = tk.BooleanVar(value=True)
        self.helper_ref_dest = tk.StringVar()
        self.helper_trigger_words = tk.StringVar(value="Niyaniya, Ibuki")
        self.wd14_source_dir = tk.StringVar()
        self.wd14_trigger_word = tk.StringVar(value="Niyaniya")
        self.model_was_loaded = False
        self.current_train_process = None
        self.train_was_stopped = False

        config = load_config()
        if config.get("gemini_api_key"):
            self.gemini_api_key.set(config["gemini_api_key"])
        if config.get("folder_path"):
            self.folder_path.set(config["folder_path"])
        if config.get("output_file"):
            self.output_file.set(config["output_file"])
        if config.get("ollama_url"):
            self.ollama_url.set(config["ollama_url"])
        if config.get("ollama_host_mode"):
            self.ollama_host_mode.set(config["ollama_host_mode"])
        if config.get("ollama_remote_ip"):
            self.ollama_remote_ip.set(config["ollama_remote_ip"])
        elif config.get("ollama_url") and "localhost" not in config["ollama_url"]:
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", config.get("ollama_url", ""))
            if m:
                self.ollama_remote_ip.set(m.group(1))
        if config.get("base_model_path"):
            self.base_model_path.set(config["base_model_path"])
        if config.get("train_data_dir"):
            self.train_data_dir.set(config["train_data_dir"])
        if config.get("lora_output_name"):
            self.lora_output_name.set(config["lora_output_name"])
        if "sort_tags_by_category" in config:
            self.sort_tags_by_category.set(config["sort_tags_by_category"])
        if config.get("helper_ref_dest"):
            self.helper_ref_dest.set(config["helper_ref_dest"])
        if config.get("helper_trigger_words"):
            self.helper_trigger_words.set(config["helper_trigger_words"])
        if config.get("wd14_source_dir"):
            self.wd14_source_dir.set(config["wd14_source_dir"])
        if config.get("wd14_trigger_word"):
            self.wd14_trigger_word.set(config["wd14_trigger_word"])

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=0)
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=0)
        main_frame.rowconfigure(3, weight=1, minsize=200)

        ttk.Label(main_frame, text="Tag2Table — AI 標籤統計與翻譯工具",
                  font=("Arial", 16, "bold")).grid(row=0, column=0, pady=(0, 15), padx=10)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)

        self.translate_page = TagTranslatePage(self.notebook, self)
        self.notebook.add(self.translate_page, text=" 標籤翻譯管理 ")
        self.lora_page = LoraTrainPage(self.notebook, self)
        self.notebook.add(self.lora_page, text=" LoRA 一鍵訓練 ")

        ttk.Label(main_frame, text="執行日誌：").grid(row=2, column=0, sticky=tk.W, pady=(15, 5), padx=10)

        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, width=80, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.toggle_translation_mode()

    def _get_ollama_url(self):
        if self.ollama_host_mode.get() == "remote":
            ip = self.ollama_remote_ip.get().strip()
            if ip:
                return f"http://{ip}:11434/v1"
            return ""
        return "http://localhost:11434/v1"

    def _sync_ollama_mode(self):
        is_remote = self.ollama_host_mode.get() == "remote"
        row = self.translate_page.ollama_remote_row
        if is_remote:
            row.grid()
            url = self._get_ollama_url()
            self.ollama_url.set(url)
        else:
            row.grid_remove()
            self.ollama_url.set("http://localhost:11434/v1")

    def save_config_clicked(self):
        config = {
            "gemini_api_key": self.gemini_api_key.get(),
            "folder_path": self.folder_path.get(),
            "output_file": self.output_file.get(),
            "ollama_url": self._get_ollama_url() or self.ollama_url.get(),
            "ollama_host_mode": self.ollama_host_mode.get(),
            "ollama_remote_ip": self.ollama_remote_ip.get(),
            "base_model_path": self.base_model_path.get(),
            "train_data_dir": self.train_data_dir.get(),
            "lora_output_name": self.lora_output_name.get(),
            "sort_tags_by_category": self.sort_tags_by_category.get(),
            "helper_ref_dest": self.helper_ref_dest.get(),
            "helper_trigger_words": self.helper_trigger_words.get(),
            "wd14_source_dir": self.wd14_source_dir.get(),
            "wd14_trigger_word": self.wd14_trigger_word.get(),
        }
        save_config(config)
        messagebox.showinfo("完成", "設定已儲存")

    def _on_close(self):
        if (self.model_was_loaded and self.translation_mode.get() == "ollama"
                and self.model_name.get()):
            try:
                url = self._get_ollama_url()
                if url:
                    unload_model(url, self.model_name.get())
            except Exception:
                pass
        if self.gemini_api_key.get() or self.folder_path.get() or self.base_model_path.get():
            url = self._get_ollama_url()
            config = {
                "gemini_api_key": self.gemini_api_key.get(),
                "folder_path": self.folder_path.get(),
                "output_file": self.output_file.get(),
                "ollama_url": url or self.ollama_url.get(),
                "ollama_host_mode": self.ollama_host_mode.get(),
                "ollama_remote_ip": self.ollama_remote_ip.get(),
                "base_model_path": self.base_model_path.get(),
                "train_data_dir": self.train_data_dir.get(),
                "lora_output_name": self.lora_output_name.get(),
                "sort_tags_by_category": self.sort_tags_by_category.get(),
                "helper_ref_dest": self.helper_ref_dest.get(),
                "helper_trigger_words": self.helper_trigger_words.get(),
                "wd14_source_dir": self.wd14_source_dir.get(),
                "wd14_trigger_word": self.wd14_trigger_word.get(),
            }
            save_config(config)
        self.root.destroy()

    def toggle_translation_mode(self):
        p = self.translate_page
        is_ollama = self.translation_mode.get() == "ollama"
        if is_ollama:
            p.gemini_label.grid_remove()
            p.gemini_row.grid_remove()
        else:
            p.gemini_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
            p.gemini_row.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=10, pady=5)
        if is_ollama:
            p.ollama_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=10, pady=5)
        else:
            p.ollama_frame.grid_remove()
        self._sync_ollama_mode()

    def generate_report_clicked(self):
        folder = self.folder_path.get().strip()
        if not folder:
            messagebox.showwarning("提示", "請先選擇訓練資料夾")
            return
        self.log("📋 正在產生資料夾報告...")
        try:
            ok = generate_folder_report(folder, report_file=None, db_path=DB_FILE, log_callback=self.log, open_after=True)
            if ok:
                messagebox.showinfo("完成", "資料夾報告已生成並開啟")
        except Exception as e:
            self.log(f"❌ 產生報告失敗: {e}")
            messagebox.showerror("錯誤", str(e))

    def open_memo_convert(self):
        MemoConvertWindow(self.root)

    def open_prompt_mixer(self):
        AutoPromptMixerWindow(self.root)

    def generate_file_report_clicked(self):
        folder = self.folder_path.get().strip()
        if not folder:
            messagebox.showwarning("提示", "請先選擇訓練資料夾")
            return
        self.log("📋 正在產生逐檔報告...")
        try:
            ok = generate_file_report(folder, report_file=None, db_path=DB_FILE, log_callback=self.log, open_after=True)
            if ok:
                messagebox.showinfo("完成", "逐檔報告已生成並開啟")
        except Exception as e:
            self.log(f"❌ 產生報告失敗: {e}")
            messagebox.showerror("錯誤", str(e))

    def browse_folder(self):
        folder = filedialog.askdirectory(title="選擇訓練資料夾")
        if folder:
            self.folder_path.set(folder)

    def browse_output_file(self):
        _ensure_txt_dir()
        filepath = filedialog.asksaveasfilename(
            title="另存為",
            defaultextension=".txt",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
            initialdir=str(TXT_OUTPUT_DIR),
            initialfile="AI_Tag_Reference.txt"
        )
        if filepath:
            self.output_file.set(filepath)

    def unload_model_clicked(self):
        model = self.model_name.get()
        if not model:
            messagebox.showwarning("提示", "請先選擇要釋放的模型")
            return
        target = "遠端" if self.ollama_host_mode.get() == "remote" else "本機"
        if not messagebox.askyesno("確認", f"確定要將模型「{model}」從{target}顯存釋放嗎？"):
            return
        self.log(f"🔄 正在從{target}釋放模型...")
        success = unload_model(self._get_ollama_url(), model, log_callback=self.log)
        if success:
            messagebox.showinfo("完成", "模型已從顯存釋放")
        else:
            messagebox.showwarning("提示", "釋放失敗，請查看日誌")

    def load_models(self):
        url = self._get_ollama_url()
        if not url:
            messagebox.showwarning("提示", "遠端模式請先填入 IP；本地模式請確認 Ollama 已啟動")
            return
        target = "遠端" if self.ollama_host_mode.get() == "remote" else "本機"
        self.log(f"🔄 正在從{target}載入模型列表...")
        models = get_ollama_models(url, log_callback=self.log)
        if models:
            self.model_was_loaded = True
            self.translate_page.model_combo['values'] = models
            if models and not self.model_name.get():
                self.model_name.set(models[0])
            self.log(f"✅ 已載入 {len(models)} 個模型")
            messagebox.showinfo("完成", f"已載入 {len(models)} 個模型，請從下拉選單選擇")
        else:
            self.log("❌ 未找到任何模型，請確認 Ollama 已啟動且已 pull 模型")
            messagebox.showwarning("提示", "未找到模型\n請確認 Ollama 服務已啟動\n並執行 ollama pull <模型名> 下載模型")

    def log(self, message, replace_last=False):
        """append message to log; replace_last=True 時覆寫最後一行（用於 tqdm 進度條）"""
        self.log_text.config(state=tk.NORMAL)
        if replace_last:
            content = self.log_text.get("1.0", tk.END)
            if content.rstrip():
                last_nl = content.rfind("\n", 0, len(content) - 1)
                start = f"1.0+{last_nl + 1}c" if last_nl >= 0 else "1.0"
                self.log_text.delete(start, tk.END)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def update_progress(self, value, max_value):
        if max_value > 0:
            self.translate_page.progress.config(mode='determinate', maximum=max_value, value=value)
        self.root.update_idletasks()

    def start_processing(self):
        if self.is_processing:
            messagebox.showwarning("警告", "處理中，請稍候...")
            return

        if not self.folder_path.get():
            messagebox.showerror("錯誤", "請選擇訓練資料夾！")
            return

        mode = self.translation_mode.get()
        if mode == "gemini":
            if not self.gemini_api_key.get():
                messagebox.showerror("錯誤", "請輸入 Gemini API Key！")
                return
        else:
            url = self._get_ollama_url()
            if not url:
                msg = "請填入遠端 IP！" if self.ollama_host_mode.get() == "remote" else "請確認 Ollama 服務地址！"
                messagebox.showerror("錯誤", msg)
                return
            if not self.model_name.get():
                messagebox.showerror("錯誤", "請選擇或輸入模型名稱！")
                return

        if not self.output_file.get():
            messagebox.showerror("錯誤", "請輸入輸出檔名！")
            return

        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        self.is_processing = True
        self.translate_page.process_button.config(state=tk.DISABLED)
        self.translate_page.progress.config(mode='indeterminate')
        self.translate_page.progress.start()

        thread = threading.Thread(target=self.process_thread, daemon=True)
        thread.start()

    def process_thread(self):
        try:
            success = process_with_ai(
                folder_path=self.folder_path.get(),
                ollama_url=self._get_ollama_url(),
                model_name=self.model_name.get(),
                output_file=self.output_file.get(),
                enable_classification=self.enable_classification.get(),
                translation_mode=self.translation_mode.get(),
                gemini_api_key=self.gemini_api_key.get(),
                log_callback=self.log,
                progress_callback=self.update_progress
            )

            if success:
                self.root.after(0, lambda: messagebox.showinfo("完成", f"處理完成！\n結果已儲存至：{self.output_file.get()}"))
            else:
                self.root.after(0, lambda: messagebox.showerror("錯誤", "處理失敗，請查看日誌。"))

        except Exception as e:
            self.log(f"❌ 發生錯誤: {e}")
            self.root.after(0, lambda: messagebox.showerror("錯誤", f"發生錯誤：{e}"))

        finally:
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.is_processing = False
        self.translate_page.process_button.config(state=tk.NORMAL)
        self.translate_page.progress.stop()
        self.translate_page.progress.config(mode='determinate', value=0, maximum=100)
