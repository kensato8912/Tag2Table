"""
介面模組：定義按鈕、分頁、群組包樣式
包含主視窗 TagProcessorGUI、Memo 轉換、題詞組合器
"""
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, simpledialog
from pathlib import Path
import threading

from database_manager import (
    load_config, save_config, load_tag_database, load_tag_map, load_prompt_presets,
    save_prompt_presets, load_tag_data_for_prompt, generate_folder_report, generate_file_report,
    DB_FILE, tag_map, TXT_OUTPUT_DIR,
)
from ollama_client import get_ollama_models, unload_model
from processor import process_with_ai


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
        preset_inner = ttk.Frame(preset_frame)
        preset_inner.pack(fill=tk.X, padx=5, pady=5)
        for name, tags in load_prompt_presets().items():
            ttk.Button(preset_inner, text=name, command=lambda t=tags: self._load_preset(t)).pack(side=tk.LEFT, padx=5, pady=3)

        self.notebook = ttk.Notebook(self.win)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10, pady=5)

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

        footer = ttk.LabelFrame(self.win, text="已選題詞", padding=8)
        footer.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=10, pady=8)
        footer.columnconfigure(0, weight=1)
        self.text_area = scrolledtext.ScrolledText(footer, height=3, font=("Consolas", 10), wrap=tk.WORD)
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        btn_row = ttk.Frame(footer)
        btn_row.grid(row=1, column=0)
        ttk.Button(btn_row, text="清空", command=self._clear_tags).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="複製提示詞", command=self._copy_result).pack(side=tk.LEFT, padx=5)

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


# --- 主視窗 ---

class TagProcessorGUI:
    """GUI 介面類"""

    def __init__(self, root):
        self.root = root
        self.root.title("Tag2Table — AI 標籤統計與翻譯工具")
        self.root.geometry("750x680")
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

        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        ttk.Label(main_frame, text="Tag2Table — AI 標籤統計與翻譯工具", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=(0, 20))

        ttk.Label(main_frame, text="訓練資料夾：").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.folder_path, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="瀏覽...", command=self.browse_folder).grid(row=1, column=2, pady=5)

        ttk.Label(main_frame, text="翻譯引擎：").grid(row=2, column=0, sticky=tk.W, pady=5)
        mode_frame = ttk.Frame(main_frame)
        mode_frame.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(mode_frame, text="雲端 Gemini（精度高，不佔顯存）",
                       variable=self.translation_mode, value="gemini",
                       command=self.toggle_translation_mode).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(mode_frame, text="本地 Ollama（完全隱私，需啟動 Ollama）",
                       variable=self.translation_mode, value="ollama",
                       command=self.toggle_translation_mode).pack(side=tk.LEFT)

        self.gemini_label = ttk.Label(main_frame, text="Gemini API Key：")
        self.gemini_label.grid(row=3, column=0, sticky=tk.W, pady=5)
        self.gemini_frame = ttk.Frame(main_frame)
        self.gemini_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.gemini_frame.columnconfigure(0, weight=1)
        self.gemini_entry = ttk.Entry(self.gemini_frame, textvariable=self.gemini_api_key, width=50, show="●")
        self.gemini_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(self.gemini_frame, text="儲存", command=self.save_config_clicked).grid(row=0, column=1, padx=5)

        self.ollama_frame = ttk.LabelFrame(main_frame, text="Ollama 設定")
        self.ollama_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        self.ollama_frame.columnconfigure(2, weight=1)
        host_frame = ttk.Frame(self.ollama_frame)
        host_frame.grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=3)
        ttk.Label(host_frame, text="連線目標：").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Radiobutton(host_frame, text="本地", variable=self.ollama_host_mode, value="local",
                        command=self._sync_ollama_mode).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(host_frame, text="遠端", variable=self.ollama_host_mode, value="remote",
                        command=self._sync_ollama_mode).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(host_frame, text="載入模型", command=self.load_models).pack(side=tk.LEFT, padx=2)
        ttk.Button(host_frame, text="卸載模型", command=self.unload_model_clicked).pack(side=tk.LEFT)
        self.ollama_remote_row = ttk.Frame(self.ollama_frame)
        self.ollama_remote_row.grid(row=1, column=0, columnspan=4, sticky=tk.W, padx=5, pady=3)
        ttk.Label(self.ollama_remote_row, text="遠端 IP：").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(self.ollama_remote_row, textvariable=self.ollama_remote_ip, width=20).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(self.ollama_remote_row, text="（同 Wi‑Fi 下 Mac/另一台電腦的 IP）").pack(side=tk.LEFT)
        ttk.Label(self.ollama_frame, text="模型：").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        model_frame = ttk.Frame(self.ollama_frame)
        model_frame.grid(row=2, column=1, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=3)
        model_frame.columnconfigure(0, weight=1)
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_name, width=45)
        self.model_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))

        ttk.Label(main_frame, text="存檔路徑：").grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_file, width=50).grid(row=5, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(main_frame, text="另存為...", command=self.browse_output_file).grid(row=5, column=2, pady=5)

        ttk.Checkbutton(main_frame, text="啟用分類輸出（依身體、衣服、姿態等分組）",
                       variable=self.enable_classification).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=7, column=0, columnspan=3, pady=20)
        self.process_button = ttk.Button(btn_frame, text="開始處理", command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="產生資料夾報告", command=self.generate_report_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="逐檔報告", command=self.generate_file_report_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Memo 轉換", command=self.open_memo_convert).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="題詞組合器", command=self.open_prompt_mixer).pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(main_frame, text="執行日誌：").grid(row=9, column=0, sticky=tk.W, pady=(10, 5))

        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(10, weight=1)
        self.toggle_translation_mode()

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80, wrap=tk.WORD, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def _get_ollama_url(self):
        if self.ollama_host_mode.get() == "remote":
            ip = self.ollama_remote_ip.get().strip()
            if ip:
                return f"http://{ip}:11434/v1"
            return ""
        return "http://localhost:11434/v1"

    def _sync_ollama_mode(self):
        is_remote = self.ollama_host_mode.get() == "remote"
        if is_remote:
            self.ollama_remote_row.grid()
            url = self._get_ollama_url()
            self.ollama_url.set(url)
        else:
            self.ollama_remote_row.grid_remove()
            self.ollama_url.set("http://localhost:11434/v1")

    def save_config_clicked(self):
        config = {
            "gemini_api_key": self.gemini_api_key.get(),
            "folder_path": self.folder_path.get(),
            "output_file": self.output_file.get(),
            "ollama_url": self._get_ollama_url() or self.ollama_url.get(),
            "ollama_host_mode": self.ollama_host_mode.get(),
            "ollama_remote_ip": self.ollama_remote_ip.get(),
        }
        save_config(config)
        messagebox.showinfo("完成", "設定已儲存")

    def _on_close(self):
        if self.translation_mode.get() == "ollama" and self.model_name.get():
            try:
                url = self._get_ollama_url()
                if url:
                    unload_model(url, self.model_name.get())
            except Exception:
                pass
        if self.gemini_api_key.get() or self.folder_path.get():
            url = self._get_ollama_url()
            config = {
                "gemini_api_key": self.gemini_api_key.get(),
                "folder_path": self.folder_path.get(),
                "output_file": self.output_file.get(),
                "ollama_url": url or self.ollama_url.get(),
                "ollama_host_mode": self.ollama_host_mode.get(),
                "ollama_remote_ip": self.ollama_remote_ip.get(),
            }
            save_config(config)
        self.root.destroy()

    def toggle_translation_mode(self):
        is_ollama = self.translation_mode.get() == "ollama"
        if is_ollama:
            self.gemini_label.grid_remove()
            self.gemini_frame.grid_remove()
        else:
            self.gemini_label.grid(row=3, column=0, sticky=tk.W, pady=5)
            self.gemini_frame.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        if is_ollama:
            self.ollama_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        else:
            self.ollama_frame.grid_remove()
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
        from database_manager import _ensure_txt_dir
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
            self.model_combo['values'] = models
            if models and not self.model_name.get():
                self.model_name.set(models[0])
            self.log(f"✅ 已載入 {len(models)} 個模型")
            messagebox.showinfo("完成", f"已載入 {len(models)} 個模型，請從下拉選單選擇")
        else:
            self.log("❌ 未找到任何模型，請確認 Ollama 已啟動且已 pull 模型")
            messagebox.showwarning("提示", "未找到模型\n請確認 Ollama 服務已啟動\n並執行 ollama pull <模型名> 下載模型")

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def update_progress(self, value, max_value):
        if max_value > 0:
            self.progress.config(mode='determinate', maximum=max_value, value=value)
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
        self.process_button.config(state=tk.DISABLED)
        self.progress.config(mode='indeterminate')
        self.progress.start()

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
        self.process_button.config(state=tk.NORMAL)
        self.progress.stop()
        self.progress.config(mode='determinate', value=0, maximum=100)
