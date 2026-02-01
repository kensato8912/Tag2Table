"""智慧裁切分頁：標籤模式 / 視覺主動偵測 (MediaPipe)"""
import os
from pathlib import Path

import gradio as gr

try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from . import shared
from src.cropper import run_crop
from src.auto_cropper import run_smart_crop_active, load_crop_parts, SUPPORTED_DETECTORS
from src.crop_engine import (
    load_crop_parts as load_engine_parts,
    run_crop_batch,
    run_calibration,
    run_multi_scale_test,
    run_detection_demo,
    run_precise_local_detect,
    run_crop_zoom_detect,
    process_roi,
    save_to_library,
    SUPPORTED_DETECTORS as ENGINE_DETECTORS,
)

_PROJECT_ROOT = Path(__file__).parent.parent
# 輸出路徑鎖定在專案底下，避免噴到 StabilityMatrix 根目錄
DEFAULT_CROP_DEST = (_PROJECT_ROOT / "data" / "crop_output").resolve()


def _run_visual_active_stream(source, dest, size_val, padding_pct, min_size_val, target_part, full_auto,
                              manual_padding=None, manual_y_offset=None, manual_confidence=None):
    """視覺主動偵測：參數驅動型 CropEngine 或舊版 run_smart_crop_active"""
    if not source or not str(source).strip():
        yield "❌ 請先選擇來源（資料夾或 .zip 壓縮檔）", []
        return

    dest = dest or str(DEFAULT_CROP_DEST)
    size_map = {"512×512": 512, "768×768": 768}
    size = size_map.get(str(size_val), 512)
    try:
        min_size = max(256, int(min_size_val)) if min_size_val not in (None, "") else 512
    except (TypeError, ValueError):
        min_size = 512
    target = str(target_part or "").strip()
    full = bool(full_auto)

    cfg = load_engine_parts()
    parts_list = [p for p in cfg.get("parts", []) if p.get("detector") in ENGINE_DETECTORS]
    part_by_id = {p["id"]: p for p in parts_list}
    use_engine = (target in part_by_id) or (full and parts_list)

    if use_engine:
        pad_override = float(manual_padding if manual_padding is not None else 20) / 100.0
        yoff_override = float(manual_y_offset if manual_y_offset is not None else 0) / 100.0
        conf_override = float(manual_confidence if manual_confidence is not None else 50) / 100.0
        to_run = [part_by_id[target]] if not full else parts_list
        log_lines = []
        all_previews = []
        for part_config in to_run:
            cfg = dict(part_config)
            cfg["padding"] = pad_override
            cfg["y_offset"] = yoff_override
            for log_line, previews, cur, tot in run_crop_batch(
                source, dest, cfg, crop_size=size, min_resolution=min_size,
                manual_confidence=conf_override
            ):
                log_lines.append(log_line)
                all_previews = previews
                yield "\n".join(log_lines), all_previews
    else:
        if Path(source).suffix.lower() == ".zip":
            yield "❌ ZIP 來源僅支援參數驅動模式，請選擇「手部 / 手指 / 腳踝到腳尖 / 精緻五官」或勾選全自動掃描", []
            return
        padding = max(0, min(1, float(padding_pct or 30) / 100))
        log_lines = []
        for log_line, previews in run_smart_crop_active(
            source, dest, crop_size=size, padding=padding, min_resolution=min_size,
            recursive=True, target_part=target, full_auto=full
        ):
            log_lines.append(log_line)
            yield "\n".join(log_lines), previews


def _run_crop_stream(source, dest, size_val, triggers_raw, target_part, padding_pct, min_size_val, strip_trigger):
    """執行裁切，串流日誌並回傳預覽圖。yield (log_text, preview_list)。"""
    if not source or not str(source).strip():
        yield "❌ 請先選擇來源（資料夾或 .zip 壓縮檔）", []
        return
    if Path(source).suffix.lower() == ".zip":
        yield "❌ 標籤模式不支援 ZIP，請改用視覺主動偵測並選擇目標部位", []
        return

    dest = dest or str(DEFAULT_CROP_DEST)
    triggers = [t.strip() for t in (triggers_raw or "").split(",") if t.strip()] or ["Niyaniya", "Ibuki"]
    size_map = {"512×512": 512, "768×768": 768}
    size = size_map.get(str(size_val), 512)
    try:
        min_size = max(256, int(min_size_val)) if min_size_val not in (None, "") else 512
    except (TypeError, ValueError):
        min_size = 512
    padding = max(0, min(100, float(padding_pct))) if padding_pct is not None else 30
    use_mediapipe_hands = "手部" in str(target_part)

    previews = []

    def producer(log_cb):
        nonlocal previews
        cnt, pr = run_crop(
            source,
            dest,
            crop_size=size,
            min_resolution=min_size,
            padding_pct=padding,
            trigger_words=triggers if strip_trigger else [],
            use_mediapipe_hands=use_mediapipe_hands,
            use_edge_detection=not use_mediapipe_hands,
            target_part=str(target_part) if target_part else None,
            log_callback=log_cb,
            preview_count=8,
        )
        previews[:] = [(str(p), label) for p, label in pr]

    for text in shared.stream_from_log_callback(producer):
        yield text, previews


def _run_log_only(src, dst, size_label, triggers, target, pad, minsz, strip, mode, full_auto,
                  manual_pad=None, manual_yoff=None, manual_conf=None):
    """輸出到日誌；視覺模式每張圖即時回報"""
    if mode and "視覺主動" in str(mode):
        for log_text, _ in _run_visual_active_stream(
            src, dst, size_label, pad, minsz, target, full_auto,
            manual_pad, manual_yoff, manual_conf
        ):
            yield log_text
    else:
        for log_text, _ in _run_crop_stream(src, dst, size_label, triggers, target, pad, minsz, strip):
            yield log_text


def _run_with_gallery(src, dst, size_label, triggers, target, pad, minsz, strip, mode, full_auto,
                     manual_pad=None, manual_yoff=None, manual_conf=None):
    """雙輸出：日誌 + Gallery 預覽"""
    if mode and "視覺主動" in str(mode):
        for log_text, previews in _run_visual_active_stream(
            src, dst, size_label, pad, minsz, target, full_auto,
            manual_pad, manual_yoff, manual_conf
        ):
            yield log_text, previews
    else:
        for log_text, previews in _run_crop_stream(src, dst, size_label, triggers, target, pad, minsz, strip):
            yield log_text, previews


def _calibration_preview(img, part, pad, yoff, conf):
    """校正預覽：回傳 (標註後的圖片, 狀態訊息)。"""
    if img is None:
        return None, ""
    pad_pct = float(pad or 20) / 100.0
    yoff_val = float(yoff or 0) / 100.0
    conf_val = float(conf or 50) / 100.0
    return run_calibration(img, str(part or "feet"), pad_pct, yoff_val, conf_val)


def render(defaults: dict, train_comps: dict):
    """建立智慧裁切分頁 (MediaPipe Powered)。"""
    with gr.Row():
        with gr.Column(scale=2):
            preview_canvas = gr.Image(
                label="校正預覽 (Landmarks=綠點, 裁切框=紅框)",
                type="numpy",
                height=360,
            )
            calibration_status = gr.Textbox(
                label="偵測狀態",
                value="",
                interactive=False,
            )
        with gr.Column(scale=1):
            calibration_test_img = gr.Image(
                label="測試圖（校正用）",
                type="numpy",
                height=160,
            )
            manual_padding = gr.Slider(
                minimum=0, maximum=100, value=20, step=1,
                label="Manual Padding (%)",
            )
            manual_y_offset = gr.Slider(
                minimum=-50, maximum=50, value=0, step=1,
                label="Manual Y-Offset",
            )
            manual_confidence = gr.Slider(
                minimum=1, maximum=100, value=50, step=1,
                label="Manual Confidence (%)",
            )
            cal_btn = gr.Button("校正預覽", variant="secondary")
            multi_scale_btn = gr.Button("Test Multi-Scale Detection", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ✂️ 智慧肢體收割機 (MediaPipe Powered)\n**視覺主動偵測**：依目標部位或「全自動掃描」同時偵測手、腳、臉並裁切，自動寫入 hand_focus/feet_focus/face_focus 標籤。\n⬇ 執行結果顯示於頁面下方「執行日誌」")

            mode_radio = gr.Radio(
                choices=["視覺主動偵測 (推薦)", "標籤模式 (依部位)"],
                value="視覺主動偵測 (推薦)",
                label="裁切模式",
            )

            with gr.Row():
                source_dir = gr.Textbox(
                    label="高品質素材來源（資料夾含 JPG/PNG/ZIP）",
                    value=defaults.get("cropper_source", defaults.get("train_data_dir", "")),
                    placeholder="E:/AI_Training/raw_manga（可含 .zip）",
                    scale=7,
                )
                gr.Button("資料夾", scale=1).click(
                    fn=lambda x: shared.browse_folder(x),
                    inputs=[source_dir],
                    outputs=[source_dir],
                )
                gr.Button("ZIP", scale=1).click(
                    fn=lambda x: shared.browse_zip(x),
                    inputs=[source_dir],
                    outputs=[source_dir],
                )

            def _reload_parts_dropdown():
                cfg = load_engine_parts()
                choices = [(p.get("label", p["id"]), p["id"]) for p in cfg.get("parts", []) if p.get("detector") in ENGINE_DETECTORS]
                if not choices:
                    choices = [("手部 (Full Hand)", "hand"), ("腳踝到腳尖", "feet"), ("精緻五官", "face")]
                return gr.update(choices=choices, value=choices[0][1])

            _parts_cfg = load_engine_parts()
            _part_choices = [(p.get("label", p["id"]), p["id"]) for p in _parts_cfg.get("parts", []) if p.get("detector") in ENGINE_DETECTORS]
            if not _part_choices:
                _part_choices = [("手部 (Hands)", "hand"), ("腳部 (Feet)", "feet"), ("臉部 (Face)", "face")]
            with gr.Row():
                target_part = gr.Dropdown(
                    choices=_part_choices,
                    value=_part_choices[0][1] if _part_choices else "hand",
                    label="目標部位（非全自動時使用）",
                    scale=9,
                )
                reload_parts_btn = gr.Button("重讀", scale=1)
            reload_parts_btn.click(fn=_reload_parts_dropdown, outputs=[target_part])
            full_auto_check = gr.Checkbox(
                label="全自動掃描",
                value=False,
            )

            with gr.Row():
                dest_dir = gr.Textbox(
                    label="輸出資料夾",
                    value=defaults.get("cropper_dest", str(DEFAULT_CROP_DEST)),
                    placeholder="裁切結果輸出位置",
                    scale=9,
                )
                gr.Button("瀏覽...", scale=1).click(
                    fn=lambda x: shared.browse_folder(x),
                    inputs=[dest_dir],
                    outputs=[dest_dir],
                )

            with gr.Row():
                padding = gr.Slider(
                    minimum=0, maximum=100, value=30,
                    label="邊緣擴張 (%) — 視覺模式：0.3=30%",
                )
                min_size = gr.Number(label="最小解析度要求", value=512)
                size_dropdown = gr.Dropdown(
                    label="裁切大小（長腿模式固定 512×768）",
                    choices=["512×512", "768×768", "512×768"],
                    value="512×512",
                )

            strip_trigger = gr.Checkbox(
                label="自動清理標籤 (去除角色觸發詞)",
                value=True,
            )
            trigger_input = gr.Textbox(
                label="觸發詞排除",
                value=defaults.get("helper_trigger_words", "Niyaniya, Ibuki"),
                placeholder="從標籤中移除的角色觸發詞（逗號分隔）",
            )
            crop_btn = gr.Button("🚀 啟動 AI 智慧裁切", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 最新裁切成果")
            preview_gallery = gr.Gallery(
                label="裁切成果",
                columns=5,
                height="400px",
                object_fit="contain",
            )

    _cal_inputs = [calibration_test_img, target_part, manual_padding, manual_y_offset, manual_confidence]
    _cal_outputs = [preview_canvas, calibration_status]

    def _multi_scale_test(img, part):
        if img is None:
            return None, ""
        return run_multi_scale_test(img, str(part or "feet"))

    multi_scale_btn.click(
        fn=_multi_scale_test,
        inputs=[calibration_test_img, target_part],
        outputs=[preview_canvas, calibration_status],
    )

    def _cal_with_status(img, part, pad, yoff, conf):
        out_img, status = _calibration_preview(img, part, pad, yoff, conf)
        return out_img, status or ""

    calibration_test_img.change(fn=_cal_with_status, inputs=_cal_inputs, outputs=_cal_outputs)
    target_part.change(fn=_cal_with_status, inputs=_cal_inputs, outputs=_cal_outputs)
    manual_padding.change(fn=_cal_with_status, inputs=_cal_inputs, outputs=_cal_outputs)
    manual_y_offset.change(fn=_cal_with_status, inputs=_cal_inputs, outputs=_cal_outputs)
    manual_confidence.change(fn=_cal_with_status, inputs=_cal_inputs, outputs=_cal_outputs)
    cal_btn.click(fn=_cal_with_status, inputs=_cal_inputs, outputs=_cal_outputs)

    bindings = [
        (crop_btn, _run_with_gallery, [
            source_dir, dest_dir, size_dropdown, trigger_input,
            target_part, padding, min_size, strip_trigger,
            mode_radio, full_auto_check,
            manual_padding, manual_y_offset, manual_confidence,
        ], preview_gallery),
    ]

    comps = {
        "cropper_source": source_dir,
        "cropper_dest": dest_dir,
        "cropper_preview": preview_gallery,
        "cropper_preview_canvas": preview_canvas,
    }
    return comps, bindings


def render_detection_demo_tab():
    """偵測驗證與視覺化 Demo：不存檔、不裁切，純預覽與參數校正。"""
    gr.Markdown("### 偵測驗證與視覺化\n上傳圖片後**點擊兩次**框選區域（第一次=左上角、第二次=右下角），選取後自動畫藍框並執行 AI 偵測。或使用下方按鈕執行偵測。")
    coord_list_state = gr.State(value=[])
    with gr.Row():
        demo_input = gr.Image(
            label="原始圖（點兩次框選區域：左上→右下）",
            type="numpy",
            height=400,
            interactive=True,
        )
        demo_output = gr.Image(label="偵測結果（藍框=選取區域，紅框=AI 修正）", type="numpy", height=400)
    coord_display = gr.Textbox(label="選取座標 (點兩次後顯示)", value="[]", interactive=False, visible=True)
    with gr.Accordion("選取區域（選填，精準偵測用）— 輸入 x1,y1,x2,y2 或點「填入整張圖」", open=False):
        with gr.Row():
            crop_x1 = gr.Number(label="x1", value=0, precision=0)
            crop_y1 = gr.Number(label="y1", value=0, precision=0)
            crop_x2 = gr.Number(label="x2", value=0, precision=0)
            crop_y2 = gr.Number(label="y2", value=0, precision=0)
        with gr.Row():
            btn_full = gr.Button("填入整張圖")
            btn_clear_select = gr.Button("清除選取")
    with gr.Row():
        detector_dropdown = gr.Dropdown(
            choices=["Face", "Hands", "Pose"],
            value="Face",
            label="偵測器",
        )
        confidence_slider = gr.Slider(
            minimum=0.1, maximum=0.5, value=0.3, step=0.05,
            label="Confidence",
        )
        resize_1024_check = gr.Checkbox(
            label="啟用 1024px 預縮放",
            value=True,
        )
        demo_btn = gr.Button("執行偵測", variant="primary")
        precise_btn = gr.Button("針對選取區域進行精準偵測", variant="secondary")

    def _run_demo(img, det, conf, resize_1024):
        if img is None:
            return None
        return run_detection_demo(img, det, conf, resize_1024)

    def _on_select(evt: gr.SelectData, img, coords, det, conf):
        if img is None:
            return None, [], "[]"
        try:
            x, y = int(evt.index[0]), int(evt.index[1])
        except (TypeError, IndexError, ValueError):
            return None, coords or [], str(coords or [])
        lst = list(coords or [])
        lst.extend([x, y])
        while len(lst) > 4:
            lst.pop(0)
            lst.pop(0)
        if len(lst) < 4:
            return None, lst, str(lst)
        x1, y1, x2, y2 = lst[0], lst[1], lst[2], lst[3]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        box = (x1, y1, x2, y2)
        annotated = run_precise_local_detect(img, det, conf, box)
        return annotated, lst, str(lst)

    def _clear_select():
        return [], 0, 0, 0, 0, "[]"

    def _run_precise(img, det, conf, cx1, cy1, cx2, cy2):
        if img is None:
            return None
        box = None
        try:
            v1, v2, v3, v4 = float(cx1 or 0), float(cy1 or 0), float(cx2 or 0), float(cy2 or 0)
            if v3 > v1 and v4 > v2:
                box = (int(v1), int(v2), int(v3), int(v4))
        except (TypeError, ValueError):
            pass
        return run_precise_local_detect(img, det, conf, box)

    def _fill_full(img):
        if img is None:
            return 0, 0, 0, 0
        h, w = img.shape[:2]
        return 0, 0, w, h

    demo_btn.click(
        fn=_run_demo,
        inputs=[demo_input, detector_dropdown, confidence_slider, resize_1024_check],
        outputs=[demo_output],
    )
    precise_btn.click(
        fn=_run_precise,
        inputs=[demo_input, detector_dropdown, confidence_slider, crop_x1, crop_y1, crop_x2, crop_y2],
        outputs=[demo_output],
    )
    btn_full.click(
        fn=_fill_full,
        inputs=[demo_input],
        outputs=[crop_x1, crop_y1, crop_x2, crop_y2],
    )
    btn_clear_select.click(
        fn=_clear_select,
        inputs=[],
        outputs=[coord_list_state, crop_x1, crop_y1, crop_x2, crop_y2, coord_display],
    )
    demo_input.select(
        fn=_on_select,
        inputs=[demo_input, coord_list_state, detector_dropdown, confidence_slider],
        outputs=[demo_output, coord_list_state, coord_display],
    )

    comps = {"demo_input": demo_input, "demo_output": demo_output}
    return comps, []


def render_crop_lib_demo_tab():
    """手動框選 + AI 修正 + 分類存檔：tool=select 獲取座標，藍色手動區→process_roi→紅色 AI 框，側邊即時預覽。"""
    gr.Markdown("### 手動框選 + AI 修正 + 分類存檔\n放棄彈窗裁切，改用 **點擊兩次** 框選區域（左上→右下）。座標為藍色手動區，立即執行 AI 偵測並畫紅色修正框。右側即時顯示裁切預覽，確認後存檔。")
    crop_state = gr.State(value=None)
    coord_list_state = gr.State(value=[])
    image_list_state = gr.State(value=[])
    current_index_state = gr.State(value=0)
    with gr.Row():
        lib_folder = gr.Textbox(label="圖片資料夾", value="", placeholder="選擇包含圖片的資料夾路徑", scale=9)
        btn_browse = gr.Button("瀏覽", scale=1)
    with gr.Row():
        btn_load_folder = gr.Button("載入資料夾")
        btn_prev = gr.Button("⬅ 上一張")
        btn_next = gr.Button("下一張 ➡")
        lib_counter = gr.Textbox(label="張數", value="0 / 0", interactive=False, scale=1)
    with gr.Row():
        lib_input = gr.Image(
            label="原始圖（點兩次框選：左上→右下）",
            type="numpy",
            height=400,
            interactive=True,
        )
        with gr.Column(scale=1):
            lib_output = gr.Image(label="偵測結果（藍框=手動，紅框=AI）", type="numpy", height=340)
            lib_preview = gr.Image(
                label="裁切預覽（存檔前確認品質）",
                type="numpy",
                height=200,
            )
    with gr.Row():
        lib_detector = gr.Dropdown(choices=["Face", "Hands", "Pose"], value="Face", label="偵測器")
        lib_confidence = gr.Slider(minimum=0.1, maximum=0.5, value=0.3, step=0.05, label="Confidence")
    with gr.Accordion("手動輸入座標（若點選無效）", open=False):
        with gr.Row():
            lib_x1 = gr.Number(label="x1", value=0, precision=0)
            lib_y1 = gr.Number(label="y1", value=0, precision=0)
            lib_x2 = gr.Number(label="x2", value=0, precision=0)
            lib_y2 = gr.Number(label="y2", value=0, precision=0)
        btn_manual = gr.Button("執行 process_roi")
    with gr.Row():
        btn_save_hand = gr.Button("💾 存至手部庫", variant="secondary")
        btn_save_feet = gr.Button("💾 存至腳部庫", variant="secondary")
        btn_save_face = gr.Button("💾 存至臉部庫", variant="secondary")
    status_text = gr.Textbox(label="偵測/存檔狀態", interactive=False, lines=2)

    def _on_select(evt: gr.SelectData, img, coords, det, conf):
        if img is None:
            return None, None, None, [], "請上傳圖片"
        try:
            x, y = int(evt.index[0]), int(evt.index[1])
        except (TypeError, IndexError, ValueError):
            return None, None, None, coords or [], "選取座標取得失敗"
        lst = list(coords or [])
        lst.extend([x, y])
        while len(lst) > 4:
            lst.pop(0)
            lst.pop(0)
        if len(lst) < 4:
            return None, None, None, lst, "請點第二次完成框選"
        x1, y1, x2, y2 = lst[0], lst[1], lst[2], lst[3]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        box = (x1, y1, x2, y2)
        annotated, preview, raw, msg = process_roi(img, box, det, conf)
        return annotated, preview, raw, lst, msg

    def _load_image(path: str):
        if not _HAS_CV2 or not path or not os.path.isfile(path):
            return None
        try:
            img = cv2.imread(path)
            if img is None:
                return None
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

    def _load_folder(folder_path):
        if not folder_path or not os.path.isdir(folder_path):
            return None, [], 0, None, None, None, [], "請選擇有效資料夾", "0 / 0"
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = []
        for f in sorted(os.listdir(folder_path)):
            if Path(f).suffix.lower() in exts:
                paths.append(os.path.join(folder_path, f))
        if not paths:
            return None, [], 0, None, None, None, [], "資料夾內無支援圖片 (jpg/png/webp)", "0 / 0"
        img = _load_image(paths[0])
        return img, paths, 0, None, None, None, [], f"已載入 {len(paths)} 張", f"1 / {len(paths)}"

    def _go_prev(img_list, idx):
        if not img_list:
            return None, 0, None, None, None, [], "", "0 / 0"
        new_idx = max(0, idx - 1)
        img = _load_image(img_list[new_idx])
        return img, new_idx, None, None, None, [], "", f"{new_idx + 1} / {len(img_list)}"

    def _go_next(img_list, idx):
        if not img_list:
            return None, 0, None, None, None, [], "", "0 / 0"
        new_idx = min(len(img_list) - 1, idx + 1)
        img = _load_image(img_list[new_idx])
        return img, new_idx, None, None, None, [], "", f"{new_idx + 1} / {len(img_list)}"

    def _save(lib_type, crop):
        return save_to_library(crop, lib_type)

    def _run_manual(img, det, conf, cx1, cy1, cx2, cy2):
        if img is None:
            return None, None, None, "請上傳圖片"
        try:
            x1, y1, x2, y2 = int(cx1 or 0), int(cy1 or 0), int(cx2 or 0), int(cy2 or 0)
            if x2 <= x1 or y2 <= y1:
                return None, None, None, "請輸入有效座標 (x2>x1, y2>y1)"
        except (TypeError, ValueError):
            return None, None, None, "座標格式錯誤"
        box = (x1, y1, x2, y2)
        annotated, preview, raw, msg = process_roi(img, box, det, conf)
        return annotated, preview, raw, msg

    btn_browse.click(fn=lambda x: shared.browse_folder(x), inputs=[lib_folder], outputs=[lib_folder])
    btn_load_folder.click(
        fn=_load_folder,
        inputs=[lib_folder],
        outputs=[lib_input, image_list_state, current_index_state, lib_output, lib_preview, crop_state, coord_list_state, status_text, lib_counter],
    )
    btn_prev.click(
        fn=_go_prev,
        inputs=[image_list_state, current_index_state],
        outputs=[lib_input, current_index_state, lib_output, lib_preview, crop_state, coord_list_state, status_text, lib_counter],
    )
    btn_next.click(
        fn=_go_next,
        inputs=[image_list_state, current_index_state],
        outputs=[lib_input, current_index_state, lib_output, lib_preview, crop_state, coord_list_state, status_text, lib_counter],
    )
    lib_input.select(
        fn=_on_select,
        inputs=[lib_input, coord_list_state, lib_detector, lib_confidence],
        outputs=[lib_output, lib_preview, crop_state, coord_list_state, status_text],
    )
    btn_manual.click(
        fn=_run_manual,
        inputs=[lib_input, lib_detector, lib_confidence, lib_x1, lib_y1, lib_x2, lib_y2],
        outputs=[lib_output, lib_preview, crop_state, status_text],
    )
    btn_save_hand.click(fn=lambda c: _save("hand", c), inputs=[crop_state], outputs=[status_text])
    btn_save_feet.click(fn=lambda c: _save("feet", c), inputs=[crop_state], outputs=[status_text])
    btn_save_face.click(fn=lambda c: _save("face", c), inputs=[crop_state], outputs=[status_text])

    comps = {"crop_editor": lib_input, "crop_output": lib_output, "crop_preview": lib_preview, "crop_state": crop_state}
    return comps, []
