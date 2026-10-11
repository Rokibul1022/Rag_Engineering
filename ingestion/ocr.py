"""OCR with Tesseract (via pytesseract). Everything degrades gracefully if Tesseract is missing."""
from __future__ import annotations

import os
import shutil
import statistics

from .models import Block, warn

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover
    pytesseract = None

_WINDOWS_DEFAULT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_status() -> dict:
    """Is OCR usable right now? Also lists installed language packs (eng, ben, ...)."""
    if pytesseract is None:
        return {"available": False, "languages": [], "reason": "Python package 'pytesseract' is not installed (pip install pytesseract)."}
    if not shutil.which("tesseract"):
        if os.name == "nt" and os.path.exists(_WINDOWS_DEFAULT):
            pytesseract.pytesseract.tesseract_cmd = _WINDOWS_DEFAULT
        else:
            return {"available": False, "languages": [],
                    "reason": "Tesseract program not found. Install it (Windows: github.com/UB-Mannheim/tesseract/wiki, "
                              "Ubuntu: sudo apt install tesseract-ocr, macOS: brew install tesseract)."}
    try:
        langs = sorted(l for l in pytesseract.get_languages(config="") if l != "osd")
    except Exception:
        langs = ["eng"]
    return {"available": True, "languages": langs, "reason": ""}


def preprocess(img):
    """Cheap image clean-up that helps Tesseract: grayscale, upscale small images, boost contrast."""
    from PIL import Image, ImageOps
    steps = []
    img = ImageOps.exif_transpose(img)
    g = ImageOps.grayscale(img)
    steps.append("grayscale")
    if g.width < 1200:
        scale = 1200 / g.width
        g = g.resize((int(g.width * scale), int(g.height * scale)), Image.LANCZOS)
        steps.append(f"upscale x{scale:.1f}")
    g = ImageOps.autocontrast(g)
    steps.append("autocontrast")
    return g, steps


def ocr_image(img, lang: str = "eng", page: int | None = None):
    """Run OCR on a PIL image.

    Returns (blocks, raw_text, warnings, stats). Each block keeps its OCR confidence in meta,
    because low confidence is a signal you may want to store and filter on later.
    """
    g, steps = preprocess(img)
    data = pytesseract.image_to_data(g, lang=lang, output_type=Output.DICT)
    paras: dict = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        conf = float(data["conf"][i])
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        paras.setdefault(key, {}).setdefault(data["line_num"][i], []).append((word, conf, data["height"][i]))

    all_heights = [h for p in paras.values() for ln in p.values() for (_, _, h) in ln]
    body_h = statistics.median(all_heights) if all_heights else 0
    blocks, texts, all_conf, low_words = [], [], [], []
    for lines in paras.values():
        words = [w for ln in lines.values() for w in ln]
        text = "\n".join(" ".join(w for w, _, _ in ln) for ln in lines.values())
        confs = [c for _, c, _ in words]
        all_conf += confs
        low = [w for w, c, _ in words if c < 60]
        low_words += low
        h = statistics.median([hh for _, _, hh in words])
        is_heading = body_h and h >= 1.35 * body_h and len(words) <= 12
        blocks.append(Block("heading" if is_heading else "paragraph", text, page=page,
                            level=1 if is_heading else 0,
                            meta={"ocr": True, "confidence": round(sum(confs) / len(confs), 1),
                                  "low_conf_words": low[:15]}))
        texts.append(text)

    mean_conf = round(sum(all_conf) / len(all_conf), 1) if all_conf else 0.0
    warnings = [warn("info", "OCR_PREPROCESS", "Image preprocessing: " + ", ".join(steps), page)]
    if not all_conf:
        warnings.append(warn("warning", "OCR_NO_TEXT", "OCR found no readable text.", page))
    elif mean_conf < 70:
        warnings.append(warn("warning", "OCR_LOW_CONF", f"Mean OCR confidence is only {mean_conf}%. Expect errors; consider a better scan.", page))
    stats = {"words": len(all_conf), "mean_confidence": mean_conf, "low_conf_words": len(low_words)}
    return blocks, "\n\n".join(texts), warnings, stats
