"""
DreamFerret Meme Lab — Flask Blueprint
Hidden tool at /go. Not linked from main site.

Generates 3 meme variants from an uploaded image:
  1) Faithful rebuild
  2) Structural reframe
  3) Anthropomorphic animals (symbolic)

Uses OCR to extract text + region buckets from source, then validates
generated output for exact text match and correct region placement.
"""

import io
import json
import os
import re
import secrets
from datetime import datetime

from flask import Blueprint, Response, jsonify, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# OCR backend — try pytesseract first, fall back to easyocr
# ---------------------------------------------------------------------------
OCR_BACKEND = None
_easyocr_reader = None

try:
    import pytesseract
    from pytesseract import Output
    OCR_BACKEND = "pytesseract"
except ImportError:
    pass

if OCR_BACKEND is None:
    try:
        import easyocr
        OCR_BACKEND = "easyocr"
    except ImportError:
        pass

if OCR_BACKEND is None:
    print("WARNING: No OCR backend available. Install pytesseract or easyocr.", flush=True)


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


# ---------------------------------------------------------------------------
# Blueprint + config
# ---------------------------------------------------------------------------
meme_lab_bp = Blueprint(
    "meme_lab",
    __name__,
    static_folder="meme_lab_static",
    static_url_path="/go/static",
)

@meme_lab_bp.errorhandler(Exception)
def _handle_meme_error(exc):
    """Ensure API errors always return JSON, not HTML."""
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")
OPENAI_IMAGE_SIZE = os.environ.get("OPENAI_IMAGE_SIZE", "1024x1024")
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "3"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "4"))

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
IMG_DIR = os.path.join(DATA_ROOT, "meme_images")
CONTAINER_DIR = os.path.join(DATA_ROOT, "meme_containers")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(CONTAINER_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Password gate
# ---------------------------------------------------------------------------
def _check_password():
    if not APP_PASSWORD:
        return True  # dev convenience
    return request.headers.get("x-df-pass") == APP_PASSWORD


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@meme_lab_bp.route("/go")
def go_page():
    """Serve the Meme Lab UI. Hidden, noindex."""
    html_path = os.path.join(os.path.dirname(__file__), "meme_lab_static", "go.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    resp = Response(content, mimetype="text/html")
    resp.headers["X-Robots-Tag"] = "noindex, nofollow"
    return resp


@meme_lab_bp.route("/api/meme/container/create", methods=["POST"])
def create_container():
    try:
        if not _check_password():
            return jsonify({"error": "Unauthorized"}), 401

        f = request.files.get("image")
        if not f:
            return jsonify({"error": "Missing image"}), 400

        container_id = "ctr_" + secrets.token_hex(10)

        # Read + square to 1024x1024
        img_bytes = f.read()
        print(f"[MemeLab] Read {len(img_bytes)} bytes from upload", flush=True)
        squared = _to_square_png(img_bytes, 1024)
        print(f"[MemeLab] Squared image: {len(squared)} bytes", flush=True)

        root_filename = f"{container_id}_root.png"
        root_path = os.path.join(IMG_DIR, root_filename)
        with open(root_path, "wb") as out:
            out.write(squared)
        root_image_rel = f"/data/meme_images/{root_filename}"
        print(f"[MemeLab] Saved root image to {root_path}", flush=True)

        # OCR
        print(f"[MemeLab] Running OCR with backend: {OCR_BACKEND}", flush=True)
        text_blocks = _extract_text_blocks(squared)
        print(f"[MemeLab] OCR found {len(text_blocks)} text blocks: "
              f"{[b['text'] for b in text_blocks]}", flush=True)

        if not text_blocks:
            return jsonify({
                "error": "No text detected in source image. "
                         "Meme Lab requires readable text in the image."
            }), 400

        container = {
            "id": container_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "text_blocks": text_blocks,
            "species_map": None,
            "nodes": [
                {
                    "node_id": "root",
                    "parent_node_id": None,
                    "depth": 0,
                    "image_rel": root_image_rel,
                    "variant": "root",
                }
            ],
        }
        _save_container(container)
        return jsonify({"container": container})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Container creation failed: {type(exc).__name__}: {exc}"}), 500


@meme_lab_bp.route("/api/meme/generate", methods=["POST"])
def generate():
    try:
        if not _check_password():
            return jsonify({"error": "Unauthorized"}), 401

        body = request.get_json(silent=True) or {}
        container_id = body.get("containerId")
        parent_node_id = body.get("parentNodeId")

        if not container_id or not parent_node_id:
            return jsonify({"error": "Missing containerId or parentNodeId"}), 400

        container = _load_container(container_id)
        if not container:
            return jsonify({"error": "Container not found"}), 404

        parent = None
        for n in container.get("nodes", []):
            if n["node_id"] == parent_node_id:
                parent = n
                break
        if not parent:
            return jsonify({"error": "Parent node not found"}), 404

        if parent["depth"] >= MAX_DEPTH:
            return jsonify({"error": f"Max depth {MAX_DEPTH} reached"}), 400

        prompts = _build_three_prompts(container)
        variant_names = ["rebuild", "reframe", "anthro"]

        # Generate all 3 variants in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _gen_one(vname, prompt_text):
            print(f"[MemeLab] Generating variant {vname} (parallel)…", flush=True)
            return vname, _generate_with_retries(
                prompt_text, container, vname, MAX_RETRIES
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_gen_one, vname, prompt): vname
                for vname, prompt in zip(variant_names, prompts)
            }
            parallel_results = {}
            for future in as_completed(futures):
                vname, img_result = future.result()
                parallel_results[vname] = img_result

        # Assemble results in order, attach node IDs
        results = []
        for vname in variant_names:
            img_result = parallel_results[vname]

            if not img_result.get("error"):
                node_id = secrets.token_hex(8)
                depth = parent["depth"] + 1
                container["nodes"].append({
                    "node_id": node_id,
                    "parent_node_id": parent_node_id,
                    "depth": depth,
                    "image_rel": img_result["url"],
                    "variant": vname,
                })
                img_result["node_id"] = node_id
                img_result["depth"] = depth

            results.append(img_result)

        _save_container(container)
        return jsonify({"container": container, "images": results})

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Generation failed: {type(exc).__name__}: {exc}"}), 500


@meme_lab_bp.route("/api/meme/diag")
def meme_diag():
    """Quick diagnostic — check OCR backend + dirs. Remove after debugging."""
    info = {
        "ocr_backend": OCR_BACKEND,
        "img_dir_exists": os.path.isdir(IMG_DIR),
        "container_dir_exists": os.path.isdir(CONTAINER_DIR),
        "openai_key_set": bool(OPENAI_API_KEY),
        "app_password_set": bool(APP_PASSWORD),
    }
    if OCR_BACKEND == "pytesseract":
        try:
            ver = pytesseract.get_tesseract_version()
            info["tesseract_version"] = str(ver)
        except Exception as e:
            info["tesseract_error"] = str(e)
    return jsonify(info)


@meme_lab_bp.route("/api/meme/test-ocr", methods=["POST"])
def test_ocr():
    """Debug endpoint: accept an image upload and return OCR results only."""
    try:
        f = request.files.get("image")
        if not f:
            return jsonify({"error": "No image", "step": "file_check"}), 400

        img_bytes = f.read()
        step = "read_file"

        squared = _to_square_png(img_bytes, 1024)
        step = "square_done"

        text_blocks = _extract_text_blocks(squared)
        step = "ocr_done"

        return jsonify({
            "step": step,
            "ocr_backend": OCR_BACKEND,
            "input_bytes": len(img_bytes),
            "squared_bytes": len(squared),
            "text_blocks": text_blocks,
        })
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc), "type": type(exc).__name__}), 500


@meme_lab_bp.route("/data/meme_images/<path:filename>")
def serve_meme_image(filename):
    return send_from_directory(IMG_DIR, filename)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _to_square_png(raw_bytes: bytes, size: int = 1024) -> bytes:
    """Contain-fit with black padding, then resize to exact square."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
    # Auto-rotate based on EXIF
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img) or img

    w, h = img.size
    mx = max(w, h)
    canvas = Image.new("RGBA", (mx, mx), (0, 0, 0, 255))
    paste_x = (mx - w) // 2
    paste_y = (mx - h) // 2
    canvas.paste(img, (paste_x, paste_y))
    canvas = canvas.resize((size, size), Image.LANCZOS).convert("RGB")

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _render_text_failed_badge(png_bytes: bytes) -> bytes:
    """Render a 'TEXT FAILED' badge onto the top-left corner of the image."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable font
    font_size = 28
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    text = "TEXT FAILED"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad = 8

    # Red/black background box at top-left
    draw.rectangle(
        [(4, 4), (4 + tw + pad * 2, 4 + th + pad * 2)],
        fill=(180, 0, 0),
        outline=(0, 0, 0),
        width=2,
    )
    draw.text((4 + pad, 4 + pad), text, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------

def _ocr_with_boxes(png_bytes: bytes) -> list:
    """Return list of {text, bbox} dicts. bbox = {x0,y0,x1,y1}."""
    if OCR_BACKEND == "pytesseract":
        return _ocr_pytesseract(png_bytes)
    elif OCR_BACKEND == "easyocr":
        return _ocr_easyocr(png_bytes)
    else:
        print("WARNING: No OCR backend — skipping text extraction", flush=True)
        return []


def _ocr_pytesseract(png_bytes: bytes) -> list:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    # Meme text is usually white/bright on busy backgrounds.
    # Run OCR on preprocessed versions × multiple PSM modes to maximize detection.
    variants = _preprocess_for_ocr(img)

    # PSM modes that work for memes:
    #   3 = fully automatic (good general fallback)
    #   6 = uniform block (good for multi-line meme text)
    #  11 = sparse text (good for single words scattered on image)
    psm_modes = ["--psm 3", "--psm 6", "--psm 11"]

    all_lines = {}
    run_id = 0
    for variant_img in variants:
        for psm in psm_modes:
            run_id += 1
            try:
                data = pytesseract.image_to_data(variant_img, output_type=Output.DICT,
                                                 config=psm)
            except Exception as e:
                print(f"[MemeLab] pytesseract failed with {psm}: {e}", flush=True)
                continue
            n = len(data.get("text", []))
            for i in range(n):
                txt = (data["text"][i] or "").strip()
                conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
                if not txt or conf < 50:
                    continue
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                vkey = (run_id, *key)
                if vkey not in all_lines:
                    all_lines[vkey] = {"texts": [], "x0": 9999, "y0": 9999, "x1": 0, "y1": 0, "conf": 0}
                all_lines[vkey]["texts"].append(txt)
                all_lines[vkey]["conf"] = max(all_lines[vkey]["conf"], conf)
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                all_lines[vkey]["x0"] = min(all_lines[vkey]["x0"], x)
                all_lines[vkey]["y0"] = min(all_lines[vkey]["y0"], y)
                all_lines[vkey]["x1"] = max(all_lines[vkey]["x1"], x + w)
                all_lines[vkey]["y1"] = max(all_lines[vkey]["y1"], y + h)

    # Deduplicate: if same text appears from multiple runs, keep highest conf
    seen_texts = {}
    for ln in all_lines.values():
        text = " ".join(ln["texts"])
        norm = text.upper().strip()
        if not norm:
            continue
        if norm not in seen_texts or ln["conf"] > seen_texts[norm]["conf"]:
            seen_texts[norm] = {
                "text": text,
                "bbox": {"x0": ln["x0"], "y0": ln["y0"], "x1": ln["x1"], "y1": ln["y1"]},
                "conf": ln["conf"],
            }

    # Post-filter: reject garbage strings
    filtered = []
    for item in seen_texts.values():
        if _is_ocr_noise(item["text"]):
            continue
        filtered.append({"text": item["text"], "bbox": item["bbox"]})

    # Remove fragments that are substrings of longer detected text
    result = _remove_substring_fragments(filtered)

    print(f"[MemeLab] pytesseract found {len(result)} text lines: "
          f"{[r['text'] for r in result]} (from {len(all_lines)} raw lines)", flush=True)
    return result


def _is_ocr_noise(text: str) -> bool:
    """Return True if a detected text string looks like OCR noise, not real meme text."""
    cleaned = text.strip()
    if not cleaned:
        return True
    # Too short — less than 4 chars (after stripping spaces) is almost always noise
    no_space = cleaned.replace(" ", "")
    if len(no_space) < 4:
        return True
    # Count actual letters vs non-letter chars
    letters = sum(1 for c in cleaned if c.isalpha())
    total = len(no_space)
    if total == 0:
        return True
    # If less than 60% of non-space chars are letters, it's noise
    if letters / total < 0.6:
        return True
    return False


def _remove_substring_fragments(items: list) -> list:
    """Remove OCR results whose text is a substring of another (longer) result.
    E.g., if we detect both 'COFFEE TIME?' and 'xi?', drop 'xi?'."""
    if len(items) <= 1:
        return items
    # Sort longest first
    sorted_items = sorted(items, key=lambda x: len(x["text"]), reverse=True)
    kept = []
    for item in sorted_items:
        norm = item["text"].upper().replace(" ", "")
        is_fragment = False
        for already in kept:
            longer = already["text"].upper().replace(" ", "")
            if norm in longer:
                is_fragment = True
                break
        if not is_fragment:
            kept.append(item)
    return kept


def _preprocess_for_ocr(img):
    """Create preprocessed versions for better meme text detection.
    Fewer variants = less noise. Focus on what works for Impact/bold text."""
    from PIL import ImageOps

    results = []
    gray = img.convert("L")

    # 1) Grayscale (baseline)
    results.append(gray)

    # 2) Inverted grayscale (white meme text becomes black-on-white)
    results.append(ImageOps.invert(gray))

    # 3) High-contrast binary — isolates bright white text
    bw = gray.point(lambda x: 255 if x > 200 else 0)
    results.append(bw)

    return results


def _ocr_easyocr(png_bytes: bytes) -> list:
    reader = _get_easyocr_reader()
    import numpy as np
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    arr = np.array(img)
    raw = reader.readtext(arr)
    result = []
    for (box, text, conf) in raw:
        # box is [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        result.append({
            "text": text.strip(),
            "bbox": {
                "x0": int(min(xs)),
                "y0": int(min(ys)),
                "x1": int(max(xs)),
                "y1": int(max(ys)),
            },
        })
    return result


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _bucket_for_box(bbox: dict) -> str:
    """Assumes 1024x1024 image."""
    x_mid = (bbox["x0"] + bbox["x1"]) / 2
    y_mid = (bbox["y0"] + bbox["y1"]) / 2
    if y_mid < 250:
        return "TOP"
    if y_mid > 775:
        return "BOTTOM"
    if x_mid < 250:
        return "LEFT"
    if x_mid > 775:
        return "RIGHT"
    return "CENTER"


def _extract_text_blocks(png_bytes: bytes) -> list:
    raw = _ocr_with_boxes(png_bytes)
    blocks = []
    seen = set()
    for item in raw:
        text = _normalize_text(item.get("text", ""))
        if not text:
            continue
        bbox = item.get("bbox")
        if not bbox:
            continue
        key = f"{text}::{bbox.get('x0')},{bbox.get('y0')},{bbox.get('x1')},{bbox.get('y1')}"
        if key in seen:
            continue
        seen.add(key)
        blocks.append({
            "text": text,
            "bucket": _bucket_for_box(bbox),
            "box": bbox,
        })
    return blocks


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_text_and_buckets(required_blocks, out_blocks, variant_name="rebuild"):
    """Validate generated image text against source requirements.

    - rebuild: enforce exact text AND region bucket match (strict)
    - reframe/anthro: enforce exact text only, ignore bucket position
    """
    out_texts = [b["text"] for b in out_blocks]
    strict_buckets = (variant_name == "rebuild")

    if strict_buckets:
        out_pairs = set(f"{b['bucket']}::{b['text']}" for b in out_blocks)

    for req in required_blocks:
        if req["text"] not in out_texts:
            return False, f'Text mismatch/missing: "{req["text"]}"'
        if strict_buckets:
            if f'{req["bucket"]}::{req["text"]}' not in out_pairs:
                return False, f'Bucket mismatch for: "{req["text"]}" expected {req["bucket"]}'
    return True, ""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _invariant_spec(container):
    blocks = container.get("text_blocks", [])
    lines = "\n".join(
        f'{i+1}. [{b["bucket"]}] "{b["text"]}"' for i, b in enumerate(blocks)
    )
    return f"""NON-NEGOTIABLE INVARIANTS:
- Output is a NEW image rendered from scratch, square 1:1.
- Do NOT copy the original framing, camera angle, geometry, or staging. No "skin over" look.
- All text must match EXACTLY as provided below (character-for-character).
- Each text block must appear in the SAME REGION BUCKET indicated (TOP/BOTTOM/LEFT/RIGHT/CENTER).
- Text must be readable and integrated, but in a NEW font/style/layout.
- Preserve the number of actors and their functional roles.
- Public figures: depict accurately only when required for the joke.
- Private individuals: replace with generic archetypes while preserving role.

EXACT TEXT BLOCKS (must be exact; must be placed in the indicated region bucket):
{lines}"""


def _build_three_prompts(container):
    base = f"""You are recreating a meme image from scratch based on a source meme concept.
Prioritize meme-speed readability and immediate comprehension. Avoid cinematic overproduction.
Avoid watermarks and artifacty text. Make the result look independently created.

{_invariant_spec(container)}"""

    v1 = """VARIANT 1 — FAITHFUL REBUILD:
- Recreate the same core scenario and joke logic.
- Change composition, perspective, and staging so it is clearly original.
- Keep the same functional roles and actor count."""

    v2 = """VARIANT 2 — STRUCTURAL REFRAME:
- Keep the exact same joke and roles, but restage it in a different environment/background.
- Change spatial relationships and composition significantly.
- Still keep the meme instantly readable."""

    # Lock species_map palette on first use
    if not container.get("species_map"):
        container["species_map"] = {
            "dominant": ["lion", "wolf", "eagle"],
            "cunning": ["fox"],
            "bureaucratic": ["beaver", "badger"],
            "everyman": ["dog", "deer", "raccoon"],
            "media": ["parrot", "crow"],
            "financial": ["squirrel"],
        }
        _save_container(container)

    v3 = """VARIANT 3 — ANTHROPOMORPHIC ANIMALS (SYMBOLIC):
- Replace all human actors with anthropomorphic animals chosen SYMBOLICALLY by their functional role.
- Use symbolic mapping guidance:
  dominant→lion/wolf/eagle; cunning→fox; bureaucratic→beaver/badger; everyman→dog/deer/raccoon; media→parrot/crow; financial→squirrel.
- Preserve actor count, roles, and EXACT TEXT BLOCKS + REGION BUCKETS.
- If a public figure is required, keep recognizable cues in animal form.
- Clean illustrative meme style. Meme-speed readability."""

    return [f"{base}\n\n{v1}", f"{base}\n\n{v2}", f"{base}\n\n{v3}"]


# ---------------------------------------------------------------------------
# OpenAI image generation
# ---------------------------------------------------------------------------

def _openai_generate_image_png(prompt_text: str) -> bytes:
    import requests as http_requests

    body = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt_text,
        "size": OPENAI_IMAGE_SIZE,
        "output_format": "png",
    }

    resp = http_requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=120,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI image gen failed ({resp.status_code}): {resp.text[:500]}")

    data = resp.json()
    b64 = data.get("data", [{}])[0].get("b64_json")
    if not b64:
        raise RuntimeError("No b64_json in OpenAI response")

    import base64
    return base64.b64decode(b64)


# ---------------------------------------------------------------------------
# Generate with retries + TEXT FAILED badge fallback
# ---------------------------------------------------------------------------

def _generate_with_retries(prompt_text, container, variant_name, attempt_limit):
    last_error = None
    last_png = None

    for attempt in range(1, attempt_limit + 1):
        try:
            png = _openai_generate_image_png(prompt_text)
            last_png = png

            # Validate OCR
            out_blocks = _extract_text_blocks(png)
            ok, err = _validate_text_and_buckets(container.get("text_blocks", []), out_blocks, variant_name)

            if not ok:
                last_error = err
                continue

            # Passed validation — save and return
            filename = f"{container['id']}_{variant_name}_{secrets.token_hex(6)}.png"
            abs_path = os.path.join(IMG_DIR, filename)
            with open(abs_path, "wb") as f:
                f.write(png)

            return {
                "id": filename,
                "url": f"/data/meme_images/{filename}",
                "variant": variant_name,
                "attempt_used": attempt,
            }

        except Exception as exc:
            last_error = str(exc)

    # Exhausted retries — return best-effort with TEXT FAILED badge
    if last_png:
        badged = _render_text_failed_badge(last_png)
        filename = f"{container['id']}_{variant_name}_{secrets.token_hex(6)}.png"
        abs_path = os.path.join(IMG_DIR, filename)
        with open(abs_path, "wb") as f:
            f.write(badged)

        return {
            "id": filename,
            "url": f"/data/meme_images/{filename}",
            "variant": variant_name,
            "attempt_used": attempt_limit,
            "warning": "TEXT_FAILED",
        }

    # No image at all (API never returned anything)
    return {
        "id": None,
        "url": None,
        "variant": variant_name,
        "error": f"Failed after {attempt_limit} attempts: {last_error or 'unknown'}",
    }


# ---------------------------------------------------------------------------
# Container persistence
# ---------------------------------------------------------------------------

def _save_container(container):
    fp = os.path.join(CONTAINER_DIR, f"{container['id']}.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(container, f, indent=2)


def _load_container(container_id):
    fp = os.path.join(CONTAINER_DIR, f"{container_id}.json")
    if not os.path.exists(fp):
        return None
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)
