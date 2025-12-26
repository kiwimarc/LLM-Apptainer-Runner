import argparse
import json
import sys
import os
from pathlib import Path
from PIL import Image
import torch
import re
from transformers import AutoProcessor, AutoModelForImageTextToText

# ---- Config -----------------------------------------------------
MODEL_NAME = "MedGemma 27B (BIG, multimodal)"
MODEL_FILE = "medgemma-27b.sif"
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/models/medgemma-27b-it")
HAS_IMAGES = False

# ---- Progress logging -------------------------------------------

def log(msg):
    print(f"[medgemma] {msg}", file=sys.stderr)

def debug(msg):
    if args.debug:
        print(f"[medgemma][debug] {msg}", file=sys.stderr)


# ---- ARGUMENTS PARSING AND HELP MENU ------------------------
def build_help_text():
    return f"""
MedGemma Apptainer Runner
=========================

This container runs MedGemma models with strict JSON-only output.
Supports single-file and batch processing.

MODEL
-----

{MODEL_NAME}
• Supports text + image input
• GPU recommended (≥ 48 GB VRAM)


USAGE
-----

Single-file mode:
  apptainer run --nv --bind $PWD:$PWD {MODEL_FILE} \\
    --instructions FILE \\
    --input FILE \\
    --output FILE [options]

Batch mode:
  apptainer run --nv --bind $PWD:$PWD {MODEL_FILE} \\
    --instructions FILE \\
    --input-dir DIR \\
    --output-dir DIR [options]


REQUIRED ARGUMENTS
------------------

--instructions FILE
    Path to a text file containing the task instructions.

Choose EXACTLY ONE:
--input FILE
--input-dir DIR


OUTPUT OPTIONS
--------------

Single-file mode:
--output FILE

Batch mode:
--output-dir DIR

MODEL BEHAVIOR
--------------

• Output is ALWAYS valid JSON
• No markdown, prose, or explanations
• Invalid JSON causes failure

EXAMPLES
--------
Image processing:
--image FILE
    Path to a single image file (PNG, JPG).

--image-dir DIR
    Directory containing images for batch mode.
    Image filenames must match text filenames:
        report1.txt -> report1.png

Single file (with image):
  apptainer run --nv {MODEL_FILE} \\
    --instructions task.txt \\
    --input report.txt \\
    --image xray.png \\
    --output result.json

Batch processing:
  apptainer run --nv --bind $PWD:$PWD {MODEL_FILE} \\
    --instructions task.txt \\
    --input-dir reports/ \\
    --image-dir images/ \\
    --output-dir results/

DEBUG OPTIONS
-------------

--debug
    Print the raw model output before JSON parsing.
    Useful for diagnosing invalid or empty model responses.

NOTES
-----

• Use --dry-run to preview actions
• Use --nv to enable GPU
• Designed for Apptainer / HPC
"""

parser = argparse.ArgumentParser(
    description=build_help_text(),
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument("--instructions", required=False)
parser.add_argument("--input")
parser.add_argument("--input-dir")
parser.add_argument("--output")
parser.add_argument("--output-dir")
parser.add_argument("--image")
parser.add_argument("--image-dir")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--debug", action="store_true")

args = parser.parse_args()

# ---- Validation -------------------------------------------------

if bool(args.input) == bool(args.input_dir):
    sys.exit("ERROR: Use exactly one of --input or --input-dir")

if args.input and not args.output:
    sys.exit("ERROR: --output is required in single-file mode")

if args.input_dir and not args.output_dir:
    sys.exit("ERROR: --output-dir is required in batch mode")

# ---- Dry run ----------------------------------------------------

if args.dry_run:
    log("DRY RUN (no model loaded)\n")

    log(f"Instructions file : {args.instructions}")

    if args.input:
        log("Mode              : single-file")
        log(f"Input file        : {args.input}")
        log(f"Output file       : {args.output}")
        log(f"Image file        : {args.image or 'none'}")
    else:
        log("Mode              : batch")
        log(f"Input directory   : {args.input_dir}")
        log(f"Output directory  : {args.output_dir}")
        log(f"Image directory   : {args.image_dir or 'none'}")

    log("\nModel             : google/medgemma-4b-it")
    log("Output format     : strict JSON")
    log("\nDry run complete.")
    sys.exit(0)

# ---- System prompt ----------------------------------------------

instructions_text = f"""
You are a medical AI system.
You MUST respond with VALID JSON ONLY.
No prose, no markdown.

JSON SCHEMA:
{{
  "task": "string",
  "input_file": "string",
  "result": "string",
  "confidence": "low | medium | high"
}}

INSTRUCTIONS:
{Path(args.instructions).read_text()}
"""

# ---- Helper Functions -------------------------------------------

def read_input_content(file_path: Path) -> str:
    """
    Reads content from txt, csv, or json files and returns a string
    formatted for the model prompt.
    """
    suffix = file_path.suffix.lower()

    try:
        # If JSON, load it and dump it
        if suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)

        # If CSV or TXT, read raw text
        else:
            return file_path.read_text(encoding="utf-8")

    except Exception as e:
        raise RuntimeError(f"Error reading input file '{file_path}': {e}")

def parse_json_strict(text: str):
    text = text.strip()

    if not text:
        raise RuntimeError("Model returned empty output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"No JSON found in model output:\n{text}")

    return json.loads(match.group(0))

# ---- Real execution ---------------------------------------------

log(f"Model loaded: {MODEL_NAME}")
if args.input_dir:
    log("Mode: batch")
else:
    log("Mode: single")

# Load model + processor once
log("Loading model into GPU memory…")
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
log("Model ready")

def generate(
    *,
    instructions_text: str,
    context_text: str,
    input_filename: str,
    image=None
):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": instructions_text
                }
            ]
        },
        {
            "role": "user",
            "content": (
                [{"type": "text", "text": context_text}] +
                ([{"type": "image", "image": image}] if image is not None else [])
            )
        }
    ]
    debug("Tokenizing input")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    debug("Running inference")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    generated = output_ids[0][input_len:]

    debug("Decoding output")
    decoded = processor.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    if args.debug:
        debug("\n===== RAW MODEL OUTPUT =====")
        debug(repr(decoded))
        debug("===== END RAW OUTPUT =====\n")


    return parse_json_strict(decoded)

# =========================
# SINGLE FILE MODE
# =========================

if args.input:
    input_path = Path(args.input)
    output_path = Path(args.output)

    image = None
    if args.image:
        image = Image.open(args.image).convert("RGB")

    # Use helper to read txt/csv/json
    context_text = read_input_content(input_path)

    data = generate(
        instructions_text=instructions_text,
        context_text=context_text,
        input_filename=input_path.name,
        image=image
    )

    try:
        with output_path.open("w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        raise RuntimeError(
            f"Cannot write output file '{output_path}'. "
            f"Make sure the directory is bind-mounted and writable."
        ) from e

# =========================
# BATCH MODE
# =========================

else:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir) if args.image_dir else None
    # Collect all supported file types
    valid_extensions = {".txt", ".csv", ".json"}
    context_files = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    ])

    total = len(context_files)
    log(f"Processing {total} files (found .txt, .csv, .json)")

    for idx, context_file in enumerate(context_files, start=1):
        log(f"→ {context_file.name} ({idx}/{total})")
        image = None

        if image_dir:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                img_path = image_dir / f"{context_file.stem}{ext}"
                if img_path.exists():
                    image = Image.open(img_path).convert("RGB")
                    break

        data = generate(
            instructions_text=instructions_text,
            context_text=context_file.read_text(),
            input_filename=context_file.name,
            image=image
        )

        output_file = output_dir / f"{context_file.stem}.json"
        try:
            with output_file.open("w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            raise RuntimeError(
                f"Cannot write output file '{output_file}'. "
                f"Make sure the directory is bind-mounted and writable."
            ) from e
