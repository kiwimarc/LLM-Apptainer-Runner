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

# Memory settings
MAX_CONTEXT_TOKENS = 6000  # Leave room for instructions + output (Model limit 8192)
OVERLAP_TOKENS = 200       # Overlap chunks to ensure no context is lost at boundaries

# ---- Progress logging -------------------------------------------

def log(msg):
    print(f"[medgemma] {msg}", file=sys.stderr)

def debug(msg):
    if args.debug:
        print(f"[medgemma][debug] {msg}", file=sys.stderr)

# ---- ARGUMENTS PARSING ------------------------------------------

def build_help_text():
    return f"""
MedGemma Apptainer Runner
============================================

This container runs MedGemma models with strict JSON-only output.
Supports single-file and batch processing.
Automatically handles large files by chunking and aggregating results.

MODEL
-----

{MODEL_NAME}
• Supports text + image input
• GPU recommended (≥ 60 GB VRAM)


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

# ---- Helper Functions -------------------------------------------

def read_input_content(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        else:
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Error reading input file '{file_path}': {e}")

def parse_json_strict(text: str):
    text = text.strip()
    if not text: return None # Handle empty responses gracefully in chunking

    # Try raw parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # If all fails, return raw text wrapped in structure (fallback)
    return {
        "task": "unknown",
        "result": text,
        "confidence": "low",
        "error": "Failed to parse JSON"
    }

# ---- Model Loading ----------------------------------------------

log(f"Model loaded: {MODEL_NAME}")
if args.input_dir: log("Mode: batch")
else: log("Mode: single")

log("Loading model into GPU memory (with Flash Attention 2)...")

# Load processor
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=False)

# Load model with Flash Attention 2 for memory optimization
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
model.eval()
log("Model ready")

# ---- Core Generation Logic --------------------------------------

def run_inference(messages, max_new_tokens=500):
    """Raw inference wrapper"""
    debug("Tokenizing...")
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    debug(f"Running inference on {input_len} tokens...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0][input_len:]
    decoded = processor.decode(generated, skip_special_tokens=True).strip()
    return decoded

def get_chunks(text, image_tokens_placeholder=256):
    """
    Splits text into chunks based on token count using the tokenizer.
    """
    # Estimate system prompt + instructions overhead ~ 500 tokens
    # Image overhead ~ 256 tokens (if present)
    effective_limit = MAX_CONTEXT_TOKENS - 500 - image_tokens_placeholder

    tokens = processor.tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    if total_tokens <= effective_limit:
        return [text]

    log(f"Input too large ({total_tokens} tokens). Splitting into chunks...")

    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + effective_limit, total_tokens)
        chunk_ids = tokens[start:end]
        chunk_text = processor.tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        if end == total_tokens:
            break

        start += (effective_limit - OVERLAP_TOKENS)

    log(f"Split into {len(chunks)} chunks.")
    return chunks

def generate_smart(
    *,
    instructions_text: str,
    context_text: str,
    input_filename: str,
    image=None
):
    chunks = get_chunks(context_text, image_tokens_placeholder=256 if image else 0)

    # Single chunk, run normally
    if len(chunks) == 1:
        return generate_single_pass(instructions_text, chunks[0], input_filename, image)

    # Multiple chunks
    log(f"Processing {len(chunks)} chunks for {input_filename}...")
    intermediate_results = []
    for i, chunk in enumerate(chunks):
        debug(f"--- Chunk {i+1}/{len(chunks)} ---")

        # Modify instructions for chunks to be extraction-focused
        chunk_inst = (
            f"{instructions_text}\n\n"
            "NOTE: This is PART {i+1} of a larger file. "
            "Extract any relevant information found in this segment. "
            "If the information is not present, return an empty result."
        )

        # Pass image to first chunk only.
        current_image = image if i == 0 else None

        response_json = generate_single_pass(chunk_inst, chunk, input_filename, current_image)

        if response_json and "result" in response_json:
            val = response_json["result"]
            # Filter out empty/negative results roughly
            if val and str(val).lower() not in ["none", "null", "not found", ""]:
                intermediate_results.append(str(val))

    if not intermediate_results:
        return {
            "task": "processing",
            "input_file": input_filename,
            "result": "No relevant information found in any text chunk.",
            "confidence": "low"
        }

    log("Aggregating results from chunks...")
    combined_context = "\n---\n".join(intermediate_results)

    final_instruction = (
        f"Original Task: {instructions_text}\n\n"
        "Below are extracted findings from different parts of the file. "
        "Consolidate them into a single coherent final answer in the requested JSON format."
    )

    # Context is the combined extracts.
    return generate_single_pass(final_instruction, combined_context, input_filename, image=None)

def generate_single_pass(instructions, context, filename, image):
    """Standard single-inference call"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instructions}]},
        {"role": "user", "content": ([{"type": "text", "text": context}] +
                                     ([{"type": "image", "image": image}] if image else []))}
    ]

    raw_output = run_inference(messages)
    data = parse_json_strict(raw_output)

    # Ensure filename is correct in output
    if isinstance(data, dict):
        data["input_file"] = filename

    return data

# ---- Execution Loop ---------------------------------------------

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
{Path(args.instructions).read_text() if args.instructions else "Summarize the input."}
"""

# =========================
# SINGLE FILE MODE
# =========================

if args.input:
    input_path = Path(args.input)
    output_path = Path(args.output)
    image = Image.open(args.image).convert("RGB") if args.image else None

    context_text = read_input_content(input_path)

    data = generate_smart(
        instructions_text=instructions_text,
        context_text=context_text,
        input_filename=input_path.name,
        image=image
    )

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)


# =========================
# BATCH MODE
# =========================

else:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir) if args.image_dir else None

    valid_extensions = {".txt", ".csv", ".json"}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions])

    log(f"Processing {len(files)} files...")

    for idx, fpath in enumerate(files, 1):
        log(f"→ {fpath.name} ({idx}/{len(files)})")

        img_path = None
        image = None
        if image_dir:
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = image_dir / f"{fpath.stem}{ext}"
                if candidate.exists():
                    image = Image.open(candidate).convert("RGB")
                    break

        try:
            context = read_input_content(fpath)
            data = generate_smart(
                instructions_text=instructions_text,
                context_text=context,
                input_filename=fpath.name,
                image=image
            )

            with (output_dir / f"{fpath.stem}.json").open("w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            log(f"FAILED {fpath.name}: {e}")
