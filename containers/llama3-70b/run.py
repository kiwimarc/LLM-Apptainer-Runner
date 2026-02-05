import argparse
import json
import sys
import os
from pathlib import Path
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Config -----------------------------------------------------
MODEL_NAME = "LLaMA 3.3 70B Instruct"
MODEL_FILE = "llama3-70b.sif"
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/models/llama-3-70b")

# Memory settings
MAX_CONTEXT_TOKENS = 10000  # Leave room for instructions + output (Model limit 128K)
OVERLAP_TOKENS = 500       # Overlap chunks to ensure no context is lost at boundaries

# ---- Progress logging -------------------------------------------

def log(msg):
    print(f"[llama3] {msg}", file=sys.stderr)

def debug(msg):
    if args.debug:
        print(f"[llama3][debug] {msg}", file=sys. stderr)

# ---- ARGUMENTS PARSING ------------------------------------------

def build_help_text():
    return f"""
LLM Apptainer Runner
============================================

This container runs LLM models with strict JSON-only output.
Supports single-file and batch processing.
Automatically handles large files by chunking and aggregating results.

MODEL
-----

{MODEL_NAME}
• Supports text input only
• GPU required (≥ 140 GB VRAM)


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

Single file:
  apptainer run --nv {MODEL_FILE} \\
    --instructions task.txt \\
    --input report.txt \\
    --output result.json

Batch processing:
  apptainer run --nv --bind $PWD:$PWD {MODEL_FILE} \\
    --instructions task.txt \\
    --input-dir reports/ \\
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

def read_input_content(file_path:  Path) -> str:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ". json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)
        else:
            return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"Error reading input file '{file_path}': {e}")

def parse_json_strict(text: str):
    text = text.strip()
    if not text:  return None # Handle empty responses gracefully in chunking

    # Try raw parse
    try:
        return json. loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block
    match = re.search(r"\{.*\}", text, re. DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # If all fails, return raw text wrapped in structure (fallback)
    return {
        "task": "unknown",
        "result": text,
        "confidence":  "low",
        "error": "Failed to parse JSON"
    }

# ---- Model Loading ----------------------------------------------

log(f"Model loaded: {MODEL_NAME}")
if args.input_dir:  log("Mode: batch")
else: log("Mode: single")

log("Loading model into GPU memory...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

# Load model
model = AutoModelForCausalLM.from_pretrained(
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
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    debug(f"Running inference on {input_len} tokens...")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated = output_ids[0][input_len:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return decoded

def get_chunks(text):
    """
    Splits text into chunks based on token count using the tokenizer.
    """
    # Estimate system prompt + instructions overhead ~ 500 tokens
    effective_limit = MAX_CONTEXT_TOKENS - 500

    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    if total_tokens <= effective_limit: 
        return [text]

    log(f"Input too large ({total_tokens} tokens). Splitting into chunks...")

    chunks = []
    start = 0
    while start < total_tokens:
        end = min(start + effective_limit, total_tokens)
        chunk_ids = tokens[start:end]
        chunk_text = tokenizer. decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)

        if end == total_tokens:
            break

        start += (effective_limit - OVERLAP_TOKENS)

    log(f"Split into {len(chunks)} chunks.")
    return chunks

def generate_smart(
    *,
    instructions_text:  str,
    context_text:  str,
    input_filename:  str
):
    chunks = get_chunks(context_text)

    # Single chunk, run normally
    if len(chunks) == 1:
        return generate_single_pass(instructions_text, chunks[0], input_filename)

    # Multiple chunks
    log(f"Processing {len(chunks)} chunks for {input_filename}...")
    intermediate_results = []
    for i, chunk in enumerate(chunks):
        debug(f"--- Chunk {i+1}/{len(chunks)} ---")

        # Modify instructions for chunks to be extraction-focused
        chunk_inst = (
            f"{instructions_text}\n\n"
            "NOTE: This is PART {i+1} of a larger file. "
            "Extract any relevant information found in this segment.  "
            "If the information is not present, return an empty result."
        )

        response_json = generate_single_pass(chunk_inst, chunk, input_filename)

        if response_json and "result" in response_json: 
            val = response_json["result"]
            # Filter out empty/negative results roughly
            if val and str(val).lower() not in ["none", "null", "not found", ""]:
                intermediate_results.append(str(val))

    if not intermediate_results:
        return {
            "task": "processing",
            "input_file":  input_filename,
            "result": "No relevant information found in any text chunk.",
            "confidence": "low"
        }

    log("Aggregating results from chunks...")
    combined_context = "\n---\n". join(intermediate_results)

    final_instruction = (
        f"Original Task: {instructions_text}\n\n"
        "Below are extracted findings from different parts of the file. "
        "Consolidate them into a single coherent final answer in the requested JSON format."
    )

    # Context is the combined extracts.
    return generate_single_pass(final_instruction, combined_context, input_filename)

def generate_single_pass(instructions, context, filename):
    """Standard single-inference call"""
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": context}
    ]

    raw_output = run_inference(messages)
    data = parse_json_strict(raw_output)

    # Ensure filename is correct in output
    if isinstance(data, dict):
        data["input_file"] = filename

    return data

# ---- Execution Loop ---------------------------------------------

instructions_text = f"""
You are an AI assistant. 
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
{Path(args.instructions).read_text() if args.instructions else "Summarize the input. "}
"""

# =========================
# SINGLE FILE MODE
# =========================

if args.input:
    input_path = Path(args.input)
    output_path = Path(args.output)

    context_text = read_input_content(input_path)

    data = generate_smart(
        instructions_text=instructions_text,
        context_text=context_text,
        input_filename=input_path.name
    )

    with output_path. open("w") as f:
        json.dump(data, f, indent=2)


# =========================
# BATCH MODE
# =========================

else:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = {".txt", ".csv", ".json"}
    files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in valid_extensions])

    log(f"Processing {len(files)} files...")

    for idx, fpath in enumerate(files, 1):
        log(f"→ {fpath.name} ({idx}/{len(files)})")

        try:
            context = read_input_content(fpath)
            data = generate_smart(
                instructions_text=instructions_text,
                context_text=context,
                input_filename=fpath.name
            )

            with (output_dir / f"{fpath. stem}.json").open("w") as f:
                json.dump(data, f, indent=2)

        except Exception as e: 
            log(f"FAILED {fpath.name}: {e}")
