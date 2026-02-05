# LLM Apptainer Runner

This project provides **offline, GPU-enabled Apptainer containers** for running multiple **LLM models** with **strict JSON-only output**, designed for **HPC and Apptainer environments**.

## Supported Models

| Model | Size | Modality | Container |
|-------|------|----------|-----------|
| [**MedGemma 4B**](https://huggingface.co/google/medgemma-4b-it) | 4B | Text + Image | `medgemma-4b. sif` |
| [**MedGemma 27B**](https://huggingface.co/google/medgemma-27b-it) | 27B | Text + Image | `medgemma-27b.sif` |
| [**LLaMA 3. 3 70B Instruct**](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 70B | Text only | `llama3-70b. sif` |
| [**GPT-OSS 120B**](https://huggingface.co/openai/gpt-oss-120b) | 120B | Text only | `gpt-oss-120b.sif` |

---

## Key Features

* ✅ Strict JSON-only output (machine-safe)
* ✅ Single-file and batch processing
* ✅ Optional image input (MedGemma models)
* ✅ Offline execution (models embedded)
* ✅ Dry-run and debug modes
* ✅ HPC / Apptainer compatible
* ✅ Multi-model support

---

## Building Containers

### Prerequisites

```bash
# Required tools
git
git-lfs
apptainer
```

### Build All Models

```bash
./build. sh
```

This will: 
1. Clone all model repositories from Hugging Face
2. Build Apptainer containers for each model
3. Generate `.sif` files ready to run

---

## Basic Usage

Always bind your working directory so outputs are writable:

```bash
--bind $PWD:$PWD
```

Always enable GPU:

```bash
--nv
```

---

## Single-file Processing

### Text-only

```bash
apptainer run --nv --bind $PWD: $PWD {MODEL}. sif \
  --instructions instructions.txt \
  --input report.txt \
  --output result.json
```

### Text + Image (MedGemma only)

```bash
apptainer run --nv --bind $PWD:$PWD {MODEL}.sif \
  --instructions instructions.txt \
  --input report.txt \
  --image xray.png \
  --output result.json
```

---

## Batch Processing

### Text-only batch

```bash
apptainer run --nv --bind $PWD:$PWD {MODEL}.sif \
  --instructions instructions.txt \
  --input-dir inputs/ \
  --output-dir outputs/
```

### Batch with images (MedGemma only)

Image filenames must match text files:

```
report1.txt -> report1.png
```

```bash
apptainer run --nv --bind $PWD:$PWD {MODEL}.sif \
  --instructions instructions.txt \
  --input-dir reports/ \
  --image-dir images/ \
  --output-dir results/
```

---

## Help Menu

Each container exposes a full help menu:

```bash
apptainer run medgemma-4b.sif --help
apptainer run llama3-70b.sif --help
apptainer run gpt-oss-120b.sif --help
```

This shows: 

* Required flags
* Single vs batch usage
* Image support (if applicable)
* Output schema
* Debug and dry-run options

---

## Dry Run Mode

Preview what will happen **without loading the model**:

```bash
apptainer run llama3-70b.sif \
  --instructions instructions.txt \
  --input report.txt \
  --output result.json \
  --dry-run
```

---

## Debug Mode

Print raw model output before JSON parsing:

```bash
apptainer run --nv --bind $PWD: $PWD gpt-oss-120b. sif \
  --instructions instructions.txt \
  --input report.txt \
  --output result.json \
  --debug
```

Useful if the model returns invalid or empty JSON.

---

## Output Format

All outputs are **JSON** with this schema:

```json
{
  "task": "string",
  "input_file": "string",
  "result": "string",
  "confidence": "low | medium | high"
}
```

If valid JSON cannot be produced: 

* The run fails
* No output file is written
* Exit code is non-zero

---

## Hardware Requirements

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| MedGemma 4B | ≥ ... GB | Text + Image support |
| MedGemma 27B | ≥ ... GB | Text + Image support |
| LLaMA 3.3 70B | ≥ 140 GB | Text only|
| GPT-OSS 120B | ≥ 80 GB | Text only |

---

## Models Folder Structure

Models must exist locally before building containers:

```
models/
├── medgemma-4b-it/
├── medgemma-27b-it/
├── Llama-3.3-70B-Instruct/
└── gpt-oss-120b/
```

These are cloned automatically by `build.sh` using Git LFS.

---

## Example Folder

```
examples/
├── single-text/
├── single-image/
├── batch-text/
└── batch-image/
```

Each example includes: 

* Instructions file
* Input text(s)
* Optional images (for MedGemma)

---

## Notes

* Internet access is **not required at runtime**
* First build requires Hugging Face access
* MedGemma requires accepting Health AI Developer Foundation's terms of use
* LLaMA 3.3 requires accepting Llama3.3 Community License Agreement

---

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

---

## Model-Specific Notes

### MedGemma (4B, 27B)
- Medical domain fine-tuned
- Supports multimodal input (text + images)
- Optimized for clinical/medical tasks

### LLaMA 3.3 70B Instruct
- General-purpose
- Text-only

### GPT-OSS 120B
- General-purpose model
- Text-only
