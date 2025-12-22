# MedGemma Apptainer Runner

This project provides **offline, GPU-enabled Apptainer containers** for running **MedGemma** models with **strict JSON-only output**, designed for **HPC, Apptainer**.

Two models are supported:

* [**medgemma-4b**](https://huggingface.co/google/medgemma-4b-it) – Multimodal (text + image)
* [**medgemma-27b**](https://huggingface.co/google/medgemma-27b-it) – Multimodal (text + image)


---

## Key Features

* ✅ Strict JSON-only output (machine-safe)
* ✅ Single-file and batch processing
* ✅ Optional image input
* ✅ Offline execution (models embedded)
* ✅ Dry-run and debug modes
* ✅ HPC / Apptainer compatible

---

## Container Files

| Model        | Container          |
| ------------ | ------------------ |
| MedGemma 4B  | `medgemma-4b.sif`  |
| MedGemma 27B | `medgemma-27b.sif` |

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
apptainer run --nv --bind $PWD:$PWD {MODEL} \
  --instructions instructions.txt \
  --input report.txt \
  --output result.json
```

### Text + Image 

```bash
apptainer run --nv --bind $PWD:$PWD {MODEL} \
  --instructions instructions.txt \
  --input report.txt \
  --image xray.png \
  --output result.json
```

---


## Batch Processing

### Text-only batch

```bash
apptainer run --nv --bind $PWD:$PWD {MODEL} \
  --instructions instructions.txt \
  --input-dir inputs/ \
  --output-dir outputs/
```

### Batch with images

Image filenames must match text files:

```
report1.txt -> report1.png
```

```bash
apptainer run --nv --bind $PWD:$PWD {MODEL} \
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
apptainer run medgemma-27b.sif --help
```

This shows:

* Required flags
* Single vs batch usage
* Image support
* Output schema
* Debug and dry-run options

---

## Dry Run Mode

Preview what will happen **without loading the model**:

```bash
apptainer run medgemma-4b.sif \
  --instructions instructions.txt \
  --input report.txt \
  --output result.json \
  --dry-run
```

---

## Debug Mode

Print raw model output before JSON parsing:

```bash
apptainer run --nv --bind $PWD:$PWD medgemma-4b.sif \
  --instructions instructions.txt \
  --input report.txt \
  --output result.json \
  --debug
```

Useful if the model returns invalid or empty JSON.

---

## Output Format

All outputs are **guaranteed JSON** with this schema, enforced by the system prompt in `run_*.py`:


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

### MedGemma 4B

* GPU recommended (≥ 8 GB VRAM)
* Text + image support

### MedGemma 27B

* GPU required (≥ 48 GB VRAM)
* Text + image support

---

## Models Folder

Models must exist locally before building containers:

```
models/
├── medgemma-4b-it/
└── medgemma-27b-it/
```

These are cloned using `git clone` with Git LFS.

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
* Optional images

---

## Notes

* Internet access is **not required at runtime**
* First build requires Hugging Face access

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.