#!/usr/bin/env bash
set -euo pipefail

# ================================
# Configuration
# ================================
MODELS_DIR="$(pwd)/models"
CONTAINERS_DIR="$(pwd)/containers"

# Model configurations
declare -A MODEL_REPOS=(
  ["medgemma-4b"]="https://huggingface.co/google/medgemma-4b-it"
  ["medgemma-27b"]="https://huggingface.co/google/medgemma-27b-it"
  ["llama3-70b"]="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"
  ["gpt-oss-120b"]="https://huggingface.co/openai/gpt-oss-120b"
)

declare -A MODEL_DIRS=(
  ["medgemma-4b"]="${MODELS_DIR}/medgemma-4b-it"
  ["medgemma-27b"]="${MODELS_DIR}/medgemma-27b-it"
  ["llama3-70b"]="${MODELS_DIR}/Llama-3.3-70B-Instruct"
  ["gpt-oss-120b"]="${MODELS_DIR}/gpt-oss-120b"
)

declare -A MODEL_NAMES=(
  ["medgemma-4b"]="MedGemma 4B"
  ["medgemma-27b"]="MedGemma 27B"
  ["llama3-70b"]="LLaMA 3.3 70B"
  ["gpt-oss-120b"]="GPT-OSS 120B"
)

declare -A MODEL_VRAM=(
  ["medgemma-4b"]="≥10GB VRAM"
  ["medgemma-27b"]="≥60GB VRAM"
  ["llama3-70b"]="≥140GB VRAM"
  ["gpt-oss-120b"]="≥80GB VRAM"
)

# ================================
# Sanity checks
# ================================
command -v git >/dev/null || { echo "ERROR: git not found"; exit 1; }
command -v git-lfs >/dev/null || { echo "ERROR: git-lfs not found"; exit 1; }
command -v apptainer >/dev/null || { echo "ERROR: apptainer not found"; exit 1; }

# Check directory structure
if [ ! -d "${CONTAINERS_DIR}" ]; then
  echo "ERROR: containers/ directory not found"
  echo "Please ensure you're running this script from the repository root"
  exit 1
fi

# ================================
# Ensure Git credential helper
# ================================
if ! git config --global --get credential.helper >/dev/null 2>&1; then
  echo "No git credential helper configured."
  echo "Setting:  git config --global credential.helper store"
  git config --global credential.helper store
else
  echo "Git credential helper:  $(git config --global --get credential. helper)"
fi

# ================================
# Prepare directories
# ================================
git lfs install --skip-repo
mkdir -p "${MODELS_DIR}"

# ================================
# Model Selection Menu
# ================================

echo ""
echo "================================"
echo "LLM Apptainer Builder"
echo "================================"
echo ""
echo "Available models:"
echo ""
echo "  1) ${MODEL_NAMES[medgemma-4b]}      (Text+Image, ${MODEL_VRAM[medgemma-4b]})"
echo "  2) ${MODEL_NAMES[medgemma-27b]}     (Text+Image, ${MODEL_VRAM[medgemma-27b]})"
echo "  3) ${MODEL_NAMES[llama3-70b]}    (Text only, ${MODEL_VRAM[llama3-70b]})"
echo "  4) ${MODEL_NAMES[gpt-oss-120b]}     (Text only, ${MODEL_VRAM[gpt-oss-120b]})"
echo "  5) Build ALL models"
echo ""
read -p "Select models to build (e.g., '1 3' or '5' for all): " selection

# Parse selection
declare -A BUILD_FLAGS
for model in medgemma-4b medgemma-27b llama3-70b gpt-oss-120b; do
  BUILD_FLAGS[$model]=false
done

for choice in $selection; do
  case $choice in
    1) BUILD_FLAGS[medgemma-4b]=true ;;
    2) BUILD_FLAGS[medgemma-27b]=true ;;
    3) BUILD_FLAGS[llama3-70b]=true ;;
    4) BUILD_FLAGS[gpt-oss-120b]=true ;;
    5)
      for model in medgemma-4b medgemma-27b llama3-70b gpt-oss-120b; do
        BUILD_FLAGS[$model]=true
      done
      ;;
    *)
      echo "ERROR: Invalid selection:  $choice"
      exit 1
      ;;
  esac
done

# Confirm selection
echo ""
echo "================================"
echo "Selected models:"
echo "================================"
for model in medgemma-4b medgemma-27b llama3-70b gpt-oss-120b; do
  if ${BUILD_FLAGS[$model]}; then
    echo "${MODEL_NAMES[$model]}"
  fi
done

echo ""
read -p "Proceed with build? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Build cancelled"
  exit 0
fi

# ================================
# Clone & Prune models
# ================================

echo ""
echo "================================"
echo "Cloning and Pruning repositories..."
echo "================================"

for model in medgemma-4b medgemma-27b llama3-70b gpt-oss-120b; do
  if ${BUILD_FLAGS[$model]}; then
    model_dir="${MODEL_DIRS[$model]}"
    model_repo="${MODEL_REPOS[$model]}"

    # 1. CLONE
    if [ ! -d "${model_dir}" ]; then
      echo ""
      echo "Cloning ${MODEL_NAMES[$model]}..."
      git clone "${model_repo}" "${model_dir}"
      (cd "${model_dir}" && git lfs pull)
      echo "${MODEL_NAMES[$model]} cloned successfully."
    else
      echo "${MODEL_NAMES[$model]} already present, skipping clone."
    fi

    # 2. PRUNE bloat BEFORE building the container
    echo "Pruning ${MODEL_NAMES[$model]} to save space..."

    case $model in
      "gpt-oss-120b")
        # GPT-OSS contains massive duplicate weights in 'original' and 'metal' folders.
        # We only need the top-level MXFP4 weights.
        if [ -d "${model_dir}/original" ]; then
            echo "  - Removing ${model_dir}/original"
            rm -rf "${model_dir}/original"
        fi
        if [ -d "${model_dir}/metal" ]; then
            echo "  - Removing ${model_dir}/metal"
            rm -rf "${model_dir}/metal"
        fi
        ;;

      "llama3-70b")
        # Check if we have both .bin (PyTorch) and .safetensors.
        # Safetensors is preferred; .bin is redundant if both exist.
        if ls "${model_dir}"/*.safetensors 1> /dev/null 2>&1; then
            if ls "${model_dir}"/*.bin 1> /dev/null 2>&1; then
                 echo "  - Detected .safetensors, removing redundant .bin files..."
                 rm -f "${model_dir}"/*.bin
            fi
        fi
        ;;
    esac
    echo "Pruning complete for ${model}."
  fi
done

echo ""
echo "Model repositories ready for build."
# ================================
# Build Apptainer images
# ================================

echo ""
echo "================================"
echo "Building Apptainer containers..."
echo "================================"

BUILT_IMAGES=()

for model in medgemma-4b medgemma-27b llama3-70b gpt-oss-120b; do
  if ${BUILD_FLAGS[$model]}; then
    echo ""
    echo "Building ${model}.sif..."

    def_file="${CONTAINERS_DIR}/${model}/container.def"
    sif_file="${CONTAINERS_DIR}/${model}/${model}.sif"

    if [ !  -f "${def_file}" ]; then
      echo "ERROR: Definition file not found: ${def_file}"
      exit 1
    fi

    apptainer build --force "${sif_file}" "${def_file}"
    BUILT_IMAGES+=("${sif_file}")
    echo "${model}. sif built successfully"
  fi
done

# ================================
# Summary
# ================================

echo ""
echo "================================"
echo "Build complete!"
echo "================================"
echo ""
echo "Generated containers:"
for img in "${BUILT_IMAGES[@]}"; do
  size=$(du -h "$img" | cut -f1)
  basename_img=$(basename "$img")
  echo "${basename_img} (${size})"
done

echo ""
echo "================================"
echo "Next steps:"
echo "================================"
echo ""
echo "Run a container:"
echo "  apptainer run --nv --bind \$PWD:\$PWD dist/<model>.sif --help"
echo ""
echo "Example usage:"
echo "  apptainer run --nv --bind \$PWD:\$PWD dist/medgemma-4b. sif \\"
echo "    --instructions task. txt \\"
echo "    --input report.txt \\"
echo "    --output result.json"
echo ""
echo "Clean up model weights (optional, saves disk space):"
echo "  rm -rf models/"
echo ""
