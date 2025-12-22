#!/usr/bin/env bash
set -euo pipefail

# ================================
# Configuration
# ================================
MODELS_DIR="$(pwd)/models"

REPO_4B="https://huggingface.co/google/medgemma-4b-it"
REPO_27B="https://huggingface.co/google/medgemma-27b-it"

DIR_4B="${MODELS_DIR}/medgemma-4b-it"
DIR_27B="${MODELS_DIR}/medgemma-27b-it"

SIF_4B="medgemma-4b.sif"
SIF_27B="medgemma-27b.sif"

DEF_4B="medgemma-4b.def"
DEF_27B="medgemma-27b.def"

# ================================
# Sanity checks
# ================================
command -v git >/dev/null || { echo "git not found"; exit 1; }
command -v git-lfs >/dev/null || { echo "git-lfs not found"; exit 1; }
command -v apptainer >/dev/null || { echo "apptainer not found"; exit 1; }

# ================================
# Ensure Git credential helper
# ================================
if ! git config --global --get credential.helper >/dev/null 2>&1; then
  echo "No git credential helper configured."
  echo "Setting: git config --global credential.helper store"
  git config --global credential.helper store
else
  echo "Git credential helper already configured:"
  git config --global --get credential.helper
fi

# ================================
# Prepare Git LFS
# ================================
git lfs install --skip-repo

mkdir -p "${MODELS_DIR}"

# ================================
# Clone models
# ================================
if [ ! -d "${DIR_4B}" ]; then
  echo "Cloning MedGemma 4B..."
  git clone "${REPO_4B}" "${DIR_4B}"
  (cd "${DIR_4B}" && git lfs pull)
else
  echo "MedGemma 4B already present, skipping"
fi

if [ ! -d "${DIR_27B}" ]; then
  echo "Cloning MedGemma 27B..."
  git clone "${REPO_27B}" "${DIR_27B}"
  (cd "${DIR_27B}" && git lfs pull)
else
  echo "MedGemma 27B already present, skipping"
fi

echo "Model repositories ready."

# ================================
# Build Apptainer images
# ================================
echo "Building ${SIF_4B}..."
apptainer build --force "${SIF_4B}" "${DEF_4B}"

echo "Building ${SIF_27B}..."
apptainer build --force "${SIF_27B}" "${DEF_27B}"

echo "================================"
echo "Build complete"
echo "Generated images:"
echo "  - ${SIF_4B}"
echo "  - ${SIF_27B}"
echo "================================"
