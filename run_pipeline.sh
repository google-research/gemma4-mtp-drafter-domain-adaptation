#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Prevent CUDA memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# =================================================================
# Centralized Configuration & File/Directory Naming Constants
# =================================================================
MODEL_NAME="google/gemma-4-E4B-it"
ASSISTANT_NAME="google/gemma-4-E4B-it-assistant"
# DATASET="PaDaS-Lab/Instruct-to-SPARQL"
# DATASET="openai/gsm8k"
# DATASET="mbpp"
DATASET="orkg/SciQA"

# Root Output Directories
RESULTS_ROOT_DIR="./results_sciqa_3e_2e-4"
BASELINE_DIR="${RESULTS_ROOT_DIR}/baseline"
TARGET_FINETUNED_DIR="${RESULTS_ROOT_DIR}/target_finetuned"
TARGET_EVAL_DIR="${RESULTS_ROOT_DIR}/target_finetuned_eval"
BOTH_FINETUNED_DIR="${RESULTS_ROOT_DIR}/both_finetuned"
BOTH_EVAL_DIR="${RESULTS_ROOT_DIR}/both_finetuned_eval"
MTP_ONLY_FINETUNED_DIR="${RESULTS_ROOT_DIR}/mtp_only_finetuned"
MTP_ONLY_EVAL_DIR="${RESULTS_ROOT_DIR}/mtp_only_eval"

# Subfolder & Result Filename Constants
TARGET_LORA_DIR_NAME="target_lora"
ASSISTANT_LORA_DIR_NAME="assistant_lora"
EVAL_RESULTS_FILE_NAME="eval_results.json"

# Hyperparameters & Run Options
NUM_EVAL_SAMPLES=500
EPOCHS=3
MAX_STEPS=-1
BATCH_SIZE=1
LR=2e-4

# MTP Loss Weights (Applied in step 4 joint training)
MTP_LOSS_WEIGHT=0.4
MTP_DISTILLATION_WEIGHT=1.0

# MTP-Only Loss Weight (Applied in step 6 where target backbone is detached/frozen)
MTP_ONLY_LOSS_WEIGHT=1.0

# Select which steps to run (comma-separated, e.g., "1,2,3,4,5,6,7")
# Can be overridden by passing as the first argument: ./run_pipeline.sh "1,2"
STEPS=${1:-"1,2,3,4,5,6,7"}

echo "================================================================="
echo "Starting Gemma 4 MTP Speculative Decoding Experiment Pipeline"
echo "================================================================="
echo "Steps to run:        $STEPS"
echo "Target Model:        $MODEL_NAME"
echo "Assistant Model:     $ASSISTANT_NAME"
echo "Dataset:             $DATASET"
echo "Eval Samples:        $NUM_EVAL_SAMPLES"
echo "Training Epochs:     $EPOCHS"
echo "Max Steps:           $MAX_STEPS"
echo "Batch Size:          $BATCH_SIZE"
echo "Learning Rate:       $LR"
echo "================================================================="

# Create results directory structure
mkdir -p "$RESULTS_ROOT_DIR"

# Step 1: Baseline Evaluation
if [[ ",$STEPS," == *",1,"* ]]; then
  echo "--- Running Step 1: Baseline Evaluation ---"
  python3 run_experiments.py \
    --mode eval \
    --model_name "$MODEL_NAME" \
    --assistant_model_name "$ASSISTANT_NAME" \
    --dataset_path "$DATASET" \
    --num_samples "$NUM_EVAL_SAMPLES" \
    --target_lora_dir "$TARGET_LORA_DIR_NAME" \
    --assistant_lora_dir "$ASSISTANT_LORA_DIR_NAME" \
    --eval_results_file "$EVAL_RESULTS_FILE_NAME" \
    --output_dir "$BASELINE_DIR"
fi

# Step 2: Finetune Target Model Only
if [[ ",$STEPS," == *",2,"* ]]; then
  echo "--- Running Step 2: Finetuning Target Model Only (SFT) ---"
  python3 run_experiments.py \
    --mode train \
    --train_mode target_only \
    --model_name "$MODEL_NAME" \
    --assistant_model_name "$ASSISTANT_NAME" \
    --dataset_path "$DATASET" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --target_lora_dir "$TARGET_LORA_DIR_NAME" \
    --assistant_lora_dir "$ASSISTANT_LORA_DIR_NAME" \
    --eval_results_file "$EVAL_RESULTS_FILE_NAME" \
    --output_dir "$TARGET_FINETUNED_DIR"
fi

# Step 3: Evaluate Finetuned Target Model
if [[ ",$STEPS," == *",3,"* ]]; then
  echo "--- Running Step 3: Evaluating Finetuned Target Model ---"
  python3 run_experiments.py \
    --mode eval \
    --model_name "$MODEL_NAME" \
    --assistant_model_name "$ASSISTANT_NAME" \
    --dataset_path "$DATASET" \
    --adapter_path "$TARGET_FINETUNED_DIR" \
    --target_lora_dir "$TARGET_LORA_DIR_NAME" \
    --assistant_lora_dir "$ASSISTANT_LORA_DIR_NAME" \
    --eval_results_file "$EVAL_RESULTS_FILE_NAME" \
    --num_samples "$NUM_EVAL_SAMPLES" \
    --output_dir "$TARGET_EVAL_DIR"
fi

# Step 4: Finetune Both Target and Assistant Model
if [[ ",$STEPS," == *",4,"* ]]; then
  echo "--- Running Step 4: Joint Finetuning of Target & Assistant (MTP Distillation) ---"
  python3 run_experiments.py \
    --mode train \
    --train_mode both \
    --model_name "$MODEL_NAME" \
    --assistant_model_name "$ASSISTANT_NAME" \
    --dataset_path "$DATASET" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --target_lora_dir "$TARGET_LORA_DIR_NAME" \
    --assistant_lora_dir "$ASSISTANT_LORA_DIR_NAME" \
    --eval_results_file "$EVAL_RESULTS_FILE_NAME" \
    --mtp_loss_weight "$MTP_LOSS_WEIGHT" \
    --mtp_distillation_weight "$MTP_DISTILLATION_WEIGHT" \
    --output_dir "$BOTH_FINETUNED_DIR"
fi

# Step 5: Evaluate Jointly Finetuned Model
if [[ ",$STEPS," == *",5,"* ]]; then
  echo "--- Running Step 5: Evaluating Jointly Finetuned Model ---"
  python3 run_experiments.py \
    --mode eval \
    --model_name "$MODEL_NAME" \
    --assistant_model_name "$ASSISTANT_NAME" \
    --dataset_path "$DATASET" \
    --adapter_path "$BOTH_FINETUNED_DIR" \
    --target_lora_dir "$TARGET_LORA_DIR_NAME" \
    --assistant_lora_dir "$ASSISTANT_LORA_DIR_NAME" \
    --eval_results_file "$EVAL_RESULTS_FILE_NAME" \
    --num_samples "$NUM_EVAL_SAMPLES" \
    --output_dir "$BOTH_EVAL_DIR"
fi

# Step 6: Finetune Assistant Model Only (Freezing Target Model with Step 2 LoRA)
if [[ ",$STEPS," == *",6,"* ]]; then
  echo "--- Running Step 6: Finetuning Assistant Model Only (MTP Only) ---"
  # In mtp_only mode, we load the fine-tuned Target LoRA from Step 2 and freeze the
  # target backbone. The Assistant LoRA will be saved to MTP_ONLY_FINETUNED_DIR.
  python3 run_experiments.py \
    --mode train \
    --train_mode mtp_only \
    --model_name "$MODEL_NAME" \
    --assistant_model_name "$ASSISTANT_NAME" \
    --dataset_path "$DATASET" \
    --adapter_path "${TARGET_FINETUNED_DIR}/${TARGET_LORA_DIR_NAME}" \
    --epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LR" \
    --target_lora_dir "$TARGET_LORA_DIR_NAME" \
    --assistant_lora_dir "$ASSISTANT_LORA_DIR_NAME" \
    --eval_results_file "$EVAL_RESULTS_FILE_NAME" \
    --mtp_loss_weight "$MTP_ONLY_LOSS_WEIGHT" \
    --mtp_distillation_weight "$MTP_DISTILLATION_WEIGHT" \
    --output_dir "$MTP_ONLY_FINETUNED_DIR"
fi

# Step 7: Evaluate MTP-Only Finetuned Model
if [[ ",$STEPS," == *",7,"* ]]; then
  echo "--- Running Step 7: Evaluating MTP-Only Finetuned Model ---"
  # Combine target and assistant LoRA adapters into a unified temporary directory for evaluation.
  EVAL_COMBINED_DIR="${MTP_ONLY_EVAL_DIR}/combined_adapters"
  mkdir -p "$EVAL_COMBINED_DIR"
  ln -sf "$(realpath ${TARGET_FINETUNED_DIR}/${TARGET_LORA_DIR_NAME})" "${EVAL_COMBINED_DIR}/${TARGET_LORA_DIR_NAME}"
  ln -sf "$(realpath ${MTP_ONLY_FINETUNED_DIR}/${ASSISTANT_LORA_DIR_NAME})" "${EVAL_COMBINED_DIR}/${ASSISTANT_LORA_DIR_NAME}"

  python3 run_experiments.py \
    --mode eval \
    --model_name "$MODEL_NAME" \
    --assistant_model_name "$ASSISTANT_NAME" \
    --dataset_path "$DATASET" \
    --adapter_path "$EVAL_COMBINED_DIR" \
    --target_lora_dir "$TARGET_LORA_DIR_NAME" \
    --assistant_lora_dir "$ASSISTANT_LORA_DIR_NAME" \
    --eval_results_file "$EVAL_RESULTS_FILE_NAME" \
    --num_samples "$NUM_EVAL_SAMPLES" \
    --output_dir "$MTP_ONLY_EVAL_DIR"
fi

echo "================================================================="
echo "Pipeline Completed! Summary of Results:"
echo "================================================================="

echo "Baseline:"
if [ -f "${BASELINE_DIR}/${EVAL_RESULTS_FILE_NAME}" ]; then
  cat "${BASELINE_DIR}/${EVAL_RESULTS_FILE_NAME}"
fi

echo "Finetuned Target Only:"
if [ -f "${TARGET_EVAL_DIR}/${EVAL_RESULTS_FILE_NAME}" ]; then
  cat "${TARGET_EVAL_DIR}/${EVAL_RESULTS_FILE_NAME}"
fi

echo "Jointly Finetuned Both:"
if [ -f "${BOTH_EVAL_DIR}/${EVAL_RESULTS_FILE_NAME}" ]; then
  cat "${BOTH_EVAL_DIR}/${EVAL_RESULTS_FILE_NAME}"
fi

echo "MTP-Only Finetuned:"
if [ -f "${MTP_ONLY_EVAL_DIR}/${EVAL_RESULTS_FILE_NAME}" ]; then
  cat "${MTP_ONLY_EVAL_DIR}/${EVAL_RESULTS_FILE_NAME}"
fi
