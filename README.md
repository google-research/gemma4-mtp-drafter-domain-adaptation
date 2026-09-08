# Gemma 4 MTP Speculative Decoding Experiments Guide

This guide details the methodology, code design, and execution instructions for
the Gemma 4 Multi-Token Prediction (MTP) drafter experiments on instruction and
chat datasets (e.g., `PaDaS-Lab/Instruct-to-SPARQL`, `GSM8K`, `SciQA`) as outlined in the `Efficient and Robust Drafting with Multi-Token Prediction` paper.

## Experiment Goals

1.  **Baseline**: Measure next-token prediction (NTP) accuracy of
    `google/gemma-4-E4B-it` and speculative decoding block efficiency using
    `google/gemma-4-E4B-it-assistant` (drafter) on the dataset.
2.  **Finetune Target**: Finetune the target model on the dataset, then measure
    how it affects accuracy and block efficiency (with the frozen baseline
    drafter).
3.  **Finetune Both**: Finetune *both* target and assistant models jointly on
    the dataset, then measure accuracy and block efficiency.
4.  **MTP Only (Distill Only)**: Freeze the fine-tuned target model (from
    Step 2) and train *only* the assistant model using Probability Distillation
    to align it with the custom vocabulary space.

--------------------------------------------------------------------------------

## How to Run

### 1. Setup Environment

On a GPU-enabled machine, install the required open-source dependencies:

```bash
pip install -U torch transformers datasets peft trl accelerate tqdm numpy
```

Ensure your Hugging Face CLI is authenticated to download Google Gemma models:

```bash
huggingface-cli login
```

### 2. Execution Options

#### Option A: Run the Full Pipeline Sequentially

The easiest way is to trigger the automated shell script:

```bash
cd gemma4_mtp_experiment/
./run_pipeline.sh
```

You can also select specific steps to run by passing them as a comma-separated
argument:

```bash
./run_pipeline.sh "1,2"
```

**Steps Definition:**

*   **Step 1**: Baseline Evaluation.
*   **Step 2**: Finetune Target Model (SFT).
*   **Step 3**: Evaluate Finetuned Target Model.
*   **Step 4**: Joint Finetuning of Target & Assistant.
*   **Step 5**: Evaluate Jointly Finetuned Model.
*   **Step 6**: Finetune Assistant Model Only (MTP Only/Distill Only).
*   **Step 7**: Evaluate MTP-Only Finetuned Model.

#### Option B: Run Specific Tasks Manually

*   **MTP Only Finetune (Drafter Only)**:

    ```bash
    python3 run_experiments.py \
      --mode train \
      --train_mode mtp_only \
      --model_name google/gemma-4-E4B-it \
      --assistant_model_name google/gemma-4-E4B-it-assistant \
      --dataset_path PaDaS-Lab/Instruct-to-SPARQL \
      --adapter_path ./results/target_finetuned/target_lora \
      --epochs 3 \
      --mtp_loss_weight 1.0 \
      --mtp_distillation_weight 1.0 \
      --output_dir ./results/mtp_only_finetuned
    ```

*   **Evaluate MTP-Only Model** (Requires combined adapter symlink or manual
    merge):

    ```bash
    # See run_pipeline.sh Step 7 for how to combine adapters for eval
    python3 run_experiments.py \
      --mode eval \
      --model_name google/gemma-4-E4B-it \
      --assistant_model_name google/gemma-4-E4B-it-assistant \
      --adapter_path ./results/mtp_only_eval/combined_adapters \
      --num_samples 100 \
      --output_dir ./results/mtp_only_eval
    ```

## Disclaimer

This project is intended for demonstration purposes only. It is not
intended for use in a production environment.

  Eligibility for the [Google Open Source Software Vulnerability Rewards
  Program](https://bughunters.google.com/open-source-security) is determined by the [Google Open Source Software Vulnerability Reward Program Rules](https://bughunters.google.com/about/rules/open-source/google-open-source-software-vulnerability-reward-program-rules).
