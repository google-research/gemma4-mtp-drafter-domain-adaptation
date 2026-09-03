# Gemma 4 MTP Speculative Decoding Experiments Guide

This guide details the methodology, code design, and execution instructions for
the Gemma 4 Multi-Token Prediction (MTP) drafter experiments on instruction and
chat datasets (e.g., `PaDaS-Lab/Instruct-to-SPARQL`, `GSM8K`, `MBPP`).

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

## Technical Design & Methodology

### 1. Speculative Decoding Metric Logging & Metrics

To evaluate speculative decoding performance, we hook into Hugging Face
`transformers` generation engine (`AssistedCandidateGenerator` and
`Gemma4CandidateGenerator`). The script dynamically patches the
`update_candidate_strategy` method of both classes to capture:

-   **`num_matches`**: The number of draft tokens validated and accepted by the
    target model in a step.
-   **`candidate_length`**: The number of draft tokens proposed by the assistant
    in that step.

Using these logs, we compute two key speculative metrics:

1.  **Acceptance Rate**: The fraction of proposed draft tokens accepted by the
    target model: <!-- disableFinding(LINE_OVER_80) --> $$\text{Acceptance
    Rate} = \frac{\sum \text{Accepted Tokens}}{\sum \text{Proposed Tokens}}$$

2.  **Block Efficiency (Tokens / Step)**: The average total generated tokens
    produced per speculative decoding step (the base token plus average accepted
    draft tokens): <!-- disableFinding(LINE_OVER_80) --> $$\text{Block
    Efficiency} = 1 + \frac{\sum \text{Accepted Tokens}}{\text{Total Speculative
    Steps}}$$

### 2. MTP Joint Loss (Target + Drafter)

During joint training (mode `both`) or MTP-only training (mode `mtp_only`), we
wrap the model with `Gemma4TrainingModel` and train with a combined loss
function:

<!-- disableFinding(LINE_OVER_80) -->

$$Loss_{\text{joint}} = Loss_{\text{backbone\_LM}} + \lambda_{\text{MTP}} \sum_{k=1}^{K} Loss_{\text{head\_}k}$$

For each speculative head $k$, the head loss is a weighted sum of causal
language modeling (LM) loss and Probability Distillation Total Variation
Distance (TVD) loss:

<!-- disableFinding(LINE_OVER_80) -->

$$Loss_{\text{head\_}k} = (1 - w_{\text{distill}}) Loss_{\text{LM\_}k} + w_{\text{distill}} Loss_{\text{TVD\_}k}$$

*   **TVD Loss**: Measures the difference in probability distributions output by
    the speculative head and the corresponding future step logits from the
    detached backbone (acting as a teacher).
    <!-- disableFinding(LINE_OVER_80) -->
    $$Loss_{\text{TVD\_}k} = \frac{1}{2} \sum_{v \in V} | P_{\text{head\_}k}(v) - P_{\text{backbone\_}k}(v) |$$

#### Hyperparameters:

*   $\lambda_{\text{MTP}}$ (`--mtp_loss_weight`): Controls the overall weight of
    the MTP loss component.
*   $w_{\text{distill}}$ (`--mtp_distillation_weight`): Controls the balance
    between LM loss (NTP) and Distillation (TVD) within the MTP loss. Setting
    this to `1.0` turns on **Distillation Only** mode.

### 3. Parameter Allocation & Freezing

-   **Target Model (`target_only` mode)**: Tuned using PEFT LoRA adapters
    targeting `q_proj`, `v_proj`, `k_proj`, `o_proj`.
-   **Joint Tuning (`both` mode)**: PEFT LoRA adapters are injected into
    `q_proj`, `k_proj`, `v_proj`, `o_proj` layers across the target backbone,
    and expanded targets (`q_proj`, `o_proj`, `gate_proj`, `up_proj`,
    `down_proj`, `pre_projection`, `post_projection`, `lm_head`) for the
    assistant. Both models train simultaneously.
-   **MTP Only (`mtp_only` mode)**: Loads the fine-tuned Target LoRA (from Step
    2). The Target Backbone is **frozen** (`requires_grad=False`) and set to
    `eval()` mode to ensure deterministic teacher outputs (no dropout noise).
    Only the Assistant Model's LoRAs are trained.

| Mode          | Target Backbone | Target LoRA   | Assistant     | MTP Loss |
:               :                 :               : LoRA          :          :
| :------------ | :-------------: | :-----------: | :-----------: | :------: |
| `target_only` | Frozen          | **Trainable** | *Not Loaded*  | No       |
| `both`        | Frozen          | **Trainable** | **Trainable** | Yes      |
| `mtp_only`    | **Frozen (Eval  | *Frozen (from | **Trainable** | Yes      |
:               : Mode)**         : Step 2)*      :               :          :

--------------------------------------------------------------------------------

## Code Structure

All necessary scripts are located in this directory:

-   **[training_gemma4.py](training_gemma4.py)**: Standalone modules containing
    `Gemma4LossFunction`, `Gemma4TrainingModel`, and `Gemma4LoRATrainer`.
-   **[eval_utils.py](eval_utils.py)**: Evaluation utilities
    (`compute_ntp_accuracy`, `compute_block_efficiency`), model unwrapping, and
    candidate generator monkeypatches.
-   **[data_utils.py](data_utils.py)**: Data loading and chat formatting
    utilities for instruction and chat datasets (e.g., `SPARQL`, `GSM8K`,
    `MBPP`).
-   **[run_experiments.py](run_experiments.py)**: Main entry point orchestrating
    training and evaluation.
-   **[run_pipeline.sh](run_pipeline.sh)**: Shell automation script to run all
    training and evaluation experiments sequentially (divided into 7 steps).

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
