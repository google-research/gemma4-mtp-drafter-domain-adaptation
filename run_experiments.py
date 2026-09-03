#!/usr/bin/env python3
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

"""Unified Gemma-4 MTP Speculative Decoding Experiments Script.

Performs:
1. Baseline NTP Accuracy and Speculative Decoding Block Efficiency evaluation.
2. Fine-tuning of target model (SFT LoRA).
3. Fine-tuning of target + assistant model (speculative heads) using MTP loss.
"""

import argparse
import json
import os
import pathlib
import data_utils
import datasets
import eval_utils
import model_utils
import peft
import training_gemma4
import transformers
import trl

# -------------------------------------------------------------
# Training Orchestrator
# -------------------------------------------------------------


def run_training(args: argparse.Namespace) -> None:
  """Orchestrates LoRA or joint MTP training for Gemma 4.

  Args:
    args: Command-line arguments namespace.
  """
  # 1. Load data
  if "gsm8k" in args.dataset_path.lower():
    dataset = datasets.load_dataset(args.dataset_path, "main")
  elif "orkg/SciQA" in args.dataset_path:
    dataset = datasets.load_dataset(args.dataset_path, revision="refs/convert/parquet")
  else:
    dataset = datasets.load_dataset(args.dataset_path)

  if "train" not in dataset:
    available_split = list(dataset.keys())[0]
    print(f"No train split found. Splitting from {available_split}...")
    dataset = dataset[available_split].train_test_split(test_size=0.2, seed=42)
    dataset["validation"] = dataset["test"]

  # 2. Load base models
  if args.train_mode == "mtp_only":
    target_adapter_path = (
        args.adapter_path
        if args.adapter_path
        else os.path.join(args.output_dir, args.target_lora_dir)
    )
    print(
        "Loading pre-trained target LoRA for mtp_only from"
        f" {target_adapter_path}..."
    )
    model, tokenizer = model_utils.load_target_model(
        args.model_name, adapter_path=target_adapter_path
    )
  else:
    model, tokenizer = model_utils.load_target_model(args.model_name)
  assistant_model, assistant_tokenizer = None, None
  if args.assistant_model_name:
    assistant_model, assistant_tokenizer = model_utils.load_assistant_model(
        args.assistant_model_name, target_model=model
    )

  # 3. Apply PEFT LoRA configuration
  print("Setting up PEFT LoRA Config...")
  if args.train_mode != "mtp_only":
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = peft.get_peft_model(model, peft_config)

  if args.train_mode in ["both", "mtp_only"] and assistant_model is not None:
    assistant_target_modules = [
        "q_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "pre_projection",
        "post_projection",
        "lm_head",
    ]
    assistant_peft_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=assistant_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        ensure_weight_tying=True,
    )
    assistant_model = peft.get_peft_model(
        assistant_model, assistant_peft_config
    )

  print("Trainable parameters for target model:")
  try:
    model.print_trainable_parameters()
  except AttributeError:
    print("Target model cannot print trainable parameters.")

  if assistant_model is not None:
    print("Trainable parameters for assistant model:")
    try:
      assistant_model.print_trainable_parameters()
    except AttributeError:
      print("Assistant model cannot print trainable parameters.")
  else:
    print("Assistant model is not loaded.")

  # 4. Dataset Formatting & Tokenization
  print("Pre-processing and tokenizing training/eval datasets...")
  formatted_dataset = dataset.map(
      lambda ex: data_utils.preprocess_dataset(ex, tokenizer, max_length=512),
      batched=True,
      remove_columns=dataset["train"].column_names,
  )

  eval_split = "validation" if "validation" in formatted_dataset else "test"
  train_dataset = formatted_dataset["train"]
  eval_dataset = formatted_dataset[eval_split]
  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Eval dataset size: {len(eval_dataset)}")

  # 5. Setup training wrapper and arguments
  training_args = trl.SFTConfig(
      output_dir=args.output_dir,
      per_device_train_batch_size=args.batch_size,
      gradient_accumulation_steps=args.gradient_accumulation_steps,
      learning_rate=args.learning_rate,
      logging_steps=10,
      save_strategy="no",
      eval_strategy="no",
      num_train_epochs=args.epochs,
      max_steps=args.max_steps,
      weight_decay=0.01,
      warmup_steps=10,
      bf16=True,
      gradient_checkpointing=True,
      gradient_checkpointing_kwargs={"use_reentrant": False},
      dataloader_num_workers=4,
      dataloader_pin_memory=True,
      report_to="none",
      remove_unused_columns=False,
      max_length=512,
      max_grad_norm=0.3,
  )

  data_collator = transformers.DataCollatorForSeq2Seq(
      tokenizer=tokenizer,
      pad_to_multiple_of=8,
      return_tensors="pt",
  )

  include_mtp_loss = args.train_mode in ["both", "mtp_only"]
  training_wrapper = training_gemma4.Gemma4TrainingModel(
      model=model,
      assistant_model=assistant_model,
      include_mtp_loss=include_mtp_loss,
      detach_mtp_inputs=not getattr(args, "disable_mtp_detaching", False),
      freeze_backbone=(args.train_mode == "mtp_only"),
      mtp_loss_weight=args.mtp_loss_weight,
      mtp_distillation_weight=args.mtp_distillation_weight,
  )

  trainer = training_gemma4.Gemma4LoRATrainer(
      model=training_wrapper,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      data_collator=data_collator,
      processing_class=tokenizer,
      include_mtp_loss=include_mtp_loss,
  )

  print("Starting Training...")
  trainer.train()

  # 6. Save final LoRA adapters independently
  if args.train_mode != "mtp_only":
    target_save_dir = os.path.join(args.output_dir, args.target_lora_dir)
    model_utils.save_lora_adapter(model, target_save_dir, tokenizer=tokenizer)

  if args.train_mode in ["both", "mtp_only"] and assistant_model is not None:
    assistant_save_dir = os.path.join(args.output_dir, args.assistant_lora_dir)
    model_utils.save_lora_adapter(
        assistant_model, assistant_save_dir, tokenizer=assistant_tokenizer
    )

  print("Training Finished Successfully.")


# -------------------------------------------------------------
# Evaluation Orchestrator
# -------------------------------------------------------------


def run_eval(args: argparse.Namespace) -> None:
  """Orchestrates NTP accuracy and Speculative Decoding evaluation for Gemma 4.

  Args:
    args: Command-line arguments namespace.

  Raises:
    FileNotFoundError: If adapter_path is specified and does not exist.
  """
  # 1. Load dataset
  eval_dataset = data_utils.load_and_format_dataset(
      args.dataset_path, split="test"
  )

  # 2. Resolve adapter paths
  target_adapter_path = None
  assistant_adapter_path = None

  if args.adapter_path:
    abs_adapter = os.path.abspath(args.adapter_path)
    sub_target = os.path.join(abs_adapter, args.target_lora_dir)
    sub_assistant = os.path.join(abs_adapter, args.assistant_lora_dir)

    if os.path.exists(sub_target):
      target_adapter_path = sub_target
      if os.path.exists(sub_assistant):
        assistant_adapter_path = sub_assistant
    elif os.path.exists(abs_adapter):
      target_adapter_path = abs_adapter
    else:
      raise FileNotFoundError(
          f"Specified adapter_path does not exist: {args.adapter_path}"
      )

  # 3. Load combined model
  (
      model,
      assistant_model,
      tokenizer,
      assistant_tokenizer,
  ) = model_utils.load_combined_model(
      args.model_name,
      args.assistant_model_name,
      target_adapter_path=target_adapter_path,
      assistant_adapter_path=assistant_adapter_path,
  )

  results = {}

  # NTP accuracy
  print("Evaluating NTP Accuracy...")
  ntp_results = eval_utils.compute_ntp_accuracy(
      model, tokenizer, eval_dataset, args.num_samples
  )
  results["ntp"] = ntp_results
  print(f"Total NTP Accuracy: {ntp_results['total_accuracy']:.4f}")

  # Block efficiency
  if assistant_model is not None:
    print("Evaluating Block Efficiency...")
    be_results = eval_utils.compute_block_efficiency(
        model,
        assistant_model,
        tokenizer,
        assistant_tokenizer,
        eval_dataset,
        args.num_samples,
    )
    results["be"] = be_results
    print(be_results)
    print(f"Block Efficiency: {be_results['block_efficiency']:.2f}")
    print(f"Acceptance Rate: {be_results['acceptance_rate']:.2f}")
    print(f"Avg Proposed Tokens: {be_results['avg_proposed_tokens']:.2f}")

  # Output results
  output_path = pathlib.Path(args.output_dir) / args.eval_results_file
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
  print(f"Results saved to {output_path}")


# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------


def main() -> None:
  """Parses command-line flags and launches training or evaluation."""
  parser = argparse.ArgumentParser(
      description="Gemma 4 MTP Drafter Experiments"
  )
  parser.add_argument(
      "--mode",
      type=str,
      required=True,
      choices=["train", "eval"],
      help="Run mode: train or eval",
  )
  parser.add_argument(
      "--train_mode",
      type=str,
      default="target_only",
      choices=["target_only", "both", "mtp_only"],
      help="Train target only, both, or mtp_only (freeze target)",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="google/gemma-4-E4B-it",
      help="Target model ID on Hugging Face",
  )
  parser.add_argument(
      "--assistant_model_name",
      type=str,
      default="google/gemma-4-E4B-it-assistant",
      help="Assistant (Drafter) model ID on Hugging Face",
  )
  parser.add_argument(
      "--dataset_path",
      type=str,
      default="PaDaS-Lab/Instruct-to-SPARQL",
      help="Hugging Face dataset identifier",
  )
  parser.add_argument(
      "--adapter_path",
      type=str,
      default=None,
      help="Path to saved LoRA adapters (for eval or resume train)",
  )
  parser.add_argument(
      "--target_lora_dir",
      type=str,
      default="target_lora",
      help="Subfolder name for saving/loading target LoRA adapter",
  )
  parser.add_argument(
      "--assistant_lora_dir",
      type=str,
      default="assistant_lora",
      help="Subfolder name for saving/loading assistant LoRA adapter",
  )
  parser.add_argument(
      "--eval_results_file",
      type=str,
      default="eval_results.json",
      help="Filename for saving evaluation results JSON",
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      default=100,
      help="Number of samples to run evaluation on",
  )
  parser.add_argument(
      "--epochs", type=int, default=3, help="Number of epochs for training"
  )
  parser.add_argument(
      "--max_steps",
      type=int,
      default=-1,
      help="Max training steps (-1 to train by epochs)",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=1,
      help="Batch size per device for training",
  )
  parser.add_argument(
      "--gradient_accumulation_steps",
      type=int,
      default=16,
      help="Number of gradient accumulation steps",
  )
  parser.add_argument(
      "--learning_rate", type=float, default=2e-5, help="Learning rate"
  )
  parser.add_argument(
      "--disable_mtp_detaching",
      action="store_true",
      help="Disable detachment of MTP inputs (defaults to True otherwise).",
  )
  parser.add_argument(
      "--mtp_loss_weight",
      type=float,
      default=0.1,
      help="Weight for overall MTP loss component in final loss.",
  )
  parser.add_argument(
      "--mtp_distillation_weight",
      type=float,
      default=0.9,
      help="Weight for distillation (TVD) vs LM loss inside MTP.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory to save metrics and adapters",
  )

  args = parser.parse_args()

  if args.mode == "train":
    run_training(args)
  elif args.mode == "eval":
    run_eval(args)


if __name__ == "__main__":
  main()
