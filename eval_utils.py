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

"""Evaluation utilities for Gemma-4 speculative decoding and NTP accuracy.

Contains model unwrapping, speculative candidate strategy monkeypatches,
NTP accuracy calculation, and block efficiency measurement.
"""

import typing
import numpy as np
import torch
import tqdm
from transformers.generation import candidate_generator

# Globals for speculative metrics tracking
matches_log = []
original_update_assisted = (
    candidate_generator.AssistedCandidateGenerator.update_candidate_strategy
)


def patched_update_assisted(
    self: typing.Any,
    input_ids: torch.Tensor,
    scores: torch.Tensor,
    num_matches: int | torch.Tensor,
) -> typing.Any:
  """Intercepts update_candidate_strategy on AssistedCandidateGenerator."""
  val = num_matches.item() if hasattr(num_matches, "item") else int(num_matches)
  candidate_length = scores.shape[1]
  if candidate_length > 1:
    matches_log.append((val, candidate_length))
  return original_update_assisted(self, input_ids, scores, num_matches)


candidate_generator.AssistedCandidateGenerator.update_candidate_strategy = (
    patched_update_assisted
)


def check_tokenizers_different(
    tokenizer: typing.Any, assistant_tokenizer: typing.Any
) -> bool:
  """Checks whether target and assistant tokenizers differ.

  Args:
    tokenizer: Target model tokenizer.
    assistant_tokenizer: Assistant model tokenizer.

  Returns:
    True if tokenizers differ in vocabulary or token IDs, False otherwise.
  """
  if tokenizer.vocab_size != assistant_tokenizer.vocab_size:
    return True
  test_tokens = ["hello", "world", " ", "\n", "1", "2"]
  for t in test_tokens:
    if tokenizer.encode(
        t, add_special_tokens=False
    ) != assistant_tokenizer.encode(t, add_special_tokens=False):
      return True
  return False


def compute_ntp_accuracy(
    model: torch.nn.Module,
    tokenizer: typing.Any,
    dataset: typing.Any,
    num_samples: int,
) -> dict[str, float | int]:
  """Computes next-token prediction (NTP) accuracy on evaluation dataset.

  Args:
    model: Model instance to evaluate.
    tokenizer: Tokenizer instance.
    dataset: Evaluation dataset containing messages.
    num_samples: Maximum number of samples to evaluate.

  Returns:
    A dictionary containing total_accuracy and mean_per_sample_accuracy.
  """
  total_correct = 0
  total_answer_tokens = 0
  per_sample_accuracies = []

  for i in tqdm.tqdm(
      range(min(num_samples, len(dataset))), desc="Evaluating NTP Accuracy"
  ):
    example = dataset[i]
    messages = example["messages"]

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    if not full_text.endswith(tokenizer.eos_token):
      full_text += tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    answer_start = min(len(prompt_ids), len(full_ids) - 1)
    num_answer_tokens = len(full_ids) - answer_start
    if num_answer_tokens <= 0:
      continue

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
      logits = model(input_ids=input_ids).logits[0]

    sample_correct = 0
    for t in range(answer_start, len(full_ids)):
      predicted = logits[t - 1].argmax(dim=-1).item()
      if predicted == full_ids[t]:
        sample_correct += 1

    sample_acc = sample_correct / num_answer_tokens
    per_sample_accuracies.append(sample_acc)
    total_correct += sample_correct
    total_answer_tokens += num_answer_tokens

  total_accuracy = (
      total_correct / total_answer_tokens if total_answer_tokens > 0 else 0.0
  )
  mean_per_sample = (
      float(np.mean(per_sample_accuracies)) if per_sample_accuracies else 0.0
  )

  return {
      "total_accuracy": total_accuracy,
      "mean_per_sample_accuracy": mean_per_sample,
      "total_answer_tokens": total_answer_tokens,
      "num_samples": len(per_sample_accuracies),
  }


def compute_block_efficiency(
    model: torch.nn.Module,
    assistant_model: torch.nn.Module,
    tokenizer: typing.Any,
    assistant_tokenizer: typing.Any,
    dataset: typing.Any,
    num_samples: int,
    max_new_tokens: int = 64,
) -> dict[str, float | int]:
  """Computes speculative decoding block efficiency on evaluation dataset.

  Args:
    model: Target model.
    assistant_model: Assistant (drafter) model.
    tokenizer: Target model tokenizer.
    assistant_tokenizer: Assistant model tokenizer.
    dataset: Evaluation dataset.
    num_samples: Number of samples to evaluate.
    max_new_tokens: Maximum number of tokens to generate per sample.

  Returns:
    A dictionary containing block_efficiency, avg_accepted_tokens, and
    avg_proposed_tokens.
  """
  diff_tokenizers = check_tokenizers_different(tokenizer, assistant_tokenizer)
  print(f"Tokenizers different: {diff_tokenizers}")
  gen_kwargs = {
      "assistant_model": assistant_model,
      "max_new_tokens": max_new_tokens,
      "do_sample": False,
  }
  if diff_tokenizers:
    gen_kwargs["tokenizer"] = tokenizer
    gen_kwargs["assistant_tokenizer"] = assistant_tokenizer

  matches_log.clear()

  for i in tqdm.tqdm(
      range(min(num_samples, len(dataset))), desc="Evaluating Block Efficiency"
  ):
    example = dataset[i]
    messages = example["messages"]

    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
      try:
        model.generate(**inputs, **gen_kwargs)
      except (RuntimeError, ValueError, AttributeError) as e:
        print(f"Error during generation for sample {i}: {e}")
        continue

  # Calculate metrics from logs
  total_accepted = sum(m[0] for m in matches_log)
  total_proposed = sum(m[1] for m in matches_log)
  total_steps = len(matches_log)

  acceptance_rate = (
      total_accepted / total_proposed if total_proposed > 0 else 0.0
  )
  block_efficiency = (
      1 + (total_accepted / total_steps) if total_steps > 0 else 1.0
  )
  avg_proposed_per_step = (
      total_proposed / total_steps if total_steps > 0 else 0.0
  )
  avg_accepted_per_step = (
      total_accepted / total_steps if total_steps > 0 else 0.0
  )

  return {
      "block_efficiency": block_efficiency,
      "acceptance_rate": acceptance_rate,
      "avg_accepted_tokens": avg_accepted_per_step,
      "avg_proposed_tokens": avg_proposed_per_step,
      "total_steps": total_steps,
  }
