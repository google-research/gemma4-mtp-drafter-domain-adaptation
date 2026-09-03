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

"""Data processing and formatting utilities for instruction and chat datasets."""

import typing
import datasets


def load_and_format_dataset(
    dataset_path: str, split: str = "test"
) -> datasets.Dataset:
  """Loads a dataset (e.g., SPARQL, GSM8K, MBPP) and formats it for chat evaluation.

  Args:
    dataset_path: Hugging Face dataset identifier.
    split: Split name to load (e.g. 'test', 'train').

  Returns:
    Formatted Dataset instance containing messages column.
  """
  print(f"Loading dataset {dataset_path} for split {split}...")
  if "gsm8k" in dataset_path.lower():
    ds = datasets.load_dataset(dataset_path, "main", split=split)
  elif "orkg/SciQA" in dataset_path:
    ds = datasets.load_dataset(
        dataset_path, revision="refs/convert/parquet", split=split
    )
  else:
    ds = datasets.load_dataset(dataset_path, split=split)

  records = []
  for row in ds:
    if (
        "question" in row
        and isinstance(row["question"], dict)
        and "string" in row["question"]
    ):
      instruction = row["question"]["string"]
      answer = row["query"]["sparql"]
    elif "question" in row and "answer" in row:
      instruction = row["question"]
      answer = row["answer"]
    elif "text" in row and "code" in row:
      instruction = row["text"]
      answer = row["code"]
    else:
      if "instructions" in row and row["instructions"]:
        instruction = row["instructions"][0]
      else:
        instruction = row.get("instruction", "")

      if not instruction:
        continue

      answer = row.get("sparql_query", row.get("query", ""))

    records.append({
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": answer},
        ]
    })
  return datasets.Dataset.from_list(records)


def preprocess_dataset(
    examples: dict[str, typing.Any],
    tokenizer: typing.Any,
    max_length: int = 2048,
) -> dict[str, list[list[int]]]:
  """Preprocesses and tokenizes dataset examples into input_ids, attention_mask, and labels.

  Args:
    examples: Batch of dataset examples.
    tokenizer: Tokenizer instance.
    max_length: Maximum sequence length for truncation.

  Returns:
    Dictionary containing input_ids, attention_mask, and labels lists.
  """
  prompts = examples.get(
      "instructions", examples.get("question", examples.get("text", []))
  )
  completions = examples.get(
      "sparql_query",
      examples.get("answer", examples.get("code", examples.get("query", []))),
  )

  input_ids_list = []
  attention_mask_list = []
  labels_list = []

  for prompt, completion in zip(prompts, completions):
    instruction = (
        prompt["string"]
        if isinstance(prompt, dict) and "string" in prompt
        else (
            prompt[0] if (isinstance(prompt, list) and prompt) else str(prompt)
        )
    )
    answer_text = (
        completion["sparql"]
        if isinstance(completion, dict) and "sparql" in completion
        else str(completion)
    )
    user_messages = [{"role": "user", "content": instruction}]
    full_messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": answer_text},
    ]

    prompt_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_length:
      full_ids = full_ids[:max_length]

    # Mask user prompt tokens in labels with -100
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
      labels[i] = -100

    input_ids_list.append(full_ids)
    attention_mask_list.append([1] * len(full_ids))
    labels_list.append(labels)

  return {
      "input_ids": input_ids_list,
      "attention_mask": attention_mask_list,
      "labels": labels_list,
  }
