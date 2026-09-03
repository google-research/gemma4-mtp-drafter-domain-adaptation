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

"""Unit tests for model_utils.py."""

import os
import tempfile
import unittest
import model_utils
import peft
import torch


class TestModelUtils(unittest.TestCase):
  """Unit tests for model_utils module functions and adapter workflows."""

  def test_load_target_and_assistant_models(self) -> None:
    """Verifies baseline load_combined_model loads clean models."""
    print("Testing baseline load_combined_model...")
    (
        target_model,
        assistant_model,
        tokenizer,
        assistant_tokenizer,
    ) = model_utils.load_combined_model(
        "google/gemma-4-E4B-it", "google/gemma-4-E4B-it-assistant"
    )

    self.assertIsNotNone(target_model)
    self.assertIsNotNone(assistant_model)
    self.assertIsNotNone(tokenizer)
    self.assertIsNotNone(assistant_tokenizer)
    self.assertFalse(hasattr(target_model, "peft_config"))
    self.assertFalse(hasattr(assistant_model, "peft_config"))

  def test_save_and_load_lora_adapter(self) -> None:
    """Verifies saving and loading LoRA adapters cleanly."""
    print("Testing save_lora_adapter and loading...")
    target_model, tokenizer = model_utils.load_target_model(
        "google/gemma-4-E4B-it"
    )

    peft_config = peft.LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    peft_target = peft.get_peft_model(target_model, peft_config)

    with tempfile.TemporaryDirectory() as temp_dir:
      saved_dir = model_utils.save_lora_adapter(
          peft_target, temp_dir, tokenizer=tokenizer
      )

      self.assertTrue(
          os.path.exists(os.path.join(saved_dir, "adapter_config.json"))
      )
      self.assertTrue(
          os.path.exists(os.path.join(saved_dir, "adapter_model.bin"))
      )

      # Reload adapted model
      reloaded_target, _ = model_utils.load_target_model(
          "google/gemma-4-E4B-it", adapter_path=saved_dir
      )
      self.assertTrue(hasattr(reloaded_target, "peft_config"))

  def test_train_save_reload_differs_from_vanilla(self) -> None:
    """Verifies training LoRA on predefined data produces different logits than vanilla."""
    print("Testing training, saving, and verifying divergence from vanilla...")
    model_name = "google/gemma-4-E4B-it"

    # 1. Get vanilla logits on test prompt
    vanilla_model, tokenizer = model_utils.load_target_model(model_name)
    test_text = (
        "<start_of_turn>user\nFind all instances of class"
        " Country.<end_of_turn>\n<start_of_turn>model\n"
    )
    inputs = tokenizer(test_text, return_tensors="pt").to(vanilla_model.device)

    with torch.no_grad():
      vanilla_logits = vanilla_model(**inputs).logits[0, -1, :]

    # 2. Add LoRA config and train for 5 steps on predefined data
    peft_config = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    lora_model = peft.get_peft_model(vanilla_model, peft_config)
    lora_model.train()

    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad], lr=5e-4
    )

    train_text = test_text + "SELECT ?s WHERE { ?s a <Country> }<end_of_turn>"
    train_inputs = tokenizer(train_text, return_tensors="pt").to(
        lora_model.device
    )
    train_inputs["labels"] = train_inputs["input_ids"].clone()

    for _ in range(5):
      optimizer.zero_grad()
      outputs = lora_model(**train_inputs)
      loss = outputs.loss
      loss.backward()
      optimizer.step()

    # 3. Save LoRA adapter
    with tempfile.TemporaryDirectory() as temp_dir:
      saved_dir = model_utils.save_lora_adapter(
          lora_model, temp_dir, tokenizer=tokenizer
      )

      # 4. Reload adapted model
      reloaded_model, _ = model_utils.load_target_model(
          model_name, adapter_path=saved_dir
      )
      reloaded_model.eval()

      with torch.no_grad():
        adapted_logits = reloaded_model(**inputs).logits[0, -1, :]

      # 5. Assert logits differ between vanilla and reloaded adapted model
      max_diff = torch.max(torch.abs(adapted_logits - vanilla_logits)).item()
      print(
          "Max logit difference between Vanilla and Adapted model:"
          f" {max_diff:.6f}"
      )
      self.assertGreater(
          max_diff,
          1e-3,
          "Adapted model logits did not differ from vanilla model"
          f" (max_diff={max_diff})",
      )


if __name__ == "__main__":
  unittest.main()
