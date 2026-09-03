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

"""Model loading and saving utilities for Gemma-4 MTP experiments."""

import os
import typing
import peft
import torch
import transformers


class PatchedClippableLinear(torch.nn.Linear):
  """PEFT-compatible nn.Linear wrapper for Gemma4ClippableLinear."""

  def __init__(
      self,
      config: typing.Any,
      in_features: int,
      out_features: int,
      bias: bool = False,
  ):
    super().__init__(in_features, out_features, bias=bias)
    self.use_clipped_linears = getattr(config, "use_clipped_linears", False)
    if self.use_clipped_linears:
      self.register_buffer("input_min", torch.tensor(-float("inf")))
      self.register_buffer("input_max", torch.tensor(float("inf")))
      self.register_buffer("output_min", torch.tensor(-float("inf")))
      self.register_buffer("output_max", torch.tensor(float("inf")))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.use_clipped_linears:
      x = torch.clamp(x, self.input_min, self.input_max)
    out = super().forward(x)
    if self.use_clipped_linears:
      out = torch.clamp(out, self.output_min, self.output_max)
    return out


def patch_gemma4_clippable_linear() -> None:
  """Patches Gemma4ClippableLinear in transformers for PEFT compatibility."""
  try:
    from transformers.models.gemma4 import modeling_gemma4  # pylint: disable=g-import-not-at-top

    modeling_gemma4.Gemma4ClippableLinear = PatchedClippableLinear
  except (ImportError, AttributeError):
    pass


# Auto-patch on import
patch_gemma4_clippable_linear()

attn_impl = "sdpa"
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
  try:
    import flash_attn  # pylint: disable=unused-import,g-import-not-at-top

    attn_impl = "flash_attention_2"
  except ImportError:
    attn_impl = "sdpa"


def load_target_model(
    model_name: str,
    adapter_path: str | None = None,
    device_map: str = "auto",
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
  """Loads target model (base or with LoRA adapter applied).

  Args:
    model_name: Hugging Face model identifier for the base model.
    adapter_path: Path to LoRA adapter directory (optional).
    device_map: Device mapping strategy.

  Returns:
    Tuple of (target_model, tokenizer).

  Raises:
    FileNotFoundError: If adapter_path is specified and does not exist.
  """
  print(f"Loading target tokenizer: {model_name}...")
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_name, trust_remote_code=True
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  print(f"Loading target model ({model_name}) using {attn_impl} attention...")
  target_model = transformers.AutoModelForCausalLM.from_pretrained(
      model_name,
      dtype=torch.bfloat16,
      attn_implementation=attn_impl,
      device_map=device_map,
      trust_remote_code=True,
  )

  if adapter_path:
    if not os.path.exists(adapter_path):
      raise FileNotFoundError(
          f"Specified target LoRA adapter path does not exist: {adapter_path}"
      )
    abs_adapter_path = os.path.abspath(adapter_path)
    print(f"Applying target LoRA adapter from {abs_adapter_path}...")
    target_model = peft.PeftModel.from_pretrained(
        target_model, abs_adapter_path
    )
    # Merge LoRA weights so the model correctly routes through HuggingFace's
    # custom generation pipeline.
    target_model = target_model.merge_and_unload()

  return target_model, tokenizer


def load_assistant_model(
    assistant_model_name: str,
    target_model: torch.nn.Module | None = None,
    adapter_path: str | None = None,
    device_map: str = "auto",
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
  """Loads assistant (drafter) model and shares backbone embeddings with target model.

  Args:
    assistant_model_name: Hugging Face model identifier for assistant model.
    target_model: Target model instance to share embeddings from (optional).
    adapter_path: Path to assistant LoRA adapter directory (optional).
    device_map: Device mapping strategy.

  Returns:
    Tuple of (assistant_model, assistant_tokenizer).

  Raises:
    FileNotFoundError: If adapter_path is specified and does not exist.
  """
  print(f"Loading assistant tokenizer: {assistant_model_name}...")
  assistant_tokenizer = transformers.AutoTokenizer.from_pretrained(
      assistant_model_name, trust_remote_code=True
  )
  if assistant_tokenizer.pad_token is None:
    assistant_tokenizer.pad_token = assistant_tokenizer.eos_token

  print(
      f"Loading assistant model ({assistant_model_name}) using"
      f" {attn_impl} attention..."
  )
  assistant_model = transformers.AutoModelForCausalLM.from_pretrained(
      assistant_model_name,
      dtype=torch.bfloat16,
      attn_implementation=attn_impl,
      device_map=device_map,
      trust_remote_code=True,
  )

  if target_model is not None and hasattr(
      assistant_model, "set_backbone_embeddings"
  ):
    base_target = (
        target_model.base_model.model
        if hasattr(target_model, "base_model")
        else target_model
    )
    if hasattr(base_target, "get_input_embeddings"):
      assistant_model.set_backbone_embeddings(
          base_target.get_input_embeddings()
      )

  if adapter_path:
    if not os.path.exists(adapter_path):
      raise FileNotFoundError(
          "Specified assistant LoRA adapter path does not exist:"
          f" {adapter_path}"
      )
    abs_adapter_path = os.path.abspath(adapter_path)
    print(f"Applying assistant LoRA adapter from {abs_adapter_path}...")
    assistant_model = peft.PeftModel.from_pretrained(
        assistant_model, abs_adapter_path
    )
    # Merge LoRA weights so the Drafter correctly routes through HuggingFace's
    # custom generation pipeline.
    assistant_model = assistant_model.merge_and_unload()

  return assistant_model, assistant_tokenizer


def load_combined_model(
    model_name: str,
    assistant_model_name: str | None = None,
    target_adapter_path: str | None = None,
    assistant_adapter_path: str | None = None,
) -> tuple[
    transformers.PreTrainedModel,
    transformers.PreTrainedModel | None,
    transformers.PreTrainedTokenizerBase,
    transformers.PreTrainedTokenizerBase | None,
]:
  """Loads target and optional assistant models cleanly and independently.

  Args:
    model_name: Base target model name.
    assistant_model_name: Base assistant model name (optional).
    target_adapter_path: Path to target model LoRA adapter (optional).
    assistant_adapter_path: Path to assistant model LoRA adapter (optional).

  Returns:
    Tuple of (target_model, assistant_model, tokenizer, assistant_tokenizer).
  """
  target_model, tokenizer = load_target_model(
      model_name, adapter_path=target_adapter_path
  )

  assistant_model, assistant_tokenizer = None, None
  if assistant_model_name:
    assistant_model, assistant_tokenizer = load_assistant_model(
        assistant_model_name,
        target_model=target_model,
        adapter_path=assistant_adapter_path,
    )

  target_model.eval()
  if assistant_model is not None:
    assistant_model.eval()

  return target_model, assistant_model, tokenizer, assistant_tokenizer


def save_lora_adapter(
    model: torch.nn.Module,
    save_dir: str,
    tokenizer: transformers.PreTrainedTokenizerBase | None = None,
) -> str:
  """Saves LoRA adapter using PEFT's built-in save_pretrained method.

  Args:
    model: PEFT model instance (or wrapped PeftModel).
    save_dir: Destination directory.
    tokenizer: Tokenizer instance to save alongside adapter (optional).

  Returns:
    Path to destination directory.
  """
  abs_save_dir = os.path.abspath(save_dir)
  print(f"Saving LoRA adapter to {abs_save_dir}...")

  # Use unwrapped model if inside DDP/Distributed wrapper
  if hasattr(model, "module") and hasattr(model.module, "save_pretrained"):
    target_peft = model.module
  else:
    target_peft = model

  # Native PEFT built-in library method
  target_peft.save_pretrained(abs_save_dir, safe_serialization=False)

  if tokenizer is not None:
    tokenizer.save_pretrained(abs_save_dir)

  print(f"LoRA adapter successfully saved to {abs_save_dir}")
  return abs_save_dir
