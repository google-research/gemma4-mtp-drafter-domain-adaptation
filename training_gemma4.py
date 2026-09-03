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

"""Language modeling training modules for Gemma 4 Multi-Token Prediction (MTP).

This module contains the standalone Gemma4LossFunction and Gemma4TrainingModel
wrapper to compute next-token and MTP speculative lookahead loss in parallel.
"""

import collections
import typing
import torch
import torch.nn.functional as F
import transformers


class Gemma4LossFunction(torch.nn.Module):
  """Standalone PyTorch Loss module for Gemma4 speculative lookahead training.

  Attributes:
    mtp_distillation_weight: Weight for distillation loss.
    mtp_loss_weight: Weight for MTP loss.
  """

  def __init__(
      self,
      mtp_distillation_weight: float = 0.9,
      mtp_loss_weight: float = 0.1,
  ):
    super().__init__()
    self.mtp_distillation_weight = mtp_distillation_weight
    self.mtp_loss_weight = mtp_loss_weight

  def forward(
      self,
      backbone_logits: torch.Tensor,
      mtp_logits_all: torch.Tensor,
      labels: torch.Tensor,
      backbone_lm_loss: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """Computes the speculative next-token and TVD distillation loss.

    Args:
      backbone_logits: Backbone logits of shape [B, T, V].
      mtp_logits_all: Speculative lookahead logits of shape [K, B, T, V].
      labels: Ground truth labels of shape [B, T].
      backbone_lm_loss: Optional pre-computed causal loss of the backbone model.

    Returns:
      losses: A unified dictionary containing the combined loss and individual
        components.
    """
    ignore_index = -100
    seq_len = labels.shape[1]
    vocab_size = backbone_logits.shape[-1]
    mtp_training_length = mtp_logits_all.shape[0]

    if backbone_lm_loss is None:
      # Causal next-token prediction loss for the backbone model.
      valid_logits = backbone_logits[:, :-1, :].reshape(-1, vocab_size)
      valid_labels = labels[:, 1:].reshape(-1)
      backbone_lm_loss = F.cross_entropy(
          valid_logits, valid_labels, ignore_index=ignore_index
      )

    total_mtp_loss = torch.tensor(0.0, device=backbone_logits.device)
    total_mtp_lm_loss = torch.tensor(0.0, device=backbone_logits.device)
    total_mtp_distill_loss = torch.tensor(0.0, device=backbone_logits.device)

    for k in range(mtp_training_length):
      # Speculative prediction bounds.
      valid_len = seq_len - (k + 1)
      if valid_len <= 0:
        continue

      # 1. Next-Token Prediction LM Loss.
      shifted_labels = labels[:, (k + 1) :]
      valid_mtp_logits = mtp_logits_all[k, :, :valid_len, :].reshape(
          -1, vocab_size
      )
      valid_labels = shifted_labels.reshape(-1)

      lm_loss = F.cross_entropy(
          valid_mtp_logits, valid_labels, ignore_index=ignore_index
      )

      # 2. Probability Distillation Total Variation Distance (TVD) Loss.
      valid_mask = valid_labels != ignore_index
      flat_mask = valid_mask.to(mtp_logits_all.dtype)
      target_logits = backbone_logits[:, k : k + valid_len, :].detach()
      flat_mtp_probs = F.softmax(
          mtp_logits_all[k, :, :valid_len, :], dim=-1
      ).reshape(-1, vocab_size)
      flat_target_probs = F.softmax(target_logits, dim=-1).reshape(
          -1, vocab_size
      )

      tvd = torch.sum(
          torch.abs(flat_mtp_probs - flat_target_probs) / 2.0, dim=-1
      )
      masked_tvd = tvd * flat_mask
      distill_loss = masked_tvd.sum() / (flat_mask.sum() + 1e-8)

      # 3. Combined loss for the current speculative head.
      mtp_loss = (
          1 - self.mtp_distillation_weight
      ) * lm_loss + self.mtp_distillation_weight * distill_loss

      total_mtp_loss = total_mtp_loss + mtp_loss
      total_mtp_lm_loss = total_mtp_lm_loss + lm_loss
      total_mtp_distill_loss = total_mtp_distill_loss + distill_loss

    final_loss = backbone_lm_loss + self.mtp_loss_weight * total_mtp_loss

    return {
        "loss": final_loss,
        "backbone_lm_loss": backbone_lm_loss.detach(),
        "mtp_lm_loss": total_mtp_lm_loss.detach(),
        "mtp_distill_loss": total_mtp_distill_loss.detach(),
    }


class Gemma4TrainingModel(torch.nn.Module):
  """Unified training model wrapper for Gemma4 speculative lookahead tuning.

  Attributes:
    model: The base model.
    include_mtp_loss: Whether to include MTP loss.
    mtp_training_length: Length of MTP training.
    assistant_model: The assistant model for MTP.
    loss_metrics: Cached metrics from the last forward pass.
  """

  def __init__(
      self,
      model: torch.nn.Module,
      *,
      assistant_model: torch.nn.Module | None = None,
      include_mtp_loss: bool = False,
      mtp_training_length: int = 3,
      mtp_distillation_weight: float = 0.9,
      mtp_loss_weight: float = 0.1,
      detach_mtp_inputs: bool = True,
      freeze_backbone: bool = False,
  ):
    super().__init__()
    self.model = model
    self.freeze_backbone = freeze_backbone
    self.assistant_model = assistant_model
    if self.assistant_model is None:
      self.assistant_model = getattr(model, "assistant_model", None)
      if self.assistant_model is None and hasattr(model, "base_model"):
        self.assistant_model = getattr(
            model.base_model.model, "assistant_model", None
        )

    self.include_mtp_loss = include_mtp_loss
    self.mtp_training_length = mtp_training_length
    self.detach_mtp_inputs = detach_mtp_inputs

    self.loss_metrics = {}  # Cache metrics from the last forward pass.

    if self.include_mtp_loss:
      if self.assistant_model is None:
        raise ValueError(
            "Base model does not have a valid assistant_model submodule. "
            "Cannot train speculative MTP without draft heads."
        )
      self.loss_fn = Gemma4LossFunction(
          mtp_distillation_weight=mtp_distillation_weight,
          mtp_loss_weight=mtp_loss_weight,
      )

    # Configure requires_grad and train/eval modes for all components natively.

    self._configure_trainable_model_components()

  def __getattr__(self, name: str) -> typing.Any:
    """Forward missing attributes to base model."""
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self.model, name)

  def _configure_trainable_model_components(self) -> None:
    """Configures requires_grad and train/eval modes for all components."""
    # 1. Set modes based on freeze_backbone.
    if self.freeze_backbone:
      # If backbone is frozen (MTP Only mode), turn off dropout in Teacher
      # (Backbone).
      self.model.eval()
      if self.assistant_model is not None:
        # Student (Drafter) should use dropout if configured.
        self.assistant_model.train()
    else:
      self.model.train()
      if self.assistant_model is not None:
        self.assistant_model.train()

    # 2. Configure trainable parameters.
    for name, param in self.model.named_parameters():
      # If freeze_backbone is True, completely freeze all parameters of
      # self.model. Only parameters belonging to self.assistant_model will be
      # un-frozen below.
      if (
          self.freeze_backbone
          and self.assistant_model is not None
          and not any(param is p for p in self.assistant_model.parameters())
      ):
        param.requires_grad = False
        continue

      # Enable trainable gradients only for actual LoRA adapter weights.
      param.requires_grad = False
      if "lora" in name:
        param.requires_grad = True

  def forward(
      self,
      input_ids: torch.LongTensor | None = None,
      pixel_values: torch.FloatTensor | None = None,
      pixel_values_videos: torch.FloatTensor | None = None,
      input_features: torch.FloatTensor | None = None,
      attention_mask: torch.Tensor | None = None,
      input_features_mask: torch.Tensor | None = None,
      position_ids: torch.LongTensor | None = None,
      image_position_ids: torch.LongTensor | None = None,
      video_position_ids: torch.LongTensor | None = None,
      past_key_values: typing.Any = None,
      mm_token_type_ids: torch.LongTensor | None = None,
      inputs_embeds: torch.FloatTensor | None = None,
      labels: torch.LongTensor | None = None,
      use_cache: bool | None = None,
      logits_to_keep: int | torch.Tensor = 0,
      **kwargs,
  ) -> typing.Any:
    # 1. Run backbone forward pass.
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache if use_cache is not None else True,
        logits_to_keep=logits_to_keep,
        output_hidden_states=self.include_mtp_loss,
        return_shared_kv_states=self.include_mtp_loss,
        **kwargs,
    )

    # If MTP is disabled, return causal backbone outputs cleanly.
    if not self.include_mtp_loss:
      self.loss_metrics = {"backbone_lm_loss": outputs.loss.detach()}
      return outputs

    # 2. Parallel lookahead speculative forward pass.
    mtp_logits_all = self.get_candidate_logits(outputs)

    # 3. Compute combined speculative MTP loss.
    losses = self.loss_fn(
        backbone_lm_loss=outputs.loss,
        backbone_logits=outputs.logits,
        mtp_logits_all=mtp_logits_all,
        labels=labels,
    )

    # Cache metrics for Trainer logging.
    self.loss_metrics = losses
    outputs.loss = losses["loss"]
    return outputs

  def get_candidate_logits(
      self,
      outputs: typing.Any,
  ) -> torch.Tensor:
    """Generates candidate logits for speculative decoding.

    Args:
      outputs: Model outputs from a forward pass.

    Returns:
      A tensor of shape [K, B, T, V] containing the logits for K speculative
        steps.
    """
    token_embeddings, *_, backbone_states = outputs.hidden_states
    seq_len = token_embeddings.shape[1]

    if self.detach_mtp_inputs:
      # Always detach embeddings as per DeepMind reference, and completely
      # detach backbone states from the MTP graph so the enormous MTP loss
      # doesn't warp the limited LoRA updates of the backbone.
      token_embeddings = token_embeddings.detach()
      backbone_states = backbone_states.detach()

      # Detach the KV cache provided to the assistant model safely supporting
      # any tuple or tensor struct.
      if outputs.shared_kv_states is not None:

        def detach_tree(obj):
          if isinstance(obj, torch.Tensor):
            return obj.detach()
          elif isinstance(obj, tuple):
            return tuple(detach_tree(x) for x in obj)
          elif isinstance(obj, list):
            return [detach_tree(x) for x in obj]
          return obj

        shared_kv_states = detach_tree(outputs.shared_kv_states)
      else:
        shared_kv_states = None
    else:
      shared_kv_states = outputs.shared_kv_states

    position_ids = (
        torch.arange(seq_len, device=token_embeddings.device)
        .unsqueeze(0)
        .expand(token_embeddings.shape[0], -1)
    )

    mtp_logits_list = []
    current_state = backbone_states

    for k in range(self.mtp_training_length):
      # Teacher Forcing: Shift token embeddings left by k in parallel to match
      # eevee.
      if k == 0:
        shifted_embedding = token_embeddings
      else:
        shifted_embedding = torch.zeros_like(token_embeddings)
        shifted_embedding[:, :-k, :] = token_embeddings[:, k:, :]

      inputs_embeds = torch.cat([shifted_embedding, current_state], dim=-1)
      out = self.assistant_model(
          inputs_embeds=inputs_embeds,
          shared_kv_states=shared_kv_states,
          position_ids=position_ids,
          use_cache=False,
      )

      mtp_logits = out.logits
      next_state = out.last_hidden_state

      mtp_logits_list.append(mtp_logits)
      current_state = next_state

    return torch.stack(mtp_logits_list, dim=0)


class Gemma4LoRATrainer(transformers.Trainer):
  """Trainer wrapper to gather and average MTP metrics for standard logging.

  Attributes:
    include_mtp_loss: Whether to include MTP loss.
  """

  def __init__(
      self,
      *args,
      include_mtp_loss: bool = False,
      mtp_training_length: int = 3,
      mtp_distillation_weight: float = 0.9,
      mtp_loss_weight: float = 0.1,
      freeze_backbone: bool = False,
      **kwargs,
  ):
    model = kwargs.get("model", args[0] if len(args) > 0 else None)
    if model is not None and not isinstance(model, Gemma4TrainingModel):
      wrapped_model = Gemma4TrainingModel(
          model=model,
          include_mtp_loss=include_mtp_loss,
          mtp_training_length=mtp_training_length,
          mtp_distillation_weight=mtp_distillation_weight,
          mtp_loss_weight=mtp_loss_weight,
          freeze_backbone=freeze_backbone,
      )
      if "model" in kwargs:
        kwargs["model"] = wrapped_model
      elif len(args) > 0:
        args = list(args)
        args[0] = wrapped_model
        args = tuple(args)

    super().__init__(*args, **kwargs)
    self.include_mtp_loss = include_mtp_loss

  def compute_loss(
      self,
      model: torch.nn.Module,
      inputs: dict[str, typing.Any],
      return_outputs: bool = False,
      **kwargs,
  ) -> torch.Tensor | tuple[torch.Tensor, typing.Any]:
    """Computes training loss and tracks MTP loss metrics."""
    outputs = model(**inputs)
    loss = (
        outputs.loss
        if isinstance(outputs, dict)
        else getattr(outputs, "loss", outputs)
    )

    # Gathers custom loss metrics from the model's cached forward pass.
    if hasattr(model, "loss_metrics") and model.loss_metrics:
      if not hasattr(self, "_custom_loss_tracker"):
        self._custom_loss_tracker = collections.defaultdict(float)
        self._custom_loss_steps = 0

      for name, value in model.loss_metrics.items():
        self._custom_loss_tracker[name] += value.item()
      self._custom_loss_steps += 1

    return (loss, outputs) if return_outputs else loss

  def log(self, logs: dict[str, float], *args, **kwargs) -> None:
    """Logs metrics and resets custom loss trackers."""
    if hasattr(self, "_custom_loss_tracker") and self._custom_loss_steps > 0:
      avg_losses = {
          name: value / self._custom_loss_steps
          for name, value in self._custom_loss_tracker.items()
      }
      for name, value in avg_losses.items():
        logs[name] = value
      self._custom_loss_tracker.clear()
      self._custom_loss_steps = 0

    super().log(logs, *args, **kwargs)
