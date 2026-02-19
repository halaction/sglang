# srt/models/gigarembed.py

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig

logger = logging.getLogger(__name__)


class GigarEmbedModel(nn.Module):
    def __init__(
        self,
        config: AutoConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model_path = prefix or getattr(config, "name_or_path", "")
        if (not self.model_path) or str(self.model_path).startswith(
            ("ai-sage/", "https://", "hf://", "hub://")
        ):
            raise RuntimeError(
                f"Expected local path (--model-path), got: {self.model_path!r}"
            )
        self.pad_id = 0

        self.hf_model: Optional[nn.Module] = None
        self._weights_loaded = False

    def load_weights(self, weights):
        if self._weights_loaded:
            return self

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.hf_model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=True,
        ).eval()
        self.hf_model.cuda()
        self._weights_loaded = True
        return self

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        pp_proxy_tensors=None,
        input_embeds=None,
        get_embedding=False,
    ):

        if self.hf_model is None:
            raise RuntimeError("forward called before load_weights()")

        seq_lens = forward_batch.extend_seq_lens.tolist()
        text_tensors = []
        start_idx = 0

        for seq_len in seq_lens:
            if seq_len > 0:
                text_tokens = input_ids[start_idx : start_idx + seq_len]
                text_tensors.append(text_tokens)
            start_idx += seq_len

        if text_tensors:
            max_len = max(len(t) for t in text_tensors)
            padded_tensors = []
            attention_masks = []

            for text_tensor in text_tensors:
                padded = torch.cat(
                    [
                        text_tensor,
                        torch.full(
                            (max_len - len(text_tensor),),
                            self.pad_id,
                            dtype=text_tensor.dtype,
                            device=text_tensor.device,
                        ),
                    ]
                )
                padded_tensors.append(padded)
                mask = torch.cat(
                    [
                        torch.ones(
                            len(text_tensor),
                            dtype=torch.bool,
                            device=text_tensor.device,
                        ),
                        torch.zeros(
                            max_len - len(text_tensor),
                            dtype=torch.bool,
                            device=text_tensor.device,
                        ),
                    ]
                )
                attention_masks.append(mask)
            batch_input_ids = torch.stack(padded_tensors, dim=0)
            batch_attention_mask = torch.stack(attention_masks, dim=0)
            final_embeddings = self.hf_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                pool_mask=batch_attention_mask,
                return_embeddings=True,
                return_dict=True,
            )

            return EmbeddingPoolerOutput(embeddings=final_embeddings)

    def forward_batch(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


EntryClass = [GigarEmbedModel]
