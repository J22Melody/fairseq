# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# Copyright (c) Facebook, Inc. All Rights Reserved


import torch

from torch import nn
import torch.nn.functional as F



try:
    # transformers==3.4.0
    from transformers.modeling_bert import (
        BertEmbeddings,
        ACT2FN,
    )
except ImportError:
    # latest
    from transformers.models.bert.modeling_bert import (
        BertEmbeddings,
        ACT2FN,
    )


class Multimodal_Projection(nn.Module):
    def __init__(self, l2_norm=False, in_dim=768, out_dim=768):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.l2_norm = l2_norm

    def forward(self, hidden_states):
        hidden_states = self.linear(hidden_states)
        if self.l2_norm:
            hidden_states = nn.functional.normalize(hidden_states, p=2, dim=1)
        return hidden_states


class VideoConv1D(nn.Module):
    def __init__(self, config, kernel_size=17):
        super().__init__()
        self.num_layers = config.conv1d
        input_dim = config.input_dim if hasattr(config, "input_dim") else 512
        for i in range(self.num_layers):
            setattr(self, f'conv1d_{i}', nn.Conv1d(input_dim, input_dim, kernel_size, padding='same'))

    def forward(self, hidden_states):
        hidden_states = torch.swapaxes(hidden_states, 1, 2)
        for i in range(self.num_layers):
            hidden_states = getattr(self, f'conv1d_{i}')(hidden_states)
        hidden_states = torch.swapaxes(hidden_states, 1, 2)
        return hidden_states


class VideoTokenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config.input_dim if hasattr(config, "input_dim") else 512
        self.dropout = nn.Dropout(p=config.dropout) if hasattr(config, "dropout") else None
        self.linear1 = nn.Linear(input_dim, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear2(hidden_states)
        if self.dropout:
            hidden_states = self.dropout(hidden_states)
        return hidden_states


class MMBertEmbeddings(BertEmbeddings):
    def __init__(self, config):
        super().__init__(config)
        self.max_video_len = config.max_video_len
        if hasattr(config, "use_seg_emb") and config.use_seg_emb:
            """the original VLM paper uses seg_embeddings for temporal space.
            although not used it changed the randomness of initialization.
            we keep it for reproducibility.
            """
            self.seg_embeddings = nn.Embedding(256, config.hidden_size)

    def forward_old(
        self,
        input_ids,
        input_video_embeds,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        input_tensor = input_ids if input_ids is not None else inputs_embeds
        if input_video_embeds is not None:
            input_shape = (
                input_tensor.size(0),
                input_tensor.size(1) + input_video_embeds.size(1),
            )
        else:
            input_shape = (input_tensor.size(0), input_tensor.size(1))

        if position_ids is None:
            """
            Auto skip position embeddings for text only case.
            use cases:
            (1) action localization and segmentation:
                feed in len-1 dummy video token needs text part to
                skip input_video_embeds.size(1) for the right
                position_ids for video [SEP] and rest text tokens.
            (2) MMFusionShare for two forward passings:
                in forward_text: input_video_embeds is None.
                    need to skip video [SEP] token.

            # video_len + 1: [CLS] + video_embed
            # self.max_video_len + 1: [SEP] for video.
            # self.max_video_len + 2: [SEP] for video.
            # self.max_video_len + input_ids.size(1): rest for text.
            """
            if input_video_embeds is not None:
                video_len = input_video_embeds.size(1)
                starting_offset = self.max_video_len + 1  # video [SEP]
                ending_offset = self.max_video_len + input_ids.size(1)
            else:
                video_len = 0
                starting_offset = self.max_video_len + 2  # first text token.
                ending_offset = self.max_video_len + input_ids.size(1) + 1
            position_ids = torch.cat([
                self.position_ids[:, :video_len + 1],
                self.position_ids[:, starting_offset:ending_offset]
                ], dim=1)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        """
        the format of input_ids is [CLS] [SEP] caption [SEP] padding.
        the goal is to build [CLS] video tokens [SEP] caption [SEP] .
        """
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        if input_video_embeds is not None:
            inputs_mm_embeds = torch.cat([
                inputs_embeds[:, :1], input_video_embeds, inputs_embeds[:, 1:]
            ], dim=1)
        else:
            # text only for MMFusionShare.
            inputs_mm_embeds = inputs_embeds

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_mm_embeds + position_embeddings
        embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def forward(
        self,
        input_ids,
        input_video_embeds,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        # ——— build the “MM” inputs just like original ———
        input_tensor = input_ids if input_ids is not None else inputs_embeds
        if input_video_embeds is not None:
            input_shape = (
                input_tensor.size(0),
                input_tensor.size(1) + input_video_embeds.size(1),
            )
        else:
            input_shape = (input_tensor.size(0), input_tensor.size(1))

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=self.position_ids.device
            )

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if input_video_embeds is not None:
            # insert video tokens after [CLS]
            inputs_mm_embeds = torch.cat([
                inputs_embeds[:, :1],          # [CLS]
                input_video_embeds,            # video frames
                inputs_embeds[:, 1:],          # rest of text
            ], dim=1)
        else:
            inputs_mm_embeds = inputs_embeds

        # ——— build fresh position_ids for the full sequence length ———
        batch_size, L, _ = inputs_mm_embeds.size()
        device = inputs_mm_embeds.device
        position_ids = torch.arange(L, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, L)

        # ——— only if we exceed the pretrained max do we interpolate ———
        old_max, D = self.position_embeddings.weight.shape
        if L > old_max:
            # 1) take (old_max, D) → (1, D, old_max)
            pe = self.position_embeddings.weight.data.transpose(0, 1).unsqueeze(0)
            # 2) interpolate along dim=2 to size=L → (1, D, L)
            pe = F.interpolate(pe, size=L, mode="linear", align_corners=False)
            # 3) back to (L, D)
            new_pe = pe.squeeze(0).transpose(0, 1)

            # 4) rebuild the nn.Embedding in-place
            new_emb = nn.Embedding(L, D).to(new_pe.device)
            with torch.no_grad():
                new_emb.weight.copy_(new_pe)
            self.position_embeddings = new_emb

        # ——— fetch embeddings and combine ———
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_mm_embeds + position_embeddings
        embeddings = embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlignHead(nn.Module):
    """this will load pre-trained weights for NSP, which is desirable."""

    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, dropout_pooled_output):
        logits = self.seq_relationship(dropout_pooled_output)
        return logits
