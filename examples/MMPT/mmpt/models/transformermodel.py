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

import sys

import torch

from torch import nn

try:
    # transformers==3.4.0
    from transformers.modeling_bert import (
        BertPreTrainedModel,
        BertModel,
        BertEncoder,
        BertPredictionHeadTransform,
    )
except ImportError:
    # latest
    from transformers.models.bert.modeling_bert import (
        BertPreTrainedModel,
        BertModel,
        BertEncoder,
        BertPredictionHeadTransform,
    )

from ..modules import VideoConv1D, VideoTokenMLP, MMBertEmbeddings, Multimodal_Projection

# --------------- fine-tuning models ---------------
class MMBertForJoint(BertPreTrainedModel):
    """A BertModel with isolated attention mask to separate modality."""

    def __init__(self, config):
        super().__init__(config)
        self.videomlp = VideoTokenMLP(config)
        self.bert = MMBertModel(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        input_video_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        separate_forward_split=None,
    ):
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )
        video_tokens = self.videomlp(input_video_embeds)

        outputs = self.bert(
            input_ids,
            video_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            separate_forward_split=separate_forward_split,
        )

        return outputs


class MMBertForTokenClassification(BertPreTrainedModel):
    """A BertModel similar to MMJointUni, with extra wrapper layer
    to be fine-tuned from other pretrained MMFusion model."""

    def __init__(self, config):
        super().__init__(config)
        self.videomlp = VideoTokenMLP(config)
        self.bert = MMBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # TODO(huxu): 779 is the number of classes for COIN: move to config?
        self.classifier = nn.Linear(config.hidden_size, 779)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        input_video_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        separate_forward_split=None,
    ):
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        video_tokens = self.videomlp(input_video_embeds)
        outputs = self.bert(
            input_ids,
            video_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            separate_forward_split=separate_forward_split,
        )

        return (self.classifier(outputs[0]),)


# ------------ pre-training models ----------------

sys.path.append("/athenahomes/zifan/pose-islr")
from src.models.components.agcn_modules.agcn import AGCN
from src.models.components.conformer_modules.conformer_ import Conformer
from src.models.components.conformer_modules.conformer import Encoder as ConformerEncoder
from src.models.components.cosign_modules.cosign import CoSign

class MMBertForEncoder(BertPreTrainedModel):
    """A BertModel for Contrastive Learning."""
    def __init__(self, config):
        super().__init__(config)
        self.videoconv = VideoConv1D(config) if hasattr(config, "conv1d") else None
        self.videomlp = VideoTokenMLP(config)
        self.bert = MMBertModel(config)
        self.init_weights()

        self.perceiver = None
        # from perceiver_pytorch import Perceiver
        # self.perceiver = Perceiver(
        #     input_channels = config.input_dim,          # number of channels for each token of the input
        #     input_axis = 1,              # number of axis for input data (2 for images, 3 for video)
        #     num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
        #     max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
        #     depth = 6,                   # depth of net. The shape of the final attention mechanism will be:
        #                                 #   depth * (cross attention -> self_per_cross_attn * self attention)
        #     num_latents = 8,           # number of latents, or induced set points, or centroids. different papers giving it different names
        #     latent_dim = config.hidden_size,            # latent dimension
        #     cross_heads = 1,             # number of heads for cross attention. paper said 1
        #     latent_heads = 8,            # number of heads for latent self attention, 8
        #     cross_dim_head = 64,         # number of dimensions per cross attention head
        #     latent_dim_head = 64,        # number of dimensions per latent self attention head
        #     num_classes = config.hidden_size,          # output number of classes
        #     attn_dropout = 0.,
        #     ff_dropout = 0.,
        #     weight_tie_layers = False,   # whether to weight tie layers (optional, as indicated in the diagram)
        #     fourier_encode_data = True,  # whether to auto-fourier encode the data, using the input_axis given. defaults to True, but can be turned off if you are fourier encoding the data yourself
        #     self_per_cross_attn = 2,      # number of self attention blocks per cross attention
        #     final_classifier_head=False,
        # )

        self.use_agcn = getattr(config, "use_agcn", False)
        self.use_conformer = getattr(config, "use_conformer", False)
        self.use_cosign = getattr(config, "use_cosign", False)

        if self.use_agcn:
            agcn_cfg = getattr(config, "agcn_config", {})
            
            # Set defaults if not provided
            graph_args = agcn_cfg.get("graph_args", {"labeling_mode": "spatial"})
            num_joints = agcn_cfg.get("num_point", 203)
            dropout = agcn_cfg.get("dropout", 0.0)
            dropedge = agcn_cfg.get("dropedge", 0.1)
            in_channels = agcn_cfg.get("in_channels", 3)  # 3D keypoints

            self.num_joints = num_joints  # used in forward reshape logic

            self.agcn = AGCN(
                in_channels=in_channels,
                graph_args=graph_args,
                edge_importance_weighting=True,
                temporal_kernel_size=9,
                num_joints=num_joints,
                init_size=64,
                num_layers=getattr(config, "agcn_layers", 4),
                dropout=dropout,
            )

            if self.use_conformer:
                conformer_cfg = getattr(config, "conformer_config", {})

                input_dim = conformer_cfg.get("input_dim", self.agcn.output_size)  # likely 1280
                hidden_size = conformer_cfg.get("hidden_size", 1024)  # specified

                self.conformer_mapper = nn.Linear(input_dim, hidden_size)

                self.conformer = Conformer(
                    dim=hidden_size,
                    depth=conformer_cfg.get("depth", 2),
                    dim_head=conformer_cfg.get("dim_head", 128),
                    heads=conformer_cfg.get("heads", 8),
                    ff_mult=conformer_cfg.get("ff_mult", 4),
                    conv_expansion_factor=conformer_cfg.get("conv_expansion_factor", 1),
                    conv_kernel_size=conformer_cfg.get("conv_kernel_size", [5]),
                    attn_dropout=conformer_cfg.get("dropout", 0.0),
                    ff_dropout=conformer_cfg.get("dropout", 0.0),
                    conv_dropout=conformer_cfg.get("dropout", 0.0),
                    pool=conformer_cfg.get("pool", "mean"),
                    use_vel=conformer_cfg.get("use_vel", False),
                )

                self.conformer_proj = nn.Linear(hidden_size, config.hidden_size)  # 1024 → 768
            else:
                # Project AGCN output to hidden size for BERT
                self.agcn_proj = nn.Linear(self.agcn.output_size, config.hidden_size)

        elif self.use_cosign:
            self.cosign = CoSign(
                body_encoder=AGCN(
                    in_channels=64,
                    graph_args={"layout": "body33", "strategy": "spatial"},
                    num_joints=33,
                    edge_importance_weighting=True,
                    temporal_kernel_size=7,
                    init_size=64,
                    num_layers=4,
                    channel_multipliers=[2, 1, 1.5, 2],
                    dropout=0.1,
                    use_se=True,
                ),
                face_encoder=AGCN(
                    in_channels=64,
                    graph_args={"layout": "face128", "strategy": "spatial"},
                    num_joints=128,
                    edge_importance_weighting=True,
                    temporal_kernel_size=7,
                    init_size=64,
                    num_layers=4,
                    channel_multipliers=[2, 1, 1.5, 2],
                    dropout=0.1,
                    use_se=True,
                ),
                hand_encoder=AGCN(
                    in_channels=64,
                    graph_args={"layout": "hand21", "strategy": "spatial"},
                    num_joints=21,
                    edge_importance_weighting=True,
                    temporal_kernel_size=7,
                    init_size=64,
                    num_layers=4,
                    channel_multipliers=[2, 1, 1.5, 2],
                    dropout=0.1,
                    use_se=True,
                ),
                sequence_model=ConformerEncoder(
                    input_dim=1536,
                    hidden_size=512,
                    depth=4,
                    dim_head=128,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=[5],
                    dropout=0.1,
                    pool=None,
                    use_vel=False,
                ),
                expert_model=ConformerEncoder(
                    input_dim=384,
                    hidden_size=512,
                    depth=4,
                    dim_head=128,
                    heads=8,
                    ff_mult=4,
                    conv_expansion_factor=2,
                    conv_kernel_size=[5],
                    dropout=0.1,
                    pool=None,
                    use_vel=False,
                ),
                separate_expert=False,
                separate_hand=False,
                output_dim=512,
                use_mapping=True,
                aggregate="flatten",
                joint_embedding=False,
                modality_fusion=False,
                pretrained=None
            )
            cosign_dim = 512
            self.cosign_proj = nn.Linear(cosign_dim, config.hidden_size)

        else:
            self.videoconv = VideoConv1D(config) if hasattr(config, "conv1d") else None
            self.videomlp = VideoTokenMLP(config)


    def forward(
        self,
        input_ids=None,
        input_video_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        if input_video_embeds is not None:
            if self.use_agcn:
                # Input: [B, T, D] → reshape to [B, T, V, C]
                B, T, D = input_video_embeds.shape
                V = self.num_joints
                C = D // V
                x = input_video_embeds.view(B, T, V, C)
                x = self.agcn(x)  # [B, C', T, V]
                x = x.mean(-1).permute(0, 2, 1)  # → [B, T, C']

                if self.use_conformer:
                    x = self.conformer_mapper(x)
                    x = self.conformer.get_features(x)  # [B, T, D], no pooling
                    x = self.conformer_proj(x)
                else:
                    x = self.agcn_proj(x)

                video_tokens = x  # Final output shape: [B, T, hidden_size]
            elif self.use_cosign:
                # Infer mask by checking for zero-padding
                with torch.no_grad():
                    frame_energy = input_video_embeds.abs().sum(dim=-1)  # [B, T]
                    inferred_mask = frame_energy > 1e-5                  # [B, T] boolean

                cosign_outputs = self.cosign(input_video_embeds, mask=inferred_mask)
                video_tokens = cosign_outputs["pooled_all"]
                video_tokens = self.cosign_proj(video_tokens)
            else:
                video_tokens = self.videomlp(self.videoconv(input_video_embeds)) if self.videoconv else self.videomlp(input_video_embeds)
        else:
            video_tokens = None

        outputs = self.bert(
            input_ids,
            video_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return outputs


class MMBertForMFMMLM(BertPreTrainedModel):
    """A BertModel with shared prediction head on MFM-MLM."""
    def __init__(self, config):
        super().__init__(config)
        self.videomlp = VideoTokenMLP(config)
        self.bert = MMBertModel(config)
        self.cls = MFMMLMHead(config)
        self.hidden_size = config.hidden_size
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        input_video_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_frame_labels=None,
        target_video_hidden_states=None,
        non_masked_frame_mask=None,
        masked_lm_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )
        if input_video_embeds is not None:
            video_tokens = self.videomlp(input_video_embeds)
        else:
            video_tokens = None

        if target_video_hidden_states is not None:
            target_video_hidden_states = self.videomlp(
                target_video_hidden_states)

            non_masked_frame_hidden_states = video_tokens.masked_select(
                non_masked_frame_mask.unsqueeze(-1)
            ).view(-1, self.hidden_size)

        outputs = self.bert(
            input_ids,
            video_tokens,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        mfm_scores, prediction_scores = None, None
        if masked_frame_labels is not None and masked_lm_labels is not None:
            # split the sequence.
            text_offset = masked_frame_labels.size(1) + 1  # [CLS]
            video_sequence_output = sequence_output[
                :, 1:text_offset
            ]  # remove [SEP] as not in video_label.
            text_sequence_output = torch.cat(
                [sequence_output[:, :1], sequence_output[:, text_offset:]],
                dim=1
            )

            hidden_size = video_sequence_output.size(-1)
            selected_video_output = video_sequence_output.masked_select(
                masked_frame_labels.unsqueeze(-1)
            ).view(-1, hidden_size)

            # only compute select tokens to training to speed up.
            hidden_size = text_sequence_output.size(-1)
            # masked_lm_labels = masked_lm_labels.reshape(-1)
            labels_mask = masked_lm_labels != -100

            selected_text_output = text_sequence_output.masked_select(
                labels_mask.unsqueeze(-1)
            ).view(-1, hidden_size)
            mfm_scores, prediction_scores = self.cls(
                selected_video_output,
                target_video_hidden_states,
                non_masked_frame_hidden_states,
                selected_text_output,
            )

        output = (
            mfm_scores,
            prediction_scores,
        ) + outputs
        return output


class BertMFMMLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly
        # resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(
        self,
        video_hidden_states=None,
        target_video_hidden_states=None,
        non_masked_frame_hidden_states=None,
        text_hidden_states=None,
    ):
        video_logits, text_logits = None, None
        if video_hidden_states is not None:
            video_hidden_states = self.transform(video_hidden_states)
            non_masked_frame_logits = torch.mm(
                video_hidden_states,
                non_masked_frame_hidden_states.transpose(1, 0)
            )
            masked_frame_logits = torch.bmm(
                video_hidden_states.unsqueeze(1),
                target_video_hidden_states.unsqueeze(-1),
            ).squeeze(-1)
            video_logits = torch.cat(
                [masked_frame_logits, non_masked_frame_logits], dim=1
            )

        if text_hidden_states is not None:
            text_hidden_states = self.transform(text_hidden_states)
            text_logits = self.decoder(text_hidden_states)
        return video_logits, text_logits


class MFMMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertMFMMLMPredictionHead(config)

    def forward(
        self,
        video_hidden_states=None,
        target_video_hidden_states=None,
        non_masked_frame_hidden_states=None,
        text_hidden_states=None,
    ):
        video_logits, text_logits = self.predictions(
            video_hidden_states,
            target_video_hidden_states,
            non_masked_frame_hidden_states,
            text_hidden_states,
        )
        return video_logits, text_logits


class MMBertForMTM(MMBertForMFMMLM):
    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)
        self.videomlp = VideoTokenMLP(config)
        self.bert = MMBertModel(config)
        self.cls = MTMHead(config)
        self.hidden_size = config.hidden_size
        self.init_weights()


class BertMTMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        video_hidden_states=None,
        target_video_hidden_states=None,
        non_masked_frame_hidden_states=None,
        text_hidden_states=None,
    ):
        non_masked_frame_hidden_states = non_masked_frame_hidden_states.transpose(1, 0)
        video_logits, text_logits = None, None
        if video_hidden_states is not None:
            video_hidden_states = self.transform(video_hidden_states)

            masked_frame_logits = torch.bmm(
                video_hidden_states.unsqueeze(1),
                target_video_hidden_states.unsqueeze(-1),
            ).squeeze(-1)

            non_masked_frame_logits = torch.mm(
                video_hidden_states,
                non_masked_frame_hidden_states
            )
            video_on_vocab_logits = self.decoder(video_hidden_states)
            video_logits = torch.cat([
                masked_frame_logits,
                non_masked_frame_logits,
                video_on_vocab_logits], dim=1)

        if text_hidden_states is not None:
            text_hidden_states = self.transform(text_hidden_states)
            # text first so label does not need to be shifted.
            text_on_vocab_logits = self.decoder(text_hidden_states)
            text_on_video_logits = torch.mm(
                text_hidden_states,
                non_masked_frame_hidden_states
            )
            text_logits = torch.cat([
                text_on_vocab_logits,
                text_on_video_logits
            ], dim=1)

        return video_logits, text_logits


class MTMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertMTMPredictionHead(config)

    def forward(
        self,
        video_hidden_states=None,
        target_video_hidden_states=None,
        non_masked_frame_hidden_states=None,
        text_hidden_states=None,
    ):
        video_logits, text_logits = self.predictions(
            video_hidden_states,
            target_video_hidden_states,
            non_masked_frame_hidden_states,
            text_hidden_states,
        )
        return video_logits, text_logits


class MMBertModel(BertModel):
    """MMBertModel has MMBertEmbedding to support video tokens."""

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        # overwrite embedding
        self.embeddings = MMBertEmbeddings(config)
        self.encoder = MultiLayerAttentionMaskBertEncoder(config)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        input_video_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        separate_forward_split=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids "
                "and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            if input_video_embeds is not None:
                input_shape = (
                    input_ids.size(0),
                    input_ids.size(1) + input_video_embeds.size(1),
                )
            else:
                input_shape = (
                    input_ids.size(0),
                    input_ids.size(1),
                )
        elif inputs_embeds is not None:
            if input_video_embeds is not None:
                input_shape = (
                    inputs_embeds.size(0),
                    inputs_embeds.size(1) + input_video_embeds.size(1),
                )
            else:
                input_shape = (
                    input_ids.size(0),
                    input_ids.size(1),
                )
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None \
            else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case
        # we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = \
            self.get_extended_attention_mask(
                attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to
        # [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or
        # [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]

        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids,
            input_video_embeds,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        if separate_forward_split is not None:
            split_embedding_output = \
                embedding_output[:, :separate_forward_split]
            split_extended_attention_mask = extended_attention_mask[
                :, :, :, :separate_forward_split, :separate_forward_split
            ]
            split_encoder_outputs = self.encoder(
                split_embedding_output,
                attention_mask=split_extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            assert (
                len(split_encoder_outputs) <= 2
            ), "we do not support merge on attention for now."
            encoder_outputs = []
            encoder_outputs.append([split_encoder_outputs[0]])
            if len(split_encoder_outputs) == 2:
                encoder_outputs.append([])
                for _all_hidden_states in split_encoder_outputs[1]:
                    encoder_outputs[-1].append([_all_hidden_states])

            split_embedding_output = \
                embedding_output[:, separate_forward_split:]
            split_extended_attention_mask = extended_attention_mask[
                :, :, :, separate_forward_split:, separate_forward_split:
            ]

            split_encoder_outputs = self.encoder(
                split_embedding_output,
                attention_mask=split_extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            assert (
                len(split_encoder_outputs) <= 2
            ), "we do not support merge on attention for now."
            encoder_outputs[0].append(split_encoder_outputs[0])
            encoder_outputs[0] = torch.cat(encoder_outputs[0], dim=1)
            if len(split_encoder_outputs) == 2:
                for layer_idx, _all_hidden_states in enumerate(
                    split_encoder_outputs[1]
                ):
                    encoder_outputs[1][layer_idx].append(_all_hidden_states)
                    encoder_outputs[1][layer_idx] = torch.cat(
                        encoder_outputs[1][layer_idx], dim=1
                    )
            encoder_outputs = tuple(encoder_outputs)
        else:
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        return (sequence_output, pooled_output) + encoder_outputs[1:]

    def get_extended_attention_mask(self, attention_mask, input_shape, device):
        """This is borrowed from `modeling_utils.py` with the support of
        multi-layer attention masks.
        The second dim is expected to be number of layers.
        See `MMAttentionMaskProcessor`.
        Makes broadcastable attention and causal masks so that future
        and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to,
                zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, \
                with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable
        # to all heads.
        if attention_mask.dim() == 4:
            extended_attention_mask = attention_mask[:, :, None, :, :]
            extended_attention_mask = extended_attention_mask.to(
                dtype=self.dtype
            )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) \
                * -10000.0
            return extended_attention_mask
        else:
            return super().get_extended_attention_mask(
                attention_mask, input_shape, device
            )


class MultiLayerAttentionMaskBertEncoder(BertEncoder):
    """extend BertEncoder with the capability of
    multiple layers of attention mask."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_attention_mask = (
                attention_mask[:, i, :, :, :]
                if attention_mask.dim() == 5
                else attention_mask
            )

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # layer_outputs = layer_module(
                #     hidden_states,
                #     layer_attention_mask,
                #     layer_head_mask,
                #     encoder_hidden_states,
                #     encoder_attention_mask,
                #     output_attentions,
                # )
                # support latest transformers package
                # see https://github.com/facebookresearch/fairseq/issues/5404#issuecomment-1963062507
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    attention_mask=layer_attention_mask,
                    head_mask=layer_head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    output_attentions=output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_attentions]
            if v is not None
        )
