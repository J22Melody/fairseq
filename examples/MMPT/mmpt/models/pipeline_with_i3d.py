import os
import gc
import sys
import math
import pickle as pkl
import shutil
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
import scipy.special
from beartype import beartype
from tqdm import tqdm

try:
    from transformers import AutoConfig, AutoTokenizer
except ImportError:
    pass

from . import transformermodel, MMFusionSeparate, MMFusionShare, MMFusion

logger = logging.getLogger(__name__)


def sliding_windows(
        rgb: torch.Tensor,
        num_in_frames: int,
        stride: int,
) -> tuple:
    """
    Return sliding windows and corresponding (middle) timestamp
    """
    C, nFrames, H, W = rgb.shape
    # If needed, pad to the minimum clip length
    if nFrames < num_in_frames:
        rgb_ = torch.zeros(C, num_in_frames, H, W, device=rgb.device)
        rgb_[:, :nFrames] = rgb
        rgb_[:, nFrames:] = rgb[:, -1].unsqueeze(1)
        rgb = rgb_
        nFrames = rgb.shape[1]

    num_clips = math.ceil((nFrames - num_in_frames) / stride) + 1
    plural = ""
    if num_clips > 1:
        plural = "s"
    #print(f"{num_clips} clip{plural} resulted from sliding window processing.")

    rgb_slided = torch.zeros(num_clips, 3, num_in_frames, H, W, device=rgb.device)
    t_mid = []
    # For each clip
    for j in range(num_clips):
        # Check if num_clips becomes 0
        actual_clip_length = min(num_in_frames, nFrames - j * stride)
        if actual_clip_length == num_in_frames:
            t_beg = j * stride
        else:
            t_beg = nFrames - num_in_frames
        t_mid.append(t_beg + num_in_frames / 2)
        rgb_slided[j] = rgb[:, t_beg : t_beg + num_in_frames, :, :]
    return rgb_slided, np.array(t_mid)

def load_video_feature_extractor(
        checkpoint_path: str,
        num_classes: int,
        num_in_frames: int,
        path_to_i3d_repo: str,
) -> torch.nn.Module:
    """Load pre-trained I3D checkpoint, put in eval mode.  Download checkpoint
    from url if not found locally.
    """
    sys.path.append(path_to_i3d_repo)
    import models
    from utils.misc import to_torch
    from utils.imutils import im_to_numpy, im_to_torch, resize_generic
    from utils.transforms import color_normalize
    
    def remove_prefix(state_dict, prefix):
        # remove prefix from all keys
        return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    
    model = models.InceptionI3d(
        num_classes=num_classes,
        spatiotemporal_squeeze=True,
        final_endpoint="Logits",
        name="inception_i3d",
        in_channels=3,
        dropout_keep_prob=0.5,
        num_in_frames=num_in_frames,
        include_embds=True,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = torch.nn.DataParallel(model).to(device)
    model.to(device)

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            state_dict = remove_prefix(checkpoint['state_dict'], 'module.')
            model.load_state_dict(state_dict)
            logger.info(f"Loaded I3D from {checkpoint_path}")
    return model

class MMFusionWithI3D(MMFusionSeparate):
    """a MMPT wrapper class for MMBert style models.
    TODO: move isolated mask to a subclass.
    """
    
    def __init__(self, config, **kwargs):
        nn.Module.__init__(self)
        transformer_config = AutoConfig.from_pretrained(
            config.dataset.bert_name)
        self.hidden_size = transformer_config.hidden_size
        self.is_train = False
        if config.dataset.train_path is not None:
            self.is_train = True
        # 0 means no iso; 1-12 means iso up to that layer.
        self.num_hidden_layers = transformer_config.num_hidden_layers
        self.last_iso_layer = 0
        self.video_feature_extractor = load_video_feature_extractor(
            checkpoint_path=config.model.checkpoint_path,
            num_classes=config.model.num_classes,
            num_in_frames=config.model.num_in_frames,
            path_to_i3d_repo=config.model.path_to_i3d_repo,
        )
        if hasattr(config.model, "freeze_feature_extractor"):
            if config.model.freeze_feature_extractor:
                for param in self.video_feature_extractor.parameters():
                    param.requires_grad = False
        self.batch_size = config.fairseq.dataset.batch_size
        if config.dataset.num_iso_layer is not None:
            self.last_iso_layer = config.dataset.num_iso_layer - 1 + 1

        if config.model.mm_encoder_cls is not None:
            mm_encoder_cls = getattr(transformermodel, config.model.mm_encoder_cls)
            model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
            model_config.max_video_len = config.dataset.max_video_len
            # TODO: a general way to add parameter for a model.
            model_config.use_seg_emb = config.model.use_seg_emb
            self.mm_encoder = mm_encoder_cls.from_pretrained(
                config.dataset.bert_name, config=model_config)
        elif config.model.video_encoder_cls is not None\
                and config.model.text_encoder_cls is not None:
            video_encoder_cls = getattr(transformermodel, config.model.video_encoder_cls)
            model_config = AutoConfig.from_pretrained(config.dataset.bert_name)
            model_config.max_video_len = config.dataset.max_video_len
            # TODO: make each model a set of config class.
            if hasattr(model_config, "num_layers"):
                model_config.num_layers = config.model.num_hidden_video_layers
            else:
                model_config.num_hidden_layers = config.model.num_hidden_video_layers
            
            if "vfeat_dim" in config.model:
                model_config.input_dim = config.model.vfeat_dim
            if "conv1d" in config.model:
                model_config.conv1d = config.model.conv1d
            if "dropout" in config.model:
                model_config.dropout = config.model.dropout
                model_config.attention_probs_dropout_prob = config.model.dropout
                model_config.hidden_dropout_prob = config.model.dropout
            self.video_encoder = video_encoder_cls.from_pretrained(
                config.dataset.bert_name, config=model_config)
            # exact same NLP model from Huggingface.
            text_encoder_cls = getattr(transformermodel, config.model.text_encoder_cls)
            self.text_encoder = text_encoder_cls.from_pretrained(
                config.dataset.bert_name)
        else:
            raise ValueError("the encoder must be either MM or two backbones.")

        if "multimodal_projection" in config.model:
            from .transformermodel import Multimodal_Projection
            self.video_projection = Multimodal_Projection()
            self.text_projection = Multimodal_Projection()
        
        self.i3d_batch_size = config.fairseq.dataset.batch_size
        if hasattr(config.model, 'i3d_batch_size'):
            if config.model.i3d_batch_size is not None:
                self.i3d_batch_size = config.model.i3d_batch_size
        logger.info(f"I3D model will use a batch_size of {self.i3d_batch_size}")

    def forward_video_feature_extractor(
            self, 
            vfeats,
            stride=4,
            num_in_frames=16,
            batch_size=16

        ):
        batch_features = []
        for rgb_input in range(vfeats.shape[0]):
            rgb_slides, t_mid = sliding_windows(
                rgb=vfeats[rgb_input],
                stride=stride,
                num_in_frames=num_in_frames,
            )
            rgb_slides = rgb_slides.half()
            num_clips = rgb_slides.shape[0]
            # Group the clips into batches
            num_batches = math.ceil(num_clips / batch_size)
            features = []
            for b in range(num_batches):
                inp = rgb_slides[b * batch_size : (b + 1) * batch_size]
                out = self.video_feature_extractor(inp)
                features.append(out['embds'].view(-1, out['embds'].shape[1]))
            features = torch.cat(tuple(features), dim=0)
            batch_features.append(features)
        batch_features = torch.stack(batch_features, dim=0)
        return batch_features

    def forward(
        self,
        caps,
        cmasks,
        vfeats,
        vmasks,
        attention_mask=None,
        video_label=None,
        text_label=None,
        output_hidden_states=False,
        **kwargs
    ):
        # vmasks: B x N_frames
        # caps: B x T
        # cmasks: B x T

        vfeats = self.forward_video_feature_extractor(
            vfeats=vfeats,
            batch_size=self.i3d_batch_size,
        ) # vfeats: B x N_frames x vfeat_dim

        pooled_video = self.forward_video(
            vfeats,
            vmasks,
            caps,
            cmasks,
            output_hidden_states
        )

        pooled_text = self.forward_text(
            caps,
            cmasks,
            output_hidden_states
        )

        return {"pooled_video": pooled_video, "pooled_text": pooled_text}