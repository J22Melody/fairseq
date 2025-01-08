# Copyright (c) Facebook, Inc. All Rights Reserved

"""
Processors for all downstream (ds) tasks.
"""

import json
import os
import pickle
import random
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from .processor import (
    MetaProcessor,
    VideoProcessor,
    TextProcessor,
    Aligner,
    MMAttentionMask2DProcessor,
)

from .how2processor import TextGenerationProcessor


# ------------- A General Aligner for all downstream tasks-----------------


class DSAligner(Aligner):
    """
    Downstream (DS) aligner shared by all datasets.
    """

    def __call__(self, video_id, video_feature, text_feature, wps=0.7):
        # random sample a starting sec for video.
        video_start = 0
        video_end = min(len(video_feature), self.max_video_len)
        # the whole sequence is a single clip.
        video_clips = {"start": [video_start], "end": [video_end]}

        text_feature = {
            "cap": [text_feature],
            "start": [video_start],
            "end": [len(text_feature) / wps],
        }
        text_clip_indexs = [0]

        vfeats, vmasks = self._build_video_seq(
            video_feature, video_clips
        )
        caps, cmasks = self._build_text_seq(
            text_feature, text_clip_indexs
        )

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,
            "vmasks": vmasks,
            "video_id": video_id,
        }


class NLGTextProcessor(TextProcessor):
    """
    Also return the original text as ref.
    """
    def __call__(self, text_id):
        return super().__call__(text_id), text_id


class DSNLGAligner(DSAligner):
    """extend with the capability of 2d mask for generation."""
    def __init__(self, config):
        super().__init__(config)
        self.attnmasker = MMAttentionMask2DProcessor()
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.bert_name, use_fast=self.use_fast,
            bos_token="[CLS]", eos_token="[SEP]"
        )
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.textgen = TextGenerationProcessor(tokenizer)

    def __call__(self, video_id, video_feature, text_feature):
        output = super().__call__(video_id, video_feature, text_feature[0])
        if self.split == "test":
            # output.update({"ref": text_feature[1]})
            output.update({"ref": self.tokenizer.decode(
                output["caps"], skip_special_tokens=True)})
            text_label = output["caps"]
            cmasks = torch.BoolTensor([1] * text_label.size(0))
            caps = torch.LongTensor([
                self.cls_token_id,
                self.sep_token_id,
                self.bos_token_id])
        else:
            caps, text_label = self.textgen(output["caps"])
            cmasks = output["cmasks"]

        attention_mask = self.attnmasker(
            output["vmasks"], cmasks, "textgen")

        output.update({
            "caps": caps,
            "cmasks": cmasks,
            "text_label": text_label,
            "attention_mask": attention_mask,
        })
        return output


# -------------------- MSRVTT ------------------------


class MSRVTTMetaProcessor(MetaProcessor):
    """MSRVTT dataset.
    reference: `howto100m/msrvtt_dataloader.py`
    """

    def __init__(self, config):
        super().__init__(config)
        import pandas as pd
        data = pd.read_csv(self._get_split_path(config))
        # TODO: add a text1ka flag.
        if config.split == "train" \
                and config.full_test_path is not None \
                and config.jsfusion_path is not None:
            # add testing videos from full_test_path not used by jfusion.
            additional_data = pd.read_csv(config.full_test_path)
            jsfusion_data = pd.read_csv(config.jsfusion_path)

            for video_id in additional_data["video_id"]:
                if video_id not in jsfusion_data["video_id"].values:
                    data = data.append(
                        {"video_id": video_id}, ignore_index=True)

        if config.dup is not None and config.split == "train":
            data = data.append([data] * (config.dup - 1), ignore_index=True)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """slightly modify with if condition to combine train/test."""
        vid, sentence = None, None
        vid = self.data["video_id"].values[idx]
        if "sentence" in self.data:  # for testing.
            sentence = self.data["sentence"].values[idx]
        else:  # for training.
            sentence = vid
        return vid, sentence


class MSRVTTTextProcessor(TextProcessor):
    """MSRVTT dataset.
    reference: `msrvtt_dataloader.py` `MSRVTT_TrainDataLoader`.
    TODO (huxu): add max_words.
    """

    def __init__(self, config):
        super().__init__(config)
        self.sentences = None
        if config.json_path is not None and config.split == "train":
            with open(config.json_path) as fd:
                self.data = json.load(fd)
            self.sentences = defaultdict(list)
            for s in self.data["sentences"]:
                self.sentences[s["video_id"]].append(s["caption"])

    def __call__(self, text_id):
        if self.sentences is not None:
            rind = random.randint(0, len(self.sentences[text_id]) - 1)
            sentence = self.sentences[text_id][rind]
        else:
            sentence = text_id
        caption = self.tokenizer(sentence, add_special_tokens=False)
        return caption["input_ids"]


class MSRVTTNLGTextProcessor(MSRVTTTextProcessor):
    """TODO: change dsaligner and merge to avoid any NLG text processor."""
    def __call__(self, text_id):
        if self.sentences is not None:
            rind = random.randint(0, len(self.sentences[text_id]) - 1)
            sentence = self.sentences[text_id][rind]
        else:
            sentence = text_id
        caption = self.tokenizer(sentence, add_special_tokens=False)
        return caption["input_ids"], sentence


class MSRVTTQAMetaProcessor(MetaProcessor):
    """MSRVTT-QA: retrieval-based multi-choice QA from JSFusion dataset.
    For simplicity, we use the train retrieval model.
    reference: `https://github.com/yj-yu/lsmdc`
    """

    def __init__(self, config):
        super().__init__(config)
        import pandas as pd
        csv_data = pd.read_csv(self._get_split_path(config), sep="\t")
        data = []
        for video_id, a1, a2, a3, a4, a5, answer in zip(
                csv_data["vid_key"].values,
                csv_data["a1"].values,
                csv_data["a2"].values,
                csv_data["a3"].values,
                csv_data["a4"].values,
                csv_data["a5"].values,
                csv_data["answer"].values):
            video_id = video_id.replace("msr", "video")
            data.append((video_id, (answer, [a1, a2, a3, a4, a5])))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MSRVTTQATextProcessor(TextProcessor):
    """MSRVTT-QA dataset.
    text_ans is of format `(answer, [a1, a2, a3, a4, a5])`.
    """

    def __call__(self, text_ans):
        for ans_idx, ans in enumerate(text_ans[1]):
            if isinstance(ans, str):
                text_ans[1][ans_idx] = self.tokenizer(ans, add_special_tokens=False)["input_ids"]
        return text_ans


class MSRVTTQAAligner(DSAligner):
    """MSRVTT dataset.
    similar to sample in how2.
    we call __call__ multiple times.
    """

    def __call__(self, video_id, video_feature, text_feature, wps=0.7):
        caps = []
        cmasks = []
        answer = text_feature[0]
        for ans_idx, _text_feature in enumerate(text_feature[1]):
            output = super().__call__(
                video_id, video_feature, _text_feature, wps)
            caps.append(output["caps"])
            cmasks.append(output["cmasks"])
        output.update({
            "caps": torch.stack(caps),
            "cmasks": torch.stack(cmasks),
            "answers": torch.LongTensor([answer]),
        })
        return output


# -------------------- Youcook -----------------------


class YoucookMetaProcessor(MetaProcessor):
    """Youcook dataset.
    reference: `howto100m/youcook_dataloader.py`
    note that the data can be different as the
    (1) some videos already in Howto100m are removed.
    (2) stop words are removed from caption
    TODO (huxu): make a flag to load the original caption.
    (see youcookii_annotations_trainval.json).

    The max_video_len can be 264 and text can be 64 tokens.
    In reality we may not need that long. see projects/task/youcook.yaml
    """

    def __init__(self, config):
        super().__init__(config)
        # vfeat_dir = config.vfeat_dir
        # print(self._get_split_path(config))
        # with open(self._get_split_path(config), "rb") as fd:
        #     data = pickle.load(fd)
        #     all_valid_video_ids = set(
        #         [os.path.splitext(fn)[0] for fn in os.listdir(vfeat_dir)]
        #     )
        #     recs = []
        #     video_ids = set()
        #     valid_video_ids = set()
        #     for rec in data:  # filter videos not available.
        #         udl_idx = rec["id"].rindex("_")
        #         video_id = rec["id"][:udl_idx]
        #         video_ids.add(video_id)
        #         if video_id in all_valid_video_ids:
        #             valid_video_ids.add(video_id)
        #             recs.append(rec)
        #     print("total video_ids in .pkl", len(video_ids))
        #     print("valid video_ids in .pkl", len(valid_video_ids))
        #     print("please verify {train,val}_list.txt")
        #     data = recs
        #     self.data = data

        with open(config.trainval_annotation) as fd:
            self.youcook_annotation = json.load(fd)["database"]
        if config.use_annotation_text is True:
            print("using text in annotation.")
            self.use_annotation_caption = True
        else:
            self.use_annotation_caption = False

        vfeat_dir = config.vfeat_dir
        all_valid_video_ids = ['fn9anlEL4FI', '-dh_uGahzYo']
        self.data = []

        for id in all_valid_video_ids:
            video_annotation = self.youcook_annotation[id]
            for annotation in video_annotation['annotations']:
                rec = {
                    "id": f"{id}_{annotation['id']}",
                }
                self.data.append(rec)

    def __getitem__(self, idx):
        def _get_video_and_caption(rec):
            vid = rec["id"]
            udl_idx = vid.rindex("_")
            video_id, clip_id = vid[:udl_idx], int(vid[udl_idx + 1:])
            clip = self.youcook_annotation[video_id]["annotations"][clip_id]
            start, end = clip["segment"]
            if self.use_annotation_caption:
                caption = clip["sentence"]
            else:
                caption = rec["caption"]
            return (video_id, start, end), caption

        rec = self.data[idx]
        video_info, text_info = _get_video_and_caption(rec)
        # print(video_info, text_info)
        return video_info, text_info


class YoucookVideoProcessor(VideoProcessor):
    """video_fn is a tuple of (video_id, start, end) now."""

    def __call__(self, video_fn):
        video_id, start, end = video_fn
        feat = np.load(os.path.join(self.vfeat_dir, video_id + ".npy"))
        return feat[start:end]


class YoucookNLGMetaProcessor(MetaProcessor):
    """NLG uses the original split:
    `train_list.txt` and `val_list.txt`
    """

    def __init__(self, config):
        super().__init__(config)
        vfeat_dir = config.vfeat_dir
        print(self._get_split_path(config))
        with open(self._get_split_path(config)) as fd:
            video_ids = [
                line.strip().split("/")[1] for line in fd.readlines()]
            print("total video_ids in train/val_list.txt", len(video_ids))

            all_valid_video_ids = set(
                [os.path.splitext(fn)[0] for fn in os.listdir(vfeat_dir)]
            )
            video_ids = [
                video_id for video_id in video_ids
                if video_id in all_valid_video_ids]

            print("valid video_ids in train/val_list.txt", len(video_ids))
        with open(config.trainval_annotation) as fd:
            self.youcook_annotation = json.load(fd)["database"]

        data = []
        for video_id in video_ids:
            for clip in self.youcook_annotation[video_id]["annotations"]:
                start, end = clip["segment"]
                caption = clip["sentence"]
                data.append(((video_id, start, end), caption))
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]


# --------------------- CrossTask -------------------------

class CrossTaskMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        np.random.seed(0)  # deterministic random split.
        task_vids = self._get_vids(
            config.train_csv_path,
            config.vfeat_dir,
            config.annotation_path)

        val_vids = self._get_vids(
            config.val_csv_path,
            config.vfeat_dir,
            config.annotation_path)

        # filter out those task and vids appear in val_vids.
        task_vids = {
            task: [
                vid for vid in vids
                if task not in val_vids or vid not in val_vids[task]]
            for task, vids in task_vids.items()}

        primary_info = self._read_task_info(config.primary_path)
        test_tasks = set(primary_info['steps'].keys())

        # if args.use_related:
        related_info = self._read_task_info(config.related_path)
        task_steps = {**primary_info['steps'], **related_info['steps']}
        n_steps = {**primary_info['n_steps'], **related_info['n_steps']}
        # else:
        #     task_steps = primary_info['steps']
        #     n_steps = primary_info['n_steps']
        all_tasks = set(n_steps.keys())
        # filter and keep task in primary or related.
        task_vids = {
            task: vids for task, vids in task_vids.items()
            if task in all_tasks}
        # vocab-by-step matrix (A) and vocab (M)
        # (huxu): we do not use BoW.
        # A, M = self._get_A(task_steps, share="words")

        train_vids, test_vids = self._random_split(
            task_vids, test_tasks, config.n_train)
        print("train_num_videos", sum(len(vids) for vids in train_vids.values()))
        print("test_num_videos", sum(len(vids) for vids in test_vids.values()))
        # added by huxu to automatically determine the split.
        split_map = {
            "train": train_vids,
            "valid": test_vids,
            "test": test_vids
        }
        task_vids = split_map[config.split]

        self.vids = []
        for task, vids in task_vids.items():
            self.vids.extend([(task, vid) for vid in vids])
        self.task_steps = task_steps
        self.n_steps = n_steps

    def __getitem__(self, idx):
        task, vid = self.vids[idx]
        n_steps = self.n_steps[task]
        steps = self.task_steps[task]
        assert len(steps) == n_steps
        return (task, vid, steps, n_steps), (task, vid, steps, n_steps)

    def __len__(self):
        return len(self.vids)

    def _random_split(self, task_vids, test_tasks, n_train):
        train_vids = {}
        test_vids = {}
        for task, vids in task_vids.items():
            if task in test_tasks and len(vids) > n_train:
                train_vids[task] = np.random.choice(
                    vids, n_train, replace=False).tolist()
                test_vids[task] = [
                    vid for vid in vids if vid not in train_vids[task]]
            else:
                train_vids[task] = vids
        return train_vids, test_vids

    def _get_vids(self, path, vfeat_dir, annotation_path):
        """refactored from
        https://github.com/DmZhukov/CrossTask/blob/master/data.py
        changes: add `vfeat_dir` to check if the video is available.
        add `annotation_path` to check if the video is available.
        """

        task_vids = {}
        with open(path, 'r') as f:
            for line in f:
                task, vid, url = line.strip().split(',')
                # double check the video is available.
                if not os.path.exists(
                        os.path.join(vfeat_dir, vid + ".npy")):
                    continue
                # double check the annotation is available.
                if not os.path.exists(os.path.join(
                        annotation_path,
                        task + "_" + vid + ".csv")):
                    continue
                if task not in task_vids:
                    task_vids[task] = []
                task_vids[task].append(vid)
        return task_vids

    def _read_task_info(self, path):
        titles = {}
        urls = {}
        n_steps = {}
        steps = {}
        with open(path, 'r') as f:
            idx = f.readline()
            while idx != '':
                idx = idx.strip()
                titles[idx] = f.readline().strip()
                urls[idx] = f.readline().strip()
                n_steps[idx] = int(f.readline().strip())
                steps[idx] = f.readline().strip().split(',')
                next(f)
                idx = f.readline()
        return {
            'title': titles,
            'url': urls,
            'n_steps': n_steps,
            'steps': steps
        }

    def _get_A(self, task_steps, share="words"):
        raise ValueError("running get_A is not allowed for BERT.")
        """Step-to-component matrices."""
        if share == 'words':
            # share words
            task_step_comps = {
                task: [step.split(' ') for step in steps]
                for task, steps in task_steps.items()}
        elif share == 'task_words':
            # share words within same task
            task_step_comps = {
                task: [[task+'_'+tok for tok in step.split(' ')] for step in steps]
                for task, steps in task_steps.items()}
        elif share == 'steps':
            # share whole step descriptions
            task_step_comps = {
                task: [[step] for step in steps] for task, steps in task_steps.items()}
        else:
            # no sharing
            task_step_comps = {
                task: [[task+'_'+step] for step in steps]
                for task, steps in task_steps.items()}
        # BERT tokenizer here?
        vocab = []
        for task, steps in task_step_comps.items():
            for step in steps:
                vocab.extend(step)
        vocab = {comp: m for m, comp in enumerate(set(vocab))}
        M = len(vocab)
        A = {}
        for task, steps in task_step_comps.items():
            K = len(steps)
            a = torch.zeros(M, K)
            for k, step in enumerate(steps):
                a[[vocab[comp] for comp in step], k] = 1
            a /= a.sum(dim=0)
            A[task] = a
        return A, M


class CrossTaskVideoProcessor(VideoProcessor):
    def __call__(self, video_fn):
        task, vid, steps, n_steps = video_fn
        video_fn = os.path.join(self.vfeat_dir, vid + ".npy")
        feat = np.load(video_fn)
        return feat


class CrossTaskTextProcessor(TextProcessor):
    def __call__(self, text_id):
        task, vid, steps, n_steps = text_id
        step_ids = []
        for step_str in steps:
            step_ids.append(
                self.tokenizer(step_str, add_special_tokens=False)["input_ids"]
            )
        return step_ids


class CrossTaskAligner(Aligner):
    """
    TODO: it's not clear yet the formulation of the task; finish this later.
    """
    def __init__(self, config):
        super().__init__(config)
        self.annotation_path = config.annotation_path
        self.sliding_window = config.sliding_window
        self.sliding_window_size = config.sliding_window_size

    def __call__(self, video_id, video_feature, text_feature):
        task, vid, steps, n_steps = video_id
        annot_path = os.path.join(
            self.annotation_path, task + '_' + vid + '.csv')
        video_len = len(video_feature)

        labels = torch.from_numpy(self._read_assignment(
            video_len, n_steps, annot_path)).float()

        vfeats, vmasks, targets = [], [], []
        # sliding window on video features and targets.
        for window_start in range(0, video_len, self.sliding_window):
            video_start = 0
            video_end = min(video_len - window_start, self.sliding_window_size)
            video_clip = {"start": [video_start], "end": [video_end]}

            vfeat, vmask = self._build_video_seq(
                video_feature[window_start: window_start + video_end],
                video_clip
            )

            target = labels[window_start: window_start + video_end]
            assert len(vfeat) >= len(target), "{},{}".format(len(vfeat), len(target))
            # TODO: randomly drop all zero targets for training ?
            # if self.split == "train" and target.sum() == 0:
            #     continue
            vfeats.append(vfeat)
            vmasks.append(vmask)
            targets.append(target)

            if (video_len - window_start) <= self.sliding_window_size:
                break

        vfeats = torch.stack(vfeats)
        vmasks = torch.stack(vmasks)
        targets = torch.cat(targets, dim=0)

        caps, cmasks = [], []
        for step in text_feature:
            step_text_feature = {"start": [0], "end": [1], "cap": [step]}
            step_text_clip_index = [0]
            cap, cmask = self._build_text_seq(
                step_text_feature, step_text_clip_index
            )
            caps.append(cap)
            cmasks.append(cmask)
        caps = torch.stack(caps)
        cmasks = torch.stack(cmasks)

        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,  # X for original code.
            "vmasks": vmasks,
            "targets": targets,
            "video_id": vid,
            "task": task,
            "video_len": video_len  # for later checking.
        }

    def _read_assignment(self, T, K, path):
        """
        refactored from https://github.com/DmZhukov/CrossTask/blob/master/data.py
        Howto interpret contraints on loss that is going to be minimized:
        lambd is a big number;
        self.lambd * C is a big number for all valid position (csv stores invalids)

        def forward(self, O, Y, C):
            return (Y*(self.lambd * C - self.lsm(O))).mean(dim=0).sum()

        This will load the csv file and fill-in the step col from start to end rows.
        """

        Y = np.zeros([T, K], dtype=np.uint8)
        with open(path, 'r') as f:
            for line in f:
                step, start, end = line.strip().split(',')
                start = int(math.floor(float(start)))
                end = int(math.ceil(float(end)))
                step = int(step) - 1
                Y[start:end, step] = 1
        return Y


# --------------------- COIN -------------------------

class MetaTextBinarizer(Aligner):
    def __call__(self, text_feature):
        text_feature = {
            "cap": [text_feature],
            "start": [0.],
            "end": [100.],
        }
        text_clip_indexs = [0]

        caps, cmasks = self._build_text_seq(
            text_feature, text_clip_indexs
        )
        return {"caps": caps, "cmasks": cmasks}


class COINActionSegmentationMetaProcessor(MetaProcessor):
    split_map = {
        "train": "training",
        "valid": "testing",
        "test": "testing",
    }

    def __init__(self, config):
        super().__init__(config)
        with open(self._get_split_path(config)) as fr:
            database = json.load(fr)["database"]
        id2label = {}
        data = []
        # filter the data by split.
        for video_id, rec in database.items():
            # always use testing to determine label_set
            if rec["subset"] == "testing":
                for segment in rec["annotation"]:
                    id2label[int(segment["id"])] = segment["label"]
        # text_labels is used for ZS setting
        self.text_labels = ["none"] * len(id2label)
        for label_id in id2label:
            self.text_labels[label_id-1] = id2label[label_id]

        id2label[0] = "O"
        print("num of labels", len(id2label))

        for video_id, rec in database.items():
            if not os.path.isfile(os.path.join(config.vfeat_dir, video_id + ".npy")):
                continue
            if rec["subset"] == COINActionSegmentationMetaProcessor.split_map[self.split]:
                starts, ends, labels = [], [], []
                for segment in rec["annotation"]:
                    start, end = segment["segment"]
                    label = int(segment["id"])
                    starts.append(start)
                    ends.append(end)
                    labels.append(label)
                data.append(
                    (video_id, {"start": starts, "end": ends, "label": labels}))
        self.data = data

    def meta_text_labels(self, config):
        from transformers import default_data_collator
        from ..utils import get_local_rank

        text_processor = TextProcessor(config)
        binarizer = MetaTextBinarizer(config)
        # TODO: add prompts to .yaml.
        text_labels = [label for label in self.text_labels]

        if get_local_rank() == 0:
            print(text_labels)

        outputs = []
        for text_label in text_labels:
            text_feature = text_processor(text_label)
            outputs.append(binarizer(text_feature))
        return default_data_collator(outputs)

    def __getitem__(self, idx):
        return self.data[idx]


class COINActionSegmentationTextProcessor(TextProcessor):
    def __call__(self, text_label):
        return text_label


class COINActionSegmentationAligner(Aligner):
    def __init__(self, config):
        super().__init__(config)
        self.sliding_window = config.sliding_window
        self.sliding_window_size = config.sliding_window_size

    def __call__(self, video_id, video_feature, text_feature):
        starts, ends, label_ids = text_feature["start"], text_feature["end"], text_feature["label"]
        # sliding window.
        video_len = len(video_feature)

        vfeats, vmasks, targets = [], [], []
        # sliding window on video features and targets.
        for window_start in range(0, video_len, self.sliding_window):
            video_start = 0
            video_end = min(video_len - window_start, self.sliding_window_size)
            video_clip = {"start": [video_start], "end": [video_end]}
            vfeat, vmask = self._build_video_seq(
                video_feature[window_start: window_start + video_end],
                video_clip
            )
            # covers video length only.
            target = torch.full_like(vmask, -100, dtype=torch.long)
            target[vmask] = 0
            for start, end, label_id in zip(starts, ends, label_ids):
                if (window_start < end) and (start < (window_start + video_end)):
                    start_offset = max(0, math.floor(start) - window_start)
                    end_offset = min(video_end, math.ceil(end) - window_start)
                    target[start_offset:end_offset] = label_id
            vfeats.append(vfeat)
            vmasks.append(vmask)
            targets.append(target)
            if (video_len - window_start) <= self.sliding_window_size:
                break

        vfeats = torch.stack(vfeats)
        vmasks = torch.stack(vmasks)
        targets = torch.stack(targets)
        video_targets = torch.full((video_len,), 0)
        for start, end, label_id in zip(starts, ends, label_ids):
            start_offset = max(0, math.floor(start))
            end_offset = min(video_len, math.ceil(end))
            video_targets[start_offset:end_offset] = label_id

        caps = torch.LongTensor(
            [[self.cls_token_id, self.sep_token_id,
              self.pad_token_id, self.sep_token_id]],
            ).repeat(vfeats.size(0), 1)
        cmasks = torch.BoolTensor(
            [[0, 1, 0, 1]]  # pad are valid for attention.
            ).repeat(vfeats.size(0), 1)
        return {
            "caps": caps,
            "cmasks": cmasks,
            "vfeats": vfeats,  # X for original code.
            "vmasks": vmasks,
            "targets": targets,
            "video_id": video_id,
            "video_len": video_len,  # for later checking.
            "video_targets": video_targets
        }


class DiDeMoMetaProcessor(MetaProcessor):
    """reference: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/data_processing.py
    """
    def __init__(self, config):
        super().__init__(config)

        assert "test" in self._get_split_path(config), "DiDeMo only supports zero-shot testing for now."

        with open(self._get_split_path(config)) as data_file:
            json_data = json.load(data_file)

        data = []
        for record in json_data:
            data.append((record["video"], record["description"]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DiDeMoTextProcessor(TextProcessor):
    """reference: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/data_processing.py
    """

    def __call__(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]


class DiDeMoAligner(DSAligner):
    """
    check video length.
    """

    def __call__(self, video_id, video_feature, text_feature):
        # print(video_feature.shape[0])
        return super().__call__(video_id, video_feature, text_feature)


# -------------------- SignCLIP common -----------------------

from pose_format import Pose
from pose_format.utils.normalization_3d import PoseNormalizer
from pose_format.utils.generic import pose_normalization_info
# from pose_format.utils.generic import flip_holistic, is_left_handed, pose_normalization_info

import mediapipe as mp
mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]


class PoseProcessor(VideoProcessor):
    def __init__(self, config):
        super().__init__(config)
        self.pose_components = config.pose_components
        self.normalize_hand = config.normalize_hand
        self.flip_pose = config.flip_pose
        self.augment2d = config.augment2d
        self.augment_temporal = config.augment_temporal
        self.gaussian_noise = config.gaussian_noise
        self.max_video_len = config.max_video_len
        self.preprocess = config.preprocess
        self.anonym_pose = config.anonym_pose
        self.is_training = (config.split == 'train') and (not config.train_for_test)

        np.random.seed(42)

    def __call__(self, video_id, pose=None):
        if video_id:
            with open(os.path.join(self.vfeat_dir, video_id + ".pose"), "rb") as f:
                buffer = f.read()
                pose = Pose.read(buffer)

        # select components
        if self.pose_components:
            if self.pose_components == 'reduced_face':
                pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                    {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})
            else:
                pose = pose.get_components(self.pose_components)
                # 3D Hand Normalization
                if self.pose_components == ['RIGHT_HAND_LANDMARKS'] and self.normalize_hand:
                    pose = self.hand_normalization(pose)

        if self.flip_pose == 'right':
            # if is_left_handed(pose):
            if np.nan_to_num(pose.get_components(["RIGHT_HAND_LANDMARKS"]).body.data).var(axis=0).sum() == 0:
                pose = flip_holistic(pose)

        if self.anonym_pose:
            from pose_anonymization.appearance import remove_appearance
            # remove appearance + add spreadthesign mean pose
            # https://github.com/sign-language-processing/pose-anonymization
            pose = remove_appearance(pose)

        if self.preprocess == 'sign-vq' or self.preprocess == 'sign-vq-original-scale':
            from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
            # reuse the preprocessing pipeline from sign-vq
            # https://github.com/sign-language-processing/sign-vq
            pose = pre_process_mediapipe(pose)

            if not self.preprocess == 'sign-vq-original-scale':
                # this removes spreadthesign mean pose
                pose = normalize_mean_std(pose)
        else:
            # normalize pose: the mean distance between the shoulders of each person equals 1
            pose = pose.normalize(self.pose_normalization_info(pose.header))
            pose = self.pose_hide_legs(pose)

        # augmentation (training only)
        if self.is_training:
            if self.flip_pose and random.random() < self.flip_pose:
                # TODO: move the flip_pose function into pose-format

                # CAUTION: flipping works on reduced set of key points only
                FLIPPED_COMPONENTS = ["POSE_LANDMARKS", "FACE_LANDMARKS", "RIGHT_HAND_LANDMARKS", "LEFT_HAND_LANDMARKS"]
                # FLIPPED_BODY_POINTS = ['RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_HIP', 'LEFT_HIP']
                FLIPPED_BODY_POINTS = ['NOSE', 'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER', 'RIGHT_EAR', 'LEFT_EAR', 'MOUTH_RIGHT', 'MOUTH_LEFT', 'RIGHT_SHOULDER', 'LEFT_SHOULDER', 'RIGHT_ELBOW', 'LEFT_ELBOW', 'RIGHT_WRIST', 'LEFT_WRIST', 'RIGHT_PINKY', 'LEFT_PINKY', 'RIGHT_INDEX', 'LEFT_INDEX', 'RIGHT_THUMB', 'LEFT_THUMB', 'RIGHT_HIP', 'LEFT_HIP', 'RIGHT_KNEE', 'LEFT_KNEE', 'RIGHT_ANKLE', 'LEFT_ANKLE', 'RIGHT_HEEL', 'LEFT_HEEL', 'RIGHT_FOOT_INDEX', 'LEFT_FOOT_INDEX']
                # face flipping based on https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
                FLIPPED_FACE_POINTS = ['0', '249', '10', '13', '14', '17', '251', '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300', '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361', '362', '365', '373', '374', '375', '377', '378', '379', '152', '380', '381', '382', '384', '385', '386', '387', '388', '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466', \
                                            '7', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63', '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109', '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '153', '154', '155', '157', '158', '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246']
                pose = pose.flip(0).get_components(FLIPPED_COMPONENTS, {"POSE_LANDMARKS": FLIPPED_BODY_POINTS, "FACE_LANDMARKS": FLIPPED_FACE_POINTS})
                pose = pose.normalize(self.pose_normalization_info(pose.header))
            if self.augment2d:
                pose = pose.augment2d()
            if self.augment_temporal and pose.body.data.shape[0] > 1:
                old_fps = pose.body.fps
                ratio = np.random.normal(loc=1, scale=0.2)
                if pose.body.data.shape[0] * ratio < self.max_video_len:
                    new_fps = round(old_fps * ratio)
                    pose = pose.interpolate(new_fps, kind='linear')
            if self.gaussian_noise:
                noise_std = 0.001
                noise = np.random.normal(scale=noise_std, size=pose.body.data.shape)
                pose.body.data = pose.body.data + noise

        feat = np.nan_to_num(pose.body.data)
        feat = feat.reshape(feat.shape[0], -1)
        
        return feat

    def pose_normalization_info(self, pose_header):
        if pose_header.components[0].name == "POSE_LANDMARKS":
            return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                                p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

        if pose_header.components[0].name == "BODY_135":
            return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

        if pose_header.components[0].name == "pose_keypoints_2d":
            return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                                p2=("pose_keypoints_2d", "LShoulder"))

        raise ValueError("Unknown pose header schema for normalization")

    def hand_normalization(self, pose):
        plane = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "PINKY_MCP"),
            p3=("RIGHT_HAND_LANDMARKS", "INDEX_FINGER_MCP")
        )
        line = pose.header.normalization_info(
            p1=("RIGHT_HAND_LANDMARKS", "WRIST"),
            p2=("RIGHT_HAND_LANDMARKS", "MIDDLE_FINGER_MCP")
        )
        normalizer = PoseNormalizer(plane=plane, line=line, size=100)
        tensor = normalizer(pose.body.data)

        pose.body.data = tensor
        pose.focus()

        return pose

    def pose_hide_legs(self, pose):
        if pose.header.components[0].name == "POSE_LANDMARKS":
            point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
            # pylint: disable=protected-access
            points = [
                pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
                for n in point_names
                for side in ["LEFT", "RIGHT"]
            ]
            pose.body.confidence[:, :, points] = 0
            pose.body.data[:, :, points, :] = 0
            return pose
        else:
            raise ValueError("Unknown pose header schema for hiding legs")


# -------------------- RWTH Fingerspelling -----------------------


class RWTHFSMetaProcessor(MetaProcessor):
    """RWTH German Fingerspelling Database
    https://www-i6.informatik.rwth-aachen.de/aslr/fingerspelling.php
    """

    def __init__(self, config):
        super().__init__(config)

        vfeat_dir = config.vfeat_dir
        split_path = self._get_split_path(config)

        self.letter_to_id = {}
        self.id_to_letter = {}

        with open(config.gesture_id_path) as f:
            for idx, line in enumerate(f):
                letter = line.split(' = ')[1].rstrip('\n')
                self.letter_to_id[letter] = idx + 1
                self.id_to_letter[str(idx + 1)] = letter

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = line.rstrip('\n') 
                signer_id, letter_id, seq_id, camera_id = video_id.split('_')

                # FIXME: for now we do full body pose estimation for all videos, so exclude cam1 where only the hands are present
                if config.video_processor == 'RWTHFSPoseProcessor' and camera_id == 'cam1':
                    continue
                
                lines.append(video_id)

            if config.split == 'train':
                self.data = []

                video_ids = defaultdict(list)
                for video_id in lines:
                    signer_id, letter_id, seq_id, camera_id = video_id.split('_')
                    video_ids[self.id_to_letter[letter_id]].append(video_id)

                length = []
                for key, value in video_ids.items():
                    length.append(len(value))
                max_length = max(length)

                for i in range(max_length):
                    for key, value in video_ids.items():
                        self.data.append(value[i % len(value)])
            else:
                self.data = lines

    def __getitem__(self, idx):
        video_id = self.data[idx]
        signer_id, letter_id, seq_id, camera_id = video_id.split('_')
        body_part = 'handshape' if camera_id == 'cam1' else 'whole body'
        text_info = f'Fingerspell the letter {self.id_to_letter[letter_id]} in German Sign Language.'
        # print(video_id, text_info)
        return video_id, text_info


class RWTHFSVideoProcessor(VideoProcessor):
    def __call__(self, video_id):
        feat = np.load(os.path.join(self.vfeat_dir, video_id + ".npy"))
        # pooling adapater (not needed when training from scratch)
        feat_dim = 512
        if feat.shape[1] > feat_dim and not self.vfeat_custom:
            # i3d feature is 1024
            # adapt feature dimension to 512 by average pooling
            feat = feat.reshape(feat.shape[0], feat_dim, int(feat.shape[1] / feat_dim))
            feat = np.average(feat, axis=2)
        return feat


class RWTHFSPoseProcessor(PoseProcessor):
    pass

# -------------------- ASL Signs -----------------------

class ASLSignMetaProcessor(MetaProcessor):
    """Google - Isolated Sign Language Recognition
    https://www.kaggle.com/competitions/asl-signs/overview
    """

    def __init__(self, config):
        super().__init__(config)

        vfeat_dir = config.vfeat_dir
        split_path = self._get_split_path(config)
        metadata_df = pd.read_csv(config.metadata_path, dtype=str)

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = line.rstrip('\n') 
                lines.append(video_id)

            metadata_df = metadata_df[metadata_df['sequence_id'].isin(lines)]
            data = metadata_df.to_dict('records')

            print(f'sign distribution in the {config.split} set:')
            print(metadata_df.groupby(['sign'])['sign'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            if config.split == 'train':
                self.data = []

                indices = defaultdict(list)
                for index, item in enumerate(data):
                    indices[item['sign']].append(index)

                length = []
                for key, value in indices.items():
                    length.append(len(value))
                max_length = max(length)

                for i in range(max_length):
                    for key, value in indices.items():
                        self.data.append(data[value[i % len(value)]])
            else:
                self.data = data

    def __getitem__(self, idx):
        video_id = self.data[idx]['path'].replace('train_landmark_files/', '')
        text_info = f'Sign the sign "{self.data[idx]["sign"]}" in American Sign Language.'
        # print(video_id, text_info)
        return video_id, text_info


class ASLSignPoseProcessor(PoseProcessor):
    NOSE = [
        1,2,98,327
    ]
    LIP = [ 0, 
        61, 185, 40, 39, 37, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ]
    REYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        246, 161, 160, 159, 158, 157, 173,
    ]
    LEYE = [
        263, 249, 390, 373, 374, 380, 381, 382, 362,
        466, 388, 387, 386, 385, 384, 398,
    ]
    FACE = sorted(NOSE + LIP + REYE + LEYE)
    FACE_FULL = np.arange(0, 468).tolist()

    LHAND = np.arange(468, 489).tolist()
    POSE = np.arange(489, 522).tolist()
    RHAND = np.arange(522, 543).tolist()
    BODY = LHAND + POSE + RHAND

    def __call__(self, video_id):
        import pyarrow.parquet as pq

        pose_df = pq.read_table(os.path.join(self.vfeat_dir, video_id)).to_pandas()

        # pd.set_option('display.max_rows', None)
        # print(pose_df[pose_df['frame'] == 18])

        # pose_df = pose_df[pose_df['type'].isin(self.pose_components)]

        points = []
        if "face" in self.pose_components:
            points = points + self.FACE
        if "face_full" in self.pose_components:
            points = points + self.FACE_FULL
        if "left_hand" in self.pose_components:
            points = points + self.LHAND
        if "pose" in self.pose_components:
            points = points + self.POSE    
        if "right_hand" in self.pose_components:
            points = points + self.RHAND

        num_frames = len(pose_df['frame'].drop_duplicates())
        dimensions = ['x', 'y', 'z']

        pose_data = pose_df[dimensions].to_numpy().reshape(num_frames, -1, len(dimensions))
        pose_data = pose_data[:, points, :]

        pose_data = pose_data.reshape(num_frames, -1)
        pose_data = np.nan_to_num(pose_data)
        
        return pose_data


# -------------------- SignCLIP v1 -----------------------

import re
import string
import importlib
from tqdm import tqdm

import tensorflow_datasets as tfds
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig

from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader


class SignCLIPMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        self.config = config
        self.task = config.task
        self.split = config.split
        self.pose_processer = SignCLIPPoseProcessor(config) # call pose_processer by meta_processor itself
        self.datasets = {}
        self.data = []

        if config.test_in_vocab:
            vocab_path = './data_stat_sp_concept_dis.csv'
            vocab_df = pd.read_csv(vocab_path)
            vocab_df = vocab_df[vocab_df['count'] > 20]
            self.vocab = list(vocab_df['text'])

        print('================================')
        print(f'Loading {self.split} data ... ')
        print('================================')

        datasets = config[f'{"test" if config.train_for_test else self.split}_datasets']
        datasets = [item if len(item) == 3 else [*item, None] for item in datasets]

        for dataset, version, split_version in datasets:
            print('--------------------------------')
            print(f'Loading the {dataset} {version} dataset, {split_version if split_version else "default"} split ... ')
            print('--------------------------------')

            # read common pose header for the dataset
            dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
            with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))

            sd_config = SignDatasetConfig(name="holistic", version=version, include_video=False, include_pose="holistic", extra={'split': split_version} if split_version else {})
            splits = ['validation' if self.split == 'valid' else self.split]
            # utilize unused validation data from some datasets for pretraining as well
            if self.split == 'train' and config.use_valid_for_pretraining and dataset not in [d[0] for d in config.valid_datasets]:
                splits = ['train', 'validation'] 

            for split in splits:
                data_l = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split]

                print(f'In total {len(data_l)} raw {split} examples.')
                print('Iterate over examples ...')

                self.datasets[dataset] = {
                    'pose_header': pose_header,
                    'data_l': data_l,
                }

                count = 0
                for dataset_index, datum in enumerate(tqdm(data_l)):
                    if not (datum['pose']['data'].shape[0] > 0 and datum['pose']['data'].shape[0] <= config.max_video_len):
                        continue

                    if config.debug and count >= 1000:
                        break
                    count = count + 1

                    if self.task and self.task.startswith('identification'):
                        # for dicta_sign
                        signed_language_map = {
                            'GSL': 'gss',
                            'BSL': 'bfi',
                            'DGS': 'gsg',
                            'LSF': 'fsl',
                        }
                        signed_language = signed_language_map[datum['signed_language'].numpy().decode('utf-8')]

                        # DGS and BSL videos contain more than one camera angle, excluding for now
                        if signed_language == 'gsg' or signed_language == 'bfi':
                            continue
                        # if signed_language == 'gsg' or signed_language == 'gss':
                        #     continue
                        
                        tag_prompt = f"<en> <{signed_language}>"
                        text_prompt = f"{tag_prompt} {datum['text_en'].numpy().decode('utf-8')}" if self.task == 'identification_oracle' else tag_prompt
                    else:
                        if config.sp_universal_tagging:
                            if dataset == 'bobsl_islr':
                                tag_prompt = "<en> <bfi>" 
                            else:
                                tag_prompt = "<en> <ase>" 
                        else:
                            tag_prompt = "<American Sign Language>"

                        text_content = datum['text'].numpy().decode('utf-8')

                        if config.test_in_vocab:
                            if dataset == 'asl_citizen':
                                text_content = text_content.lower()
                                text_content = text_content.rstrip(string.digits)
                            elif dataset == 'sem_lex':
                                text_content = text_content.rstrip(string.digits)
                                text_content = text_content.rstrip('_')
                                # text_content = re.sub(r'_\d+$', '', text_content)
                                text_content = text_content.replace('_', ' ')

                            if text_content not in self.vocab:
                                continue

                        text_prompt = f"{tag_prompt} {text_content}"

                    vfeat = None
                    # save memory consumption for 3.5M BOBSL ISLR examples to under 300GB
                    # better solution: use SignCLIPMetaProcessorV2 to load data asynchronously
                    if config.pre_compute_vfeat: 
                        # reconstruct pose object
                        tf_pose = datum['pose']
                        fps = int(tf_pose["fps"].numpy())
                        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
                        pose = Pose(pose_header, pose_body)
                        vfeat = self.pose_processer(pose)

                    self.data.append(dict(
                        # datum,
                        id=f"{dataset}_{datum['id'].numpy().decode('utf-8')}",
                        text=text_prompt,
                        vfeat=vfeat,
                    ))

                # if config.debug:
                #     self.data = self.data*3000

                print(f'In total {count} wellformed {split} examples.')

        print(f'In total {len(self.data)} wellformed {split} examples from all datasets.')
        
        if self.split == 'train':
            random.shuffle(self.data)

        # Group examples by text prompts
        self.text_to_idxs = defaultdict(list)
        for idx, datum in enumerate(self.data):
            self.text_to_idxs[datum['text']].append(idx)

        print('Number of examples grouped by the text prompts:')
        text_to_idxs_num = [(text, len(idxs)) for text, idxs in self.text_to_idxs.items()]
        text_to_idxs_num = sorted(text_to_idxs_num, key=lambda x: x[1], reverse=True)
        for i, entry in enumerate(text_to_idxs_num):
            if i < 10 or (len(text_to_idxs_num) - i < 10):
                print(entry)
            elif i == 10:
                print('...')
        print('Total classes:', len(text_to_idxs_num))
        
        # unique sampler: sample config.unique_sampler_num examples of different text prompts randomly (for a batch)
        if self.split == 'train' and config.unique_sampler_num:
            if config.unique_sampler_num > len(text_to_idxs_num):
                raise ValueError(f'Impossible to sample {config.unique_sampler_num} unique examples given {len(text_to_idxs_num)} unique text prompts.')

            self.unique_sampler_num = config.unique_sampler_num
            self.text_prompts = list(self.text_to_idxs.keys())
            self.text_prompts_sampled = []

    def __getitem__(self, idx):
        if hasattr(self, 'unique_sampler_num'):
            # reset when starting a new epoch or when a batch is full
            if idx == 0 or len(self.text_prompts_sampled) == self.unique_sampler_num:
                self.text_prompts = list(self.text_to_idxs.keys())
                self.text_prompts_sampled = []
                # print('reset')

            # randomly sample one example per text prompt
            sampled_text = random.choice(self.text_prompts)
            sampled_idx = random.choice(self.text_to_idxs[sampled_text])
            self.text_prompts.remove(sampled_text)
            self.text_prompts_sampled.append(sampled_text)

            # print(sampled_idx, sampled_text)
            return sampled_idx, sampled_text
        else:
            datum = self.data[idx]

            # vfeat pre-computed during __init__
            if 'vfeat' in datum:
                return idx, datum['text'], datum['vfeat']

            # reconstruct pose object
            tf_pose = datum['pose']
            # tf_pose = dataset['data_l'][datum['dataset_index']]['pose']
            fps = int(tf_pose["fps"].numpy())
            pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
            pose = Pose(dataset['pose_header'], pose_body)
            vfeat = self.pose_processer(pose)

            return idx, datum['text'], vfeat


class SignCLIPPoseProcessor(PoseProcessor):
    def __call__(self, pose):
        # the pose objects are passed to PoseProcessor
        feat = super().__call__(None, pose)
        return feat
        

# -------------------- SignCLIP pretrained on Spreadthesign -----------------------

class SignCLIPPretrainMetaProcessor(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        if config.debug:
            config.split = 'test'

        self.vfeat_dir = config.vfeat_dir
        self.task = config.task
        split_path = self._get_split_path(config)
        metadata_df = pd.read_csv(config.metadata_path, dtype=str)

        with open(split_path) as f:
            lines = []
            for line in f:
                video_id = int(line.rstrip('\n'))
                lines.append(video_id)

            metadata_df = metadata_df.reset_index()
            metadata_df = metadata_df[metadata_df['index'].isin(lines)]
            metadata_df = metadata_df[metadata_df['language'] == 'en']

            print(metadata_df)

            print(f'text distribution in the {config.split} set:')
            print(metadata_df.groupby(['text'])['text'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            print(f'language distribution in the {config.split} set:')
            print(metadata_df.groupby(['videoLanguage'])['videoLanguage'].count().reset_index(name='count').sort_values(['count'], ascending=False))

            data = metadata_df.to_dict('records')
            data = [datum for datum in tqdm(data) if os.path.exists(os.path.join(config.vfeat_dir, datum['pose']))]

            print(f'In total {len(data)} {config.split} examples with poses.')

            self.data = data

            if config.split == 'train':
                random.shuffle(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]
        video_id = datum['pose'].replace('.pose', '')
        text = '' if self.task == 'identification' else datum['text']
        vlan = '<ase>' if self.task == 'conceptualization' else f"<{datum['videoLanguage']}>"
        text_info = f"<{datum['language']}> {vlan} {text}"
        return video_id, text_info


class SignCLIPMetaProcessorV2(MetaProcessor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(42)

        self.config = config
        self.split = config.split
        self.pose_processer = SignCLIPPoseProcessor(config) # call pose_processer by meta_processor itself

        print('================================')
        print(f'Loading {self.split} data ... ')
        print('================================')

        datasets = config[f'{"test" if config.train_for_test else self.split}_datasets']
        datasets = [item if len(item) == 3 else [*item, None] for item in datasets]

        for dataset, version, split_version in datasets:
            print('--------------------------------')
            print(f'Loading the {dataset} {version} dataset, {split_version if split_version else "default"} split ... ')
            print('--------------------------------')

            # read common pose header for the dataset
            dataset_module = importlib.import_module("sign_language_datasets.datasets." + dataset + "." + dataset)
            with open(dataset_module._POSE_HEADERS['holistic'], "rb") as buffer:
                pose_header = PoseHeader.read(BufferReader(buffer.read()))

            sd_config = SignDatasetConfig(name="holistic", version=version, include_video=False, include_pose="holistic", extra={'split': split_version} if split_version else {})
            split = 'validation' if self.split == 'valid' else self.split

            self.pose_header = pose_header

            self.data_l = tfds.load(name=dataset, builder_kwargs=dict(config=sd_config), data_dir=config.data_dir)[split]
            print("Dataset initialized")
            print(f'In total {len(self.data_l)} raw {split} examples.')


    def reset_data_iter(self):
        print("Resetting iterator")
        self.data_iter = iter(self.data_l)


    def __len__(self): 
        return len(self.data_l)


    def __getitem__(self, idx):
        # print(f"Fetching item {idx}")

        if idx == 0:
            self.reset_data_iter()

        try:
            # Get the next data from the TensorFlow generator
            datum = next(self.data_iter)
        except StopIteration:
            print("Iterator exhausted, resetting...")
            # If the generator is exhausted, reset it
            self.reset_data_iter()
            # Retrieve the first element from the reset generator
            datum = next(self.data_iter)

        tag_prompt = "<en> <bfi>" 
        text_content = datum['text'].numpy().decode('utf-8')
        text_prompt = f"{tag_prompt} {text_content}"

        # print(text_prompt)

        # reconstruct pose object
        tf_pose = datum['pose']
        fps = int(tf_pose["fps"].numpy())
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(self.pose_header, pose_body)
        vfeat = self.pose_processer(pose)

        return idx, text_prompt, vfeat