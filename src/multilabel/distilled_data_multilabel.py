from copy import copy
import json
import logging
import os
from functools import reduce
from abc import ABCMeta, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Union
from distilled_data import (
    DistilledAttentionLabels,
    DistilledDataConfig,
    DistilledFeature,
    DistilledInputEmbedding,
    DistilledLR,
    LearnerTrainConfig,
)

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class MultilabelDistilledLabel(DistilledFeature):
    def __init__(
        self,
        data_per_label: int = 1,
        num_labels: int = 2,
        num_elements: int = 2,
        label_type: Literal["hard", "soft", "unrestricted"] = "hard",
    ):
        self.label_type = label_type

        label_ids = torch.arange(num_elements * data_per_label) % num_labels
        self.data = torch.eye(num_labels)[label_ids]

        if label_type != "hard":
            self.data.requires_grad_()

    def __getitem__(self, index):
        if self.label_type == "soft":
            return self.data[index].softmax(dim=-1)

        return self.data[index]


class DistilledMultilabelData:
    def __init__(
        self,
        config: DistilledDataConfig,
        train_config: LearnerTrainConfig,
        task_num_labels: Dict[str, int],
        hidden_size: int = 768,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
    ):
        if not isinstance(config, DistilledDataConfig):
            config = DistilledDataConfig(**config)
        self.config = config

        if not isinstance(train_config, LearnerTrainConfig):
            train_config = LearnerTrainConfig(**train_config)
        self.train_config = train_config

        self.task_num_labels = task_num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.splitter = "-"

        if self.config.multilabel_labels_type == "max":
            self.num_elements = max(self.task_num_labels.values())
        elif self.config.multilabel_labels_type == "sum":
            self.num_elements = sum(self.task_num_labels.values())
        elif self.config.multilabel_labels_type == "mult":
            self.num_elements = reduce(
                lambda x, y: x * y, self.task_num_labels.values()
            )
        else:
            # print(self.config.data_per_label)
            raise NotImplementedError

        if self.config.fix_order:
            assert config.data_per_label % train_config.batch_size_per_label == 0

        self.inputs_embeds = DistilledInputEmbedding(
            data_per_label=config.data_per_label,
            num_labels=self.num_elements,
            seq_length=config.seq_length,
            hidden_size=hidden_size,
        )

        self.labels = {}
        for task_name, num_labels in self.task_num_labels.items():
            self.labels[task_name] = MultilabelDistilledLabel(
                data_per_label=config.data_per_label,
                num_labels=num_labels,
                num_elements=self.num_elements,
                label_type=config.label_type,
            )

        if self.config.separate_lr:
            self.lr = {}
            for task_name, num_labels in self.task_num_labels.items():
                self.lr[task_name] = DistilledLR(
                    lr_init=config.lr_init,
                    lr_for_step=config.lr_for_step,
                    lr_linear_decay=config.lr_linear_decay,
                    train_step=train_config.train_step,
                )
        else:
            self.lr = DistilledLR(
                lr_init=config.lr_init,
                lr_for_step=config.lr_for_step,
                lr_linear_decay=config.lr_linear_decay,
                train_step=train_config.train_step,
            )

        self.data: dict[str, DistilledFeature] = {
            "inputs_embeds": self.inputs_embeds,
            "labels": self.labels,
            "lr": self.lr,
        }

        # attention labels
        if config.attention_label_type in ("cls", "all"):
            if self.config.separate_attention_label:
                self.attention_labels = {}
                for task_name, num_labels in self.task_num_labels.items():
                    self.attention_labels[task_name] = DistilledAttentionLabels(
                        data_per_label=config.data_per_label,
                        num_labels=self.num_elements,
                        seq_length=config.seq_length,
                        num_layers=num_layers,
                        num_heads=num_heads,
                        attention_label_type=config.attention_label_type,
                    )
            else:
                self.attention_labels = DistilledAttentionLabels(
                    data_per_label=config.data_per_label,
                    num_labels=self.num_elements,
                    seq_length=config.seq_length,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    attention_label_type=config.attention_label_type,
                )
            self.data["attention_labels"] = self.attention_labels
        else:
            assert config.attention_label_type == "none"
            self.attention_labels = None

        self.attention_loss_lambda = self.config.attention_loss_lambda

    def get_batch(self, step: int, task_name: str):
        indices = self.get_batch_indices(step)

        return {
            "inputs_embeds": self.inputs_embeds[indices],
            "labels": self.labels[task_name][indices],
            "attention_labels": (
                (
                    self.attention_labels[task_name][indices]
                    if self.config.separate_attention_label
                    else self.attention_labels[indices]
                )
                if self.attention_labels is not None
                else None
            ),
            "lr": (
                self.lr[task_name][step] if self.config.separate_lr else self.lr[step]
            ),
        }

    def get_batch_indices(self, step):
        batch_size = self.num_elements * self.train_config.batch_size_per_label
        # data_size = self.num_labels * self.config.data_per_label
        data_size = self.num_elements * self.config.data_per_label
        if self.config.fix_order:
            cycle = step % int(data_size / batch_size)
            return torch.arange(batch_size * cycle, batch_size * (cycle + 1))
        else:
            return torch.randperm(data_size)[:batch_size]

    def data_dict(self, detach: bool = False):
        data_dict = {}
        for name, feature in self.data.items():
            if isinstance(feature, dict):
                for feature_name, subfeature in feature.items():
                    data_dict[self.splitter.join((name, feature_name))] = (
                        subfeature.data
                    )
            else:
                data_dict[name] = feature.data

        if detach:
            data_dict = {name: data.detach().cpu() for name, data in data_dict.items()}
        return data_dict

    def save_pretrained(self, save_path: Union[str, os.PathLike]):
        os.makedirs(save_path, exist_ok=True)

        # save config as json file
        config = {
            "config": asdict(self.config),
            "train_config": asdict(self.train_config),
            "task_num_labels": self.task_num_labels,
            "num_elements": self.num_elements,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }
        json.dump(config, open(os.path.join(save_path, "config.json"), "w"), indent=4)

        # save distilled data
        torch.save(self.data_dict(detach=True), os.path.join(save_path, "data_dict"))

        logger.info(f"Save distilled data in `{save_path}`")

    def load_data_dict(self, data_dict: dict[str, torch.Tensor]):
        # assert (
        #     self.data.keys() == data_dict.keys()
        # ), f"given keys: {self.data.keys()}, expected keys: {data_dict.keys()}"
        for name in self.data.keys():
            if isinstance(self.data[name], dict):
                for subname in self.data[name].keys():
                    self.data[name][subname].initialize_data(
                        data_dict[self.splitter.join((name, subname))]
                    )
            else:
                self.data[name].initialize_data(data_dict[name])

    @classmethod
    def from_pretrained(cls, save_path: Union[str, os.PathLike]):
        assert os.path.exists(save_path)

        # load config
        config = json.load(open(os.path.join(save_path, "config.json")))
        config.pop("num_elements")
        distilled_data = cls(**config)

        # load data
        pretrained_data = torch.load(os.path.join(save_path, "data_dict"))
        distilled_data.load_data_dict(pretrained_data)
        logger.info(f"Load distilled data from `{save_path}`")

        return distilled_data

    def cuda(self):
        for _, feature in self.data.items():
            if isinstance(feature, dict):
                for sub_feature in feature.values():
                    sub_feature.cuda()
            else:
                feature.cuda()


def get_mean_distilled_data(
    distilled_datas: List[DistilledMultilabelData],
) -> DistilledMultilabelData:
    for i in range(len(distilled_datas) - 1):
        assert (
            distilled_datas[i].config.separate_lr
            == distilled_datas[i].config.separate_lr
        )
        assert (
            distilled_datas[i].config.separate_attention_label
            == distilled_datas[i].config.separate_attention_label
        )
        assert (
            distilled_datas[i].config.fix_order == distilled_datas[i].config.fix_order
        )
        assert (
            distilled_datas[i].config.multilabel_labels_type
            == distilled_datas[i + 1].config.multilabel_labels_type
        )
        assert (
            distilled_datas[i].task_num_labels == distilled_datas[i + 1].task_num_labels
        )
        assert distilled_datas[i].hidden_size == distilled_datas[i + 1].hidden_size
        assert distilled_datas[i].num_layers == distilled_datas[i + 1].num_layers
        assert distilled_datas[i].num_heads == distilled_datas[i + 1].num_heads
        assert distilled_datas[i].splitter == distilled_datas[i + 1].splitter
        assert (
            distilled_datas[i].attention_loss_lambda
            == distilled_datas[i + 1].attention_loss_lambda
        )

    datas_len = len(distilled_datas)

    result = copy(distilled_datas[0])
    for feature_name, feature in result.data.items():
        if isinstance(feature, dict):
            for sub_feature_name, sub_feature in feature.items():
                result.data[feature_name][sub_feature_name].data = (
                    sub_feature.data.detach() / datas_len
                )
        else:
            result.data[feature_name].data = feature.data.detach() / datas_len

    for distilled_data in distilled_datas[1:]:
        for feature_name, feature in distilled_data.data.items():
            if isinstance(feature, dict):
                for sub_feature_name, sub_feature in feature.items():
                    result.data[feature_name][sub_feature_name].data += (
                        sub_feature.data.detach() / datas_len
                    )
            else:
                result.data[feature_name].data += feature.data.detach() / datas_len

    return result
