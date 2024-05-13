import logging
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from datasets import Dataset, disable_progress_bar, load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

disable_progress_bar()


TASK_ATTRS = {
    # AGNEWS
    "ag_news": {
        "load_args": ("ag_news",),
        "sentence_keys": ("text",),
        "problem_type": "single_label_classification",
        "test_split_key": "test",
        "metric_keys": ("accuracy",),
    },
    # GLUE
    "mrpc": {
        "load_args": ("glue", "mrpc"),
        "sentence_keys": ("sentence1", "sentence2"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "mrpc"),
    },
    "qnli": {
        "load_args": ("glue", "qnli"),
        "sentence_keys": ("question", "sentence"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "qnli"),
    },
    "sst2": {
        "load_args": ("glue", "sst2"),
        "sentence_keys": ("sentence",),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "sst2"),
    },
    "qqp": {
        "load_args": ("glue", "qqp"),
        "sentence_keys": ("question1", "question2"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation",
        "metric_keys": ("glue", "qqp"),
    },
    "mnli": {
        "load_args": ("glue", "mnli"),
        "sentence_keys": ("premise", "hypothesis"),
        "problem_type": "single_label_classification",
        "test_split_key": "validation_matched",
        "metric_keys": ("glue", "mnli"),
    },
    # SciCITE
    "scicite": {
        "problem_type": "single_label_classification",
        "text_label": "string",
        "label_label": "label",
        "train_file": "train.jsonl",
        "validation_file": "dev.jsonl",
        "test_file": "test.jsonl",
        "metric_keys": ("f1",),  # ("f1_macro",),
        "metric_args": {"average": "macro"},
    },
    "scicite_cite_worthiness": {
        "problem_type": "single_label_classification",
        "text_label": "cleaned_cite_text",
        "label_label": "is_citation",
        "train_file": "scaffolds/cite-worthiness-scaffold-train.jsonl",
        "validation_file": None,
        "metric_keys": ("accuracy",),  # ("f1_macro",),
    },
    "scicite_sections": {
        "problem_type": "single_label_classification",
        "text_label": "cleaned_cite_text",
        "label_label": "section_title",
        "train_file": "scaffolds/sections-scaffold-train.jsonl",
        "validation_file": None,
        "metric_keys": ("accuracy",),  # ("f1_macro",),
    },
}


@dataclass
class DataConfig:
    task_name: str
    datasets_path: Path
    preprocessed_datasets_path: Path
    train_batch_size: int = 32
    valid_batch_size: int = 256
    test_batch_size: int = 256
    num_proc: int = 1
    force_preprocess: bool = False
    dataset_source_path: Optional[str] = None
    to_distill: bool = True


class DataModule:
    """DataModule class
    ```
    data_module = DataModule(
        config.data,
        tokenizer_generator=generator.tokenizer,
        tokenizer_learner=learner.tokenizer,
    )
    # preprocess datasets
    data_module.run_preprocess(tokenizer=tokenizer)
    # preprocess external dataset (distilled data)
    data_module.preprocess_dataset(tokenizer=tokenizer, dataset=dataset)
    ```
    """

    def __init__(self, config: DataConfig):
        self.config = config

        # load raw dataset
        self.dataset_attr = TASK_ATTRS[self.config.task_name]
        self.datasets: DatasetDict = self.get_dataset()
        logger.info(f"Datasets: {self.datasets}")

        self.num_labels = self.datasets["train"].features["labels"].num_classes

        # preprocessed_dataset
        self.preprocessed_datasets = None

        # data collator
        self.data_collator = None

    def get_dataset(self):
        """load raw datasets from source"""
        if os.path.exists(self.config.datasets_path):
            datasets = load_from_disk(self.config.datasets_path)
        elif self.config.task_name.startswith("scicite"):
            source_path = Path(self.config.dataset_source_path)
            task_config = TASK_ATTRS[self.config.task_name]

            datasets = DatasetDict()
            if task_config["validation_file"] is not None:
                datasets["train"] = self._load_scicite_dataset(
                    path_to_json=source_path.joinpath(task_config["train_file"]),
                    text_label=task_config["text_label"],
                    label_label=task_config["label_label"],
                )
                datasets["validation"] = self._load_scicite_dataset(
                    path_to_json=source_path.joinpath(task_config["validation_file"]),
                    text_label=task_config["text_label"],
                    label_label=task_config["label_label"],
                )

                if task_config["test_file"] is not None:
                    # print("!!! test file is not none")
                    datasets["test"] = self._load_scicite_dataset(
                        path_to_json=source_path.joinpath(task_config["test_file"]),
                        text_label=task_config["text_label"],
                        label_label=task_config["label_label"],
                    )
            else:
                full_dataset = self._load_scicite_dataset(
                    path_to_json=source_path.joinpath(task_config["train_file"]),
                    text_label=task_config["text_label"],
                    label_label=task_config["label_label"],
                )
                splitted_dataset = full_dataset.train_test_split(
                    test_size=0.1, shuffle=True, seed=20240201
                )
                datasets["train"] = splitted_dataset["train"]
                # datasets["train"] = splitted_dataset["test"]
                datasets["validation"] = splitted_dataset["validation"]
                datasets["test"] = splitted_dataset["test"]

            os.makedirs(os.path.dirname(self.config.datasets_path), exist_ok=True)
            datasets.save_to_disk(self.config.datasets_path)
        else:
            assert self.config.task_name in TASK_ATTRS
            datasets = load_dataset(*self.dataset_attr["load_args"])

            if "validation" not in datasets:
                datasets["validation"] = datasets.pop(
                    self.dataset_attr["test_split_key"]
                )
            assert datasets.keys() >= {"train", "validation"}

            os.makedirs(os.path.dirname(self.config.datasets_path), exist_ok=True)
            datasets.save_to_disk(self.config.datasets_path)

        if (
            TASK_ATTRS[self.config.task_name]["problem_type"]
            == "single_label_classification"
        ):
            # rename label_key
            assert "label" in datasets["train"].features
            datasets = datasets.rename_column("label", "labels")
        else:
            raise NotImplementedError

        return datasets

    def run_preprocess(self, tokenizer: PreTrainedTokenizerFast):
        """datasets preprocessing"""

        # set data_collator
        if self.data_collator is None:
            self.data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer, padding="longest", pad_to_multiple_of=8
            )

        if (
            os.path.exists(self.config.preprocessed_datasets_path)
            and not self.config.force_preprocess
        ):
            logger.info(
                "Load preprocessed datasets from `{}`".format(
                    self.config.preprocessed_datasets_path
                )
            )
            self.preprocessed_datasets = load_from_disk(
                self.config.preprocessed_datasets_path
            )
            return

        self.preprocessed_datasets = self.preprocess_dataset(
            tokenizer=tokenizer, dataset=self.datasets
        )

        logger.info(
            f"Save preprocessed datasets to `{self.config.preprocessed_datasets_path}`"
        )
        os.makedirs(
            os.path.dirname(self.config.preprocessed_datasets_path), exist_ok=True
        )
        self.preprocessed_datasets.save_to_disk(self.config.preprocessed_datasets_path)

    def preprocess_dataset(
        self,
        tokenizer: PreTrainedTokenizerFast,
        dataset: Optional[Union[Dataset, DatasetDict]],
    ) -> Union[Dataset, DatasetDict]:
        # print("Preprocess dataset")

        # sentence keys for task
        if "sentence_keys" in TASK_ATTRS[self.config.task_name]:
            sentence_keys = TASK_ATTRS[self.config.task_name]["sentence_keys"]

            def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
                sentences = [[s.strip() for s in batch[key]] for key in sentence_keys]
                return tokenizer(
                    *sentences, max_length=tokenizer.model_max_length, truncation=True
                )

        else:  # scicite datasets

            def tokenize_fn(batch: dict[str, Any]) -> dict[str, Any]:
                sentences = [[s.strip() for s in batch["text"]]]
                return tokenizer(
                    *sentences, max_length=tokenizer.model_max_length, truncation=True
                )

        # tokenize
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=self.config.num_proc,
            desc="Tokenize datasets",
        )

        remove_keys = [
            col
            for col in dataset["train"].column_names
            if col not in ["input_ids", "token_type_ids", "attention_mask", "labels"]
        ]
        dataset = dataset.remove_columns(remove_keys)

        return dataset

    def train_loader(self) -> DataLoader:
        assert "train" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["train"],
            batch_size=self.config.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
        )

    def valid_loader(self) -> DataLoader:
        assert "validation" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["validation"],
            batch_size=self.config.test_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=False,
        )

    def has_test(self) -> bool:
        return "test" in self.preprocessed_datasets

    def test_loader(self) -> DataLoader:
        assert "test" in self.preprocessed_datasets
        assert self.data_collator is not None

        return DataLoader(
            self.preprocessed_datasets["test"],
            batch_size=self.config.test_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            drop_last=False,
        )

    def _load_scicite_dataset(
        self, path_to_json: str, text_label: str = "text", label_label: str = "label"
    ) -> Dataset:
        dataset = []

        assert os.path.exists(path_to_json)

        rb = open(path_to_json, "r", encoding="utf-8")
        for line in rb.readlines():
            paper = json.loads(line.strip())
            dataset.append({"text": paper[text_label], "label": paper[label_label]})

        result = Dataset.from_list(dataset)
        return result.class_encode_column("label")
