from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer
)
from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BERT_INPUTS_DOCSTRING,
    BertModel,
)


from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from data import TASK_ATTRS
from model import AUTO_MODEL_CLASSES, MODEL_ATTRS, ModelConfig


class MultitaskLearnerModel(nn.Module):
    def __init__(self, config: ModelConfig, task_names: List[str], nums_labels: List[int]):
        super().__init__()
        self.config = config
        if self.config.disable_dropout:
            dropout_configs = {
                dropout_key: 0.0
                for dropout_key in MODEL_ATTRS[self.config.model_name]["dropout_keys"]
            }
        else:
            dropout_configs = {}

        self.num_labels = {
            task_name: num_labels 
            for task_name, num_labels in zip(task_names, nums_labels)
        }

        self.bert_model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.model_name,
            **dropout_configs,
        )
        self.bert_model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
            self.config.model_name,
            config=self.bert_model_config,
            from_tf=bool(".ckpt" in config.model_name),
            task_labels_map=self.num_labels,
        )

        if self.config.use_pretrained_model:
            self.initial_state_dict = self.bert_model.state_dict()
            self.initialized_module_names = MODEL_ATTRS[self.config.model_name][
                "initialized_module_names"
            ]

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.model_max_length = 512

    def forward(self, *args, **kwargs) -> SequenceClassifierOutput:
        assert "task_name" in kwargs
        task_name: str = kwargs["task_name"]

        labels: torch.LongTensor = kwargs.pop("labels") if "labels" in kwargs else None

        outputs: SequenceClassifierOutput = self.bert_model(*args, **kwargs)

        loss = None
        if labels is not None:
            if outputs.logits.shape == labels.shape:
                labels = labels.view(-1, self.num_labels[task_name])
            else:
                assert labels.ndim == 1

            loss = F.cross_entropy(
                outputs.logits.view(-1, self.num_labels[task_name]), labels, reduction="none"
            )
            assert loss.shape == labels.shape[:1]

        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        return self.bert_model.resize_token_embeddings(*args, **kwargs)

    def get_input_embeddings(self):
        return self.bert_model.get_input_embeddings()
                        
    def init_weights(self):
        """init_weights
        Initialize additional weights of pretrained model in the same way
        when calling AutoForSequenceClassification.from_pretrained()
        """
        self.bert_model.load_state_dict(self.initial_state_dict)
        self.bert_model.init_classifier_weights()

    @property
    def device(self):
        return self.bert_model.device


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(transformers.PretrainedConfig())
        self.num_labels = kwargs.get("task_labels_map", {})
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # add task specific output heads
        for task_name, num_labels_ in self.num_labels.items():
            setattr(self, f"classifier_{task_name}", nn.Linear(config.hidden_size, num_labels_))


    @add_start_docstrings_to_model_forward(
        BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @add_code_sample_docstrings(
        output_type=SequenceClassifierOutput,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        classifier = getattr(self, f"classifier_{task_name}")

        logits = classifier(pooled_output)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def init_classifier_weights(self):
        for task_name in self.num_labels.keys():
            classifier = getattr(self, f"classifier_{task_name}")
            classifier.weight.data.normal_(
                    mean=0.0, std=self.config.initializer_range
                )
            if classifier.bias is not None:
                classifier.bias.data.zero_()
