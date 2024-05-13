import logging
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import evaluate
import numpy as np
import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from data import TASK_ATTRS
from distilled_data import DistilledData
from model import LearnerModel
from utils import average, batch_on_device
from evaluator import EvaluateConfig, Metric
from multilabel.distilled_data_multilabel import DistilledMultilabelData
from multitask.trainer_multitask import rename_results

logger = logging.getLogger(__name__)


class MultilabelEvaluator:
    def __init__(self, config: EvaluateConfig, models: Dict[str, LearnerModel]):
        self.config = config
        self.models = models
        self.metrics = {task_name: Metric(task_name) for task_name in models.keys()}

    def evaluate(
        self,
        distilled_data: DistilledMultilabelData,
        eval_loader: DataLoader,
        task_name: str,
        n_eval_model: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Tuple[float]]:
        model = self.models[task_name].cuda()
        distilled_data.cuda()
        if n_eval_model is None:
            n_eval_model = self.config.n_eval_model

        all_results = []
        for i in trange(n_eval_model, dynamic_ncols=True, leave=False, desc="Evaluate"):
            # train model on distilled data
            model.init_weights()
            self.train_model(model, distilled_data, task_name)

            # evaluate trained model
            results = self.evaluate_model(model, eval_loader, task_name)
            if verbose:
                logger.info(
                    "[{:>{}}/{}]: {}".format(
                        i,
                        len(str(self.config.n_eval_model)),
                        self.config.n_eval_model,
                        results,
                    )
                )

            all_results.append(results)

        average_results = average(all_results, std=True)
        avg = {k: v[0] for k, v in average_results.items()}
        if verbose:
            logger.info(f"Average results: {avg}")

        return average_results

    def train_model(
        self,
        model: LearnerModel,
        distilled_data: DistilledMultilabelData,
        task_name: str,
    ):
        model.train()
        train_config = distilled_data.train_config

        for step in trange(
            train_config.train_step,
            leave=False,
            dynamic_ncols=True,
            desc="Train model",
        ):
            batch = distilled_data.get_batch(step, task_name)

            # compute loss
            outputs = model(
                inputs_embeds=batch["inputs_embeds"],
                labels=batch["labels"],
                output_attentions=True,
            )
            loss_task = outputs.loss.mean()

            attention_labels = batch["attention_labels"]
            if attention_labels is not None:
                attn_weights = torch.stack(outputs.attentions, dim=1)
                attn_weights = attn_weights[..., : attention_labels.size(-2), :]
                assert attn_weights.shape == attention_labels.shape
                loss_attn = F.kl_div(
                    torch.log(attn_weights + 1e-12),
                    attention_labels,
                    reduction="none",
                )
                loss_attn = loss_attn.sum(-1).mean()
            else:
                loss_attn = 0.0
            loss = loss_task + distilled_data.attention_loss_lambda * loss_attn

            # update model
            model.zero_grad()
            loss.backward()
            for params in model.parameters():
                if params.grad is not None:
                    with torch.no_grad():
                        params.sub_(batch["lr"] * params.grad)

    def evaluate_model(
        self, model: LearnerModel, data_loader: DataLoader, task_name: str
    ) -> Dict[str, float]:
        model.eval()

        total_loss, num_samples = 0, 0
        for batch in tqdm(
            data_loader, dynamic_ncols=True, leave=False, desc="Evaluate learner"
        ):
            batch = batch_on_device(batch)

            # if -1 in batch["labels"]:
            if torch.equal(batch["labels"], -torch.ones_like(batch["labels"])):
                print(num_samples)
                continue

            with torch.no_grad():
                # with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                with amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                    outputs = model(**batch)

            assert outputs.loss.shape == (len(batch["labels"]),)

            self.metrics[task_name].add_batch(outputs.logits, batch["labels"])
            total_loss += outputs.loss.sum().item()
            num_samples += len(batch["labels"])

        results = self.metrics[task_name].compute()
        results["loss"] = total_loss / num_samples

        return results

    def evaluate_fast(
        self,
        distilled_data: DistilledMultilabelData,
        eval_loader: DataLoader,
        task_name: str,
        n_eval_model: Optional[int] = None,
    ) -> Dict[str, float]:
        model = self.models[task_name].cuda()
        distilled_data.cuda()

        if n_eval_model is None:
            n_eval_model = self.config.n_eval_model

        reset_model_interval = max(len(eval_loader) // n_eval_model, 1)

        total_loss, num_samples = 0, 0
        for i, batch in enumerate(
            tqdm(eval_loader, dynamic_ncols=True, leave=False, desc="Evaluate learner")
        ):
            if i % reset_model_interval == 0:
                # train model
                model.init_weights()
                self.train_model(model, distilled_data, task_name)

            # evaluate
            model.eval()
            batch = batch_on_device(batch)
            with torch.no_grad():
                with amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                    # with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = model(**batch)

            assert outputs.loss.shape == (len(batch["labels"]),)

            self.metrics[task_name].add_batch(outputs.logits, batch["labels"])
            total_loss += outputs.loss.sum().item()
            num_samples += len(batch["labels"])

        results = self.metrics[task_name].compute()
        results["loss"] = total_loss / num_samples

        return results

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16
