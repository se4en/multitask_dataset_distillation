import logging
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import mlflow
from multilabel.distilled_data_multilabel import DistilledMultilabelData
from multilabel.evaluator_multilabel import MultilabelEvaluator
from multitask.evaluator_multitask import StrIgnoreDevice
from multitask.trainer_multitask import MultitaskDataloader, rename_results
import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from trainer import TrainConfig
from transformers import get_scheduler

from distilled_data import DistilledData
from model import LearnerModel
from utils import batch_on_device

logger = logging.getLogger(__name__)


class MultilabelTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    def fit(
        self,
        distilled_data: DistilledMultilabelData,
        models: Dict[str, LearnerModel],
        train_loaders: Dict[str, DataLoader],
        valid_loaders: Dict[str, DataLoader],
        evaluator: MultilabelEvaluator,
    ):
        for task_name in models.keys():
            models[task_name].cuda()
        distilled_data.cuda()

        multitask_loader = MultitaskDataloader(train_loaders)
        max_training_steps = self.config.epoch * len(multitask_loader)
        if self.config.log_interval == -1:
            self.config.log_interval = len(multitask_loader) // 10

        optimizer, scheduler = self.configure_optimizer(
            distilled_data, max_training_steps=max_training_steps
        )
        scaler = amp.GradScaler(enabled=self.use_amp)

        # evaluate before training

        results = {}
        for task_name, valid_loader in valid_loaders.items():
            if task_name in models:
                task_results = evaluator.evaluate_fast(
                    distilled_data=distilled_data,
                    eval_loader=valid_loader,
                    task_name=task_name,
                    n_eval_model=self.config.n_eval_model,
                )
                results.update(rename_results(task_results, task_name))
        mlflow.log_metrics(results, step=0)
        logger.info(
            "Validation [{:>{}}/{}]: {}".format(
                0, len(str(self.config.epoch)), self.config.epoch, results
            )
        )

        best_ckpt_path = os.path.join(
            self.config.save_ckpt_dir, "best_ckpt_" + task_name
        )
        distilled_data.save_pretrained(best_ckpt_path)
        mlflow.log_artifact(best_ckpt_path)

        best_val_losses = {}
        for task_name, valid_loader in valid_loaders.items():
            if task_name in models:
                best_val_losses[f"{task_name}_loss"] = results[f"{task_name}_loss"]

        logger.info("Start training!")
        for i in range(self.config.epoch):
            log_train_loss = 0
            log_train_loss_task = {task_name: 0 for task_name in train_loaders.keys()}
            outer_step_per_task = {task_name: 0 for task_name in train_loaders.keys()}
            with tqdm(
                multitask_loader,
                dynamic_ncols=True,
                leave=False,
                desc=f"Train synthetic data (Epoch[{i+1:>2}/{self.config.epoch}])",
            ) as pbar:
                for outer_step, batch_real in enumerate(pbar):
                    # initialize model
                    model = models[batch_real["task_name"]]
                    model.train()
                    model.init_weights()

                    params = dict(model.named_parameters())
                    buffers = dict(model.named_buffers())

                    def compute_loss(
                        params,
                        buffers,
                        input_ids=None,
                        attention_labels=None,
                        task_name=None,
                        **kwargs,
                    ):
                        kwargs["output_attentions"] = True
                        with amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                            outputs = torch.func.functional_call(
                                models[task_name],
                                (params, buffers),
                                args=input_ids,
                                kwargs=kwargs,
                            )

                        loss_task = outputs.loss.mean()

                        if attention_labels is not None:
                            attn_weights = torch.stack(outputs.attentions, dim=1)
                            attn_weights = attn_weights[
                                ..., : attention_labels.size(-2), :
                            ]
                            assert attn_weights.shape == attention_labels.shape
                            loss_attn = F.kl_div(
                                torch.log(attn_weights + 1e-12),
                                attention_labels,
                                reduction="none",
                            )
                            loss_attn = loss_attn.sum(-1).mean()
                        else:
                            loss_attn = 0.0

                        return (
                            loss_task + distilled_data.attention_loss_lambda * loss_attn
                        )
                        # return loss_task

                    # for task_name in models.keys():
                    for inner_step in range(self.config.inner_loop):
                        batch_syn = distilled_data.get_batch(
                            inner_step, task_name=batch_real["task_name"]
                        )

                        inputs_embeds = batch_syn.pop("inputs_embeds")
                        syn_lr = batch_syn.pop("lr")

                        # update model on distilled data
                        grads = torch.func.grad(compute_loss)(
                            params,
                            buffers,
                            inputs_embeds=inputs_embeds,
                            task_name=batch_real["task_name"],
                            **batch_syn,
                        )
                        params = {
                            name: (p - syn_lr * grads[name])
                            for name, p in params.items()
                        }

                    task_batch_real = batch_on_device(batch_real["batch"])
                    batch_task_name = batch_real["task_name"]
                    task_loss_real = compute_loss(
                        params=params,
                        buffers=buffers,
                        task_name=batch_task_name,
                        **task_batch_real,
                    )
                    loss_real = task_loss_real
                    # loss_real = sum(loss_real_task.values())

                    # compute gradient
                    optimizer.zero_grad()
                    scaler.scale(loss_real).backward()

                    # gradient clipping
                    if self.config.max_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            distilled_data.data_dict().values(),
                            max_norm=self.config.max_grad_norm,
                        )

                    # update distilled data
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()

                    # detach params
                    for name, param in params.items():
                        params[name] = param.detach().requires_grad_()

                    outer_step_per_task[batch_task_name] += 1
                    log_train_loss += loss_real.item()
                    # for task_name in loss_real_task.keys():
                    #     log_train_loss_task[task_name] += loss_real_task[task_name].item()
                    log_train_loss_task[batch_task_name] += loss_real.item()

                    pbar.set_postfix({"train_loss": loss_real.item()})

                    if (outer_step + 1) % self.config.log_interval == 0:
                        log_train_loss /= self.config.log_interval

                        for task_name in log_train_loss_task.keys():
                            if outer_step_per_task[task_name] > 0:
                                log_train_loss_task[task_name] /= outer_step_per_task[
                                    task_name
                                ]

                        # log loss
                        mlflow.log_metric(
                            "train_loss",
                            log_train_loss,
                            step=len(multitask_loader) * i + outer_step,
                        )
                        for task_name in log_train_loss_task.keys():
                            mlflow.log_metric(
                                f"train_loss_{task_name}",
                                log_train_loss_task[task_name],
                                step=len(multitask_loader) * i + outer_step,
                            )

                        # log lr
                        for task_name in models.keys():
                            mlflow.log_metrics(
                                {
                                    f"lr.{i}": distilled_data.lr[task_name][i].item()
                                    for i in range(self.config.inner_loop)
                                },
                                step=len(multitask_loader) * i + outer_step,
                            )
                        mlflow.log_metric(
                            "optimizer_lr",
                            scheduler.get_last_lr()[0],
                            step=len(multitask_loader) * i + outer_step,
                        )

                        logger.info(
                            "TRAIN (Epoch[{:>4.1f}]): train_loss={}".format(
                                (outer_step + 1) / len(multitask_loader) + i,
                                log_train_loss,
                            )
                        )

                        log_train_loss = 0
                        for task_name in log_train_loss_task.keys():
                            log_train_loss_task[task_name] = 0
                            outer_step_per_task[task_name] = 0

            if (i + 1) % self.config.val_interval == 0:
                results = {}
                for task_name, valid_loader in valid_loaders.items():
                    if task_name in models:
                        task_results = evaluator.evaluate_fast(
                            distilled_data=distilled_data,
                            eval_loader=valid_loader,
                            task_name=task_name,
                            n_eval_model=self.config.n_eval_model,
                        )
                        results.update(rename_results(task_results, task_name))
                mlflow.log_metrics(results, step=len(multitask_loader) * (i + 1))
                logger.info(
                    "VALIDATION (Epoch[{:>2}/{}]): {}".format(
                        i + 1, self.config.epoch, results
                    )
                )

                for task_name, valid_loader in valid_loaders.items():
                    if task_name in models:
                        if (
                            results[f"{task_name}_loss"]
                            < best_val_losses[f"{task_name}_loss"]
                        ):
                            best_ckpt_path = os.path.join(
                                self.config.save_ckpt_dir, "best_ckpt_" + task_name
                            )
                            best_val_losses[f"{task_name}_loss"] = results[
                                f"{task_name}_loss"
                            ]
                            distilled_data.save_pretrained(best_ckpt_path)
                            mlflow.log_artifact(best_ckpt_path)
                            logger.info(
                                f"Save best {task_name} checkpoint at `{best_ckpt_path}`"
                            )

        logger.info("Finish training!!")

        # save last checkpoint
        last_ckpt_path = os.path.join(self.config.save_ckpt_dir, "last-ckpt")
        distilled_data.save_pretrained(last_ckpt_path)
        mlflow.log_artifact(last_ckpt_path)
        logger.info(f"Save last checkpoint at `{last_ckpt_path}`")

        # load best checkpoint
        best_checkpoint = torch.load(os.path.join(best_ckpt_path, "data_dict"))
        distilled_data.load_data_dict(best_checkpoint)

    def configure_optimizer(
        self,
        distilled_data: DistilledMultilabelData,
        max_training_steps: int,
    ) -> Tuple[Optimizer, _LRScheduler]:

        optimizer_class = {"sgd": SGD, "momentum": SGD, "adam": Adam, "adamw": AdamW}
        assert self.config.optimizer_type in optimizer_class

        data_dict = distilled_data.data_dict()

        grouped_params = []
        for name in data_dict.keys():
            if name == "inputs_embeds":
                grouped_params.append(
                    {
                        "params": data_dict["inputs_embeds"],
                        "weight_decay": self.config.weight_decay,
                        "lr": self.config.lr_inputs_embeds,
                    }
                )
            elif name == "lr":
                grouped_params.append(
                    {
                        "params": data_dict["lr"],
                        "lr": self.config.lr_lr,
                    }
                )
            elif name.startswith("labels"):
                grouped_params.append(
                    {"params": data_dict[name], "lr": self.config.lr_labels}
                )

        if "attention_labels" in data_dict:
            grouped_params.append(
                {
                    "params": data_dict["attention_labels"],
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.lr_attention_labels,
                }
            )

        optimizer = optimizer_class[self.config.optimizer_type](
            grouped_params, lr=1.0
        )  # `lr=1.0` is not used (dummy)
        logger.info(f"Optimizer: {optimizer}")

        # learning rate scheduler
        scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=optimizer if optimizer is not None else optimizer,
            num_warmup_steps=max_training_steps * self.config.warmup_ratio,
            num_training_steps=max_training_steps,
        )

        return optimizer, scheduler

    @property
    def use_amp(self):
        return self.config.fp16 or self.config.bf16

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16
