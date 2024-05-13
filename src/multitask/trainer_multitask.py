import logging
import os
from dataclasses import dataclass
from typing import Dict

import mlflow
import numpy as np
import torch
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from data import DataModule
from distilled_data import DistilledData
from evaluator import Evaluator
from multitask.evaluator_multitask import MultitaskEvaluator, StrIgnoreDevice
from multitask.model_multitask import MultitaskLearnerModel
from utils import batch_on_device
from trainer import TrainConfig

logger = logging.getLogger(__name__)


def rename_results(results: dict, task_name: str) -> dict:
    return {
        (task_name + "_" + res_name): res_value for res_name, res_value in results.items()
    }


class MultitaskDataloader:
    def __init__(self, train_loaders: Dict[str, DataLoader]):
        self.train_loaders = train_loaders
        self.loaders_len = {
            task_name: len(train_loader) for task_name, train_loader in self.train_loaders.items()
        }
        self.task_name_list = list(self.train_loaders.keys())

    def __len__(self):
        return sum(self.loaders_len.values())

    def __iter__(self):
        task_choice_list = []
        min_loader_len = min(self.loaders_len.values())
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.loaders_len[task_name]

        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)

        dataloader_iter_dict = {
            task_name: iter(dataloader) 
            for task_name, dataloader in self.train_loaders.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield {
                "task_name": task_name, 
                "batch": next(dataloader_iter_dict[task_name])
            }


class MultitaskTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    def fit(
        self,
        distilled_datas: Dict[str, DistilledData],
        model: MultitaskLearnerModel,
        train_loaders: Dict[str, DataLoader],
        valid_loaders: Dict[str, DataLoader],
        evaluator: MultitaskEvaluator,
    ):
        model.cuda()
        for distilled_data in distilled_datas.values():
            distilled_data.cuda()

        # compute training steps count
        multitask_loader = MultitaskDataloader(train_loaders)
        max_training_steps = self.config.epoch * len(multitask_loader)
        if self.config.log_interval == -1:
            self.config.log_interval = len(multitask_loader) // 10

        # setup optimizer for datas
        optimizer, scheduler = self.configure_optimizer(
            distilled_datas, max_training_steps=max_training_steps
        )
        scaler = amp.GradScaler(enabled=self.use_amp)

        # evaluate before training
        results = {}
        for task_name, valid_loader in valid_loaders.items():
            if task_name in distilled_datas:
                task_results = evaluator.evaluate_fast(
                    distilled_data=distilled_datas[task_name], 
                    eval_loader=valid_loader, 
                    task_name=task_name, 
                    n_eval_model=self.config.n_eval_model
                )
                results.update(rename_results(task_results, task_name))
        mlflow.log_metrics(results, step=0)
        logger.info(
            "Validation [{:>{}}/{}]: {}".format(
                0, len(str(self.config.epoch)), self.config.epoch, results
            )
        )
        for task_name, distilled_data in distilled_datas.items():
            best_ckpt_path = os.path.join(self.config.save_ckpt_dir, "best_ckpt_" + task_name)
            distilled_data.save_pretrained(best_ckpt_path)
            mlflow.log_artifact(best_ckpt_path)

        best_val_losses = {}
        for task_name, valid_loader in valid_loaders.items():
            if task_name in distilled_datas:
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
                    model.train()
                    model.init_weights()

                    params = dict(model.named_parameters())
                    buffers = dict(model.named_buffers())                  

                    def compute_loss(
                        params, buffers, input_ids=None, attention_labels=None, task_name=None, **kwargs
                    ):
                        kwargs["output_attentions"] = True
                        with amp.autocast(enabled=self.use_amp, dtype=torch.float16):
                            outputs = torch.func.functional_call(
                                model, (params, buffers), args=input_ids, kwargs=dict(kwargs, task_name=StrIgnoreDevice(task_name))
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

                    for task_name, distilled_data in distilled_datas.items():
                        for inner_step in range(self.config.inner_loop):
                            batch_syn = distilled_data.get_batch(inner_step)

                            inputs_embeds = batch_syn.pop("inputs_embeds")
                            syn_lr = batch_syn.pop("lr")

                            # update model on distilled data
                            grads = torch.func.grad(compute_loss)(
                                params, buffers, inputs_embeds=inputs_embeds, task_name=task_name, **batch_syn
                            )
                            params = {
                                name: (p - syn_lr * grads[name]) for name, p in params.items()
                            }

                    task_batch_real = batch_on_device(batch_real["batch"])
                    batch_task_name = batch_real["task_name"]
                    task_loss_real = compute_loss(
                        params=params, 
                        buffers=buffers, 
                        task_name=batch_task_name,
                        **task_batch_real
                    )
                    loss_real = self.config.task_weights[batch_task_name] * task_loss_real

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
                    log_train_loss_task[batch_task_name] += loss_real.item()

                    pbar.set_postfix({"train_loss": loss_real.item()})

                    if (outer_step + 1) % self.config.log_interval == 0:
                        log_train_loss /= self.config.log_interval

                        for task_name in log_train_loss_task.keys():
                            if outer_step_per_task[task_name] > 0:
                                log_train_loss_task[task_name] /= outer_step_per_task[task_name]

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
                        for task_name, distilled_data in distilled_datas.items():
                            mlflow.log_metrics(
                                {
                                    f"lr_{task_name}.{i}": distilled_data.lr[i].item()
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
                    if task_name in distilled_datas:
                        task_results = evaluator.evaluate_fast(
                            distilled_data=distilled_datas[task_name], 
                            eval_loader=valid_loader, 
                            task_name=task_name, 
                            n_eval_model=self.config.n_eval_model
                        )
                        results.update(rename_results(task_results, task_name))
                mlflow.log_metrics(results, step=len(multitask_loader) * (i + 1))
                logger.info(
                    "VALIDATION (Epoch[{:>2}/{}]): {}".format(
                        i + 1, self.config.epoch, results
                    )
                )

                for task_name, valid_loader in valid_loaders.items():
                    if task_name in distilled_datas:
                        if results[f"{task_name}_loss"] < best_val_losses[f"{task_name}_loss"]:
                            best_ckpt_path = os.path.join(self.config.save_ckpt_dir, "best_ckpt_" + task_name)
                            best_val_losses[f"{task_name}_loss"] = results[f"{task_name}_loss"]
                            distilled_data.save_pretrained(best_ckpt_path)
                            mlflow.log_artifact(best_ckpt_path)
                            logger.info(f"Save best {task_name} checkpoint at `{best_ckpt_path}`")

            

        logger.info("Finish training!!")

        for task_name, distilled_data in distilled_datas.items():
            # save last checkpoint
            last_ckpt_path = os.path.join(self.config.save_ckpt_dir, "last-ckpt")
            distilled_data.save_pretrained(last_ckpt_path)
            mlflow.log_artifact(last_ckpt_path)
            logger.info(f"Save last {task_name} checkpoint at `{last_ckpt_path}`")

            # load best checkpoint
            best_ckpt_path = os.path.join(self.config.save_ckpt_dir, "best_ckpt_" + task_name)
            best_checkpoint = torch.load(os.path.join(best_ckpt_path, "data_dict"))
            distilled_data.load_data_dict(best_checkpoint)

    def configure_optimizer(
        self,
        distilled_datas: Dict[str, DistilledData],
        max_training_steps: int,
    ) -> tuple[Optimizer, _LRScheduler]:

        optimizer_class = {"sgd": SGD, "momentum": SGD, "adam": Adam, "adamw": AdamW}
        assert self.config.optimizer_type in optimizer_class

        grouped_params = []
        for task_name, distilled_data in distilled_datas.items():
            data_dict = distilled_data.data_dict()
            assert data_dict.keys() >= {
                "inputs_embeds",
                "labels",
                "lr",
            }, f"{data_dict.keys()}"

            grouped_params += [
                {
                    "params": data_dict["inputs_embeds"],
                    "weight_decay": self.config.weight_decay,
                    "lr": self.config.lr_inputs_embeds,
                },
                {
                    "params": data_dict["labels"], 
                    "lr": self.config.lr_labels,
                },
                {
                    "params": data_dict["lr"], 
                    "lr": self.config.lr_lr,
                },
            ]
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
