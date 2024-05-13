import glob
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import List

import hydra
from main import BaseConfig
import mlflow
from hydra.core.config_store import ConfigStore
from multilabel.distilled_data_multilabel import (
    DistilledMultilabelData,
    MultilabelDistilledLabel,
    get_mean_distilled_data,
)
from multilabel.evaluator_multilabel import MultilabelEvaluator
from multilabel.trainer_multilabel import MultilabelTrainer
from omegaconf import OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import set_seed

from data import DataConfig, DataModule
from distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig
from multitask.evaluator_multitask import MultitaskEvaluator
from model import LearnerModel, ModelConfig

# from multitask.model_multilabel import MultilabelLearnerModel
from trainer import TrainConfig
from utils import log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)


@dataclass
class MultilabelConfig:
    base: BaseConfig
    data: List[DataConfig]
    model: ModelConfig
    distilled_data: DistilledDataConfig
    learner_train: LearnerTrainConfig
    train: TrainConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=MultilabelConfig)


def mlflow_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: MultilabelConfig, *args, **kwargs):
        mlflow.set_experiment(experiment_name=config.base.experiment_name)
        with mlflow.start_run(run_name=config.base.run_name):
            output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
            # add hydra config
            hydra_config_files = glob.glob(os.path.join(output_dir, ".hydra/*"))
            for file in hydra_config_files:
                mlflow.log_artifact(file)
            with logging_redirect_tqdm():
                out = func(config, *args, **kwargs)
            # add main.log
            if os.path.exists(os.path.join(output_dir, "main.log")):
                mlflow.log_artifact(os.path.join(output_dir, "main.log"))
        return out

    return wrapper


@hydra.main(
    config_path="../configs", config_name="default_multilabel", version_base=None
)
@mlflow_start_run_with_hydra
def main(config: MultilabelConfig):

    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # log config (mlflow)
    log_params_from_omegaconf_dict(config)

    # Set seed
    set_seed(config.base.seed)

    # DataModules
    data_modules = {}
    for dataset in config.data:
        logger.info(f"Loading datasets: (`{dataset.task_name}`)")
        data_module = DataModule(dataset)
        data_modules[dataset.task_name] = data_module

    task_names, nums_labels = [], []
    for task_name, data_module in data_modules.items():
        task_names.append(task_name)
        nums_labels.append(data_module.num_labels)

    # Learner
    logger.info(f"Building Leaner model: (`{config.model.model_name}`)")
    # multitask
    # model = MultitaskLearnerModel(config.model, task_names, nums_labels)
    learner_models = {}
    for task_name, data_module in data_modules.items():
        config.model.task_name = task_name
        learner_models[task_name] = LearnerModel(config.model, data_module.num_labels)
    # model = LearnerModel(config.model, nums_labels[0])

    for task_name, data_module in data_modules.items():
        data_module.run_preprocess(tokenizer=learner_models[task_name].tokenizer)

    # Distilled data
    distilled_data = DistilledMultilabelData(
        config=config.distilled_data,
        train_config=config.learner_train,
        task_num_labels={
            task_name: data_module.num_labels
            for task_name, data_module in data_modules.items()
        },
        hidden_size=list(learner_models.values())[0].bert_model_config.hidden_size,
        num_layers=list(learner_models.values())[0].bert_model_config.num_hidden_layers,
        num_heads=list(learner_models.values())[
            0
        ].bert_model_config.num_attention_heads,
    )

    # Evaluator
    evaluator = MultilabelEvaluator(config=config.evaluate, models=learner_models)

    train_loaders = {
        task_name: data_module.train_loader()
        for task_name, data_module in data_modules.items()
    }
    val_loaders = {
        task_name: data_module.valid_loader()
        for task_name, data_module in data_modules.items()
    }
    test_loaders = {
        task_name: data_module.test_loader()
        for task_name, data_module in data_modules.items()
        if data_module.has_test()
    }

    # Train distilled data
    if not config.train.skip_train:
        trainer = MultilabelTrainer(config.train)
        trainer.fit(
            distilled_data=distilled_data,
            models=learner_models,
            train_loaders=train_loaders,
            valid_loaders=val_loaders,
            evaluator=evaluator,
        )

    task_ckpts = []
    for ckpt_task_name in data_modules.keys():
        best_ckpt_path = os.path.join(
            trainer.config.save_ckpt_dir, f"best_ckpt_{ckpt_task_name}"
        )
        distilled_data = DistilledMultilabelData.from_pretrained(best_ckpt_path)
        
        task_ckpts.append(distilled_data)
    mean_ckpt = get_mean_distilled_data(task_ckpts)

    # Evaluate
    val_results = {}
    for task_name in data_modules.keys():
        task_results = evaluator.evaluate(
            mean_ckpt,
            eval_loader=val_loaders[task_name],
            task_name=task_name,
            verbose=True,
        )

        mlflow.log_metrics(
            {f"val_{task_name}_avg.{k}": v[0] for k, v in task_results.items()}
        )
        mlflow.log_metrics(
            {f"val_{task_name}_std.{k}": v[1] for k, v in task_results.items()}
        )

        task_results = {k: f"{v[0]}±{v[1]}" for k, v in task_results.items()}
        val_results[task_name] = task_results

    logger.info(f"Final Val Results: {val_results}")
    save_path = os.path.join(config.base.save_dir, "results_val.json")
    json.dump(val_results, open(save_path, "w"))
    mlflow.log_artifact(save_path)


if __name__ == "__main__":
    main()
