import glob
import json
import logging
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import List

import hydra
import mlflow
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import set_seed

from data import DataConfig, DataModule
from distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig
from multitask.evaluator_multitask import MultitaskEvaluator
from model import LearnerModel, ModelConfig
from multitask.model_multitask import MultitaskLearnerModel
from trainer import TrainConfig
from multitask.trainer_multitask import MultitaskTrainer
from utils import log_params_from_omegaconf_dict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    experiment_name: str
    method: str
    run_name: str
    save_dir_root: str
    save_method_dir: str
    save_dir: str
    data_dir_root: str
    seed: int = 42


@dataclass
class MultitaskConfig:
    base: BaseConfig
    data: List[DataConfig]
    model: ModelConfig
    distilled_data: DistilledDataConfig
    learner_train: LearnerTrainConfig
    train: TrainConfig
    evaluate: EvaluateConfig


cs = ConfigStore.instance()
cs.store(name="config", node=MultitaskConfig)


def mlflow_start_run_with_hydra(func):
    @wraps(func)
    def wrapper(config: MultitaskConfig, *args, **kwargs):
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


@hydra.main(config_path="../configs", config_name="default_multitask", version_base=None)
@mlflow_start_run_with_hydra
def main(config: MultitaskConfig):

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
    model = MultitaskLearnerModel(config.model, task_names, nums_labels)
    # model = LearnerModel(config.model, nums_labels[0])

    for data_module in data_modules.values():
        data_module.run_preprocess(tokenizer=model.tokenizer)

    # Distilled datas
    distilled_datas = {}
    for task_name, data_module in data_modules.items():
        if data_module.config.to_distill:
            if config.distilled_data.pretrained_data_path is not None:
                path = Path.joinpath(config.distilled_data.pretrained_data_path, task_name)
                distilled_data = DistilledData.from_pretrained(path)
            else:
                # multitask
                # distilled_data = DistilledData(
                #     config=config.distilled_data,
                #     train_config=config.learner_train,
                #     num_labels=data_module.num_labels,
                #     hidden_size=model.hidden_size,
                #     num_layers=model.num_hidden_layers,
                #     num_heads=model.num_attention_heads,
                # )
                # single task
                distilled_data = DistilledData(
                    config=config.distilled_data,
                    train_config=config.learner_train,
                    num_labels=data_module.num_labels,
                    hidden_size=model.bert_model_config.hidden_size,
                    num_layers=model.bert_model_config.num_hidden_layers,
                    num_heads=model.bert_model_config.num_attention_heads,
                )
            distilled_datas[task_name] = distilled_data

    # Evaluator
    evaluator = MultitaskEvaluator(config=config.evaluate, model=model, task_names=task_names)

    train_loaders = {task_name: data_module.train_loader() for task_name, data_module in data_modules.items()}
    val_loaders = {task_name: data_module.valid_loader() for task_name, data_module in data_modules.items()}
    test_loaders = {task_name: data_module.test_loader() for task_name, data_module in data_modules.items() if data_module.has_test()}

    # Train distilled data
    if not config.train.skip_train:
        trainer = MultitaskTrainer(config.train)
        trainer.fit(
            distilled_datas=distilled_datas,
            model=model,
            train_loaders=train_loaders,
            valid_loaders=val_loaders,
            evaluator=evaluator,
        )

    # Evaluate
    val_results = {}
    for task_name, distilled_data in distilled_datas.items():
        task_results = evaluator.evaluate(
            distilled_data, eval_loader=val_loaders[task_name], task_name=task_name, verbose=True
        )

        mlflow.log_metrics({f"val_{task_name}_avg.{k}": v[0] for k, v in task_results.items()})
        mlflow.log_metrics({f"val_{task_name}_std.{k}": v[1] for k, v in task_results.items()})

        task_results = {f"val_{task_name}.{k}": f"{v[0]}±{v[1]}" for k, v in task_results.items()}
        val_results[task_name] = task_results

    logger.info(f"Final Val Results: {val_results}")
    save_path = os.path.join(config.base.save_dir, "results_val.json")
    json.dump(val_results, open(save_path, "w"))
    mlflow.log_artifact(save_path)
    
    test_results = {}
    for task_name, distilled_data in distilled_datas.items():
        if task_name in test_loaders:
            task_results = evaluator.evaluate(
                distilled_data, eval_loader=test_loaders[task_name], task_name=task_name, verbose=True
            )

            mlflow.log_metrics({f"test_{task_name}_avg.{k}": v[0] for k, v in task_results.items()})
            mlflow.log_metrics({f"test_{task_name}_std.{k}": v[1] for k, v in task_results.items()})

            task_results = {f"test_{task_name}.{k}": f"{v[0]}±{v[1]}" for k, v in task_results.items()}
            test_results[task_name] = task_results

    logger.info(f"Final Test Results: {test_results}")
    save_path = os.path.join(config.base.save_dir, "results_test.json")
    json.dump(test_results, open(save_path, "w"))
    mlflow.log_artifact(save_path)


if __name__ == "__main__":
    main()
