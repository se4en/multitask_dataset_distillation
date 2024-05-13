import argparse
import os
import logging
import yaml

from hydra import compose, initialize
import hydra
from omegaconf import OmegaConf
import mlflow
from transformers import set_seed

from data import DataConfig, DataModule
from distilled_data import DistilledData, DistilledDataConfig, LearnerTrainConfig
from evaluator import EvaluateConfig
from multitask.evaluator_multitask import MultitaskEvaluator
from model import LearnerModel, ModelConfig
from multilabel.distilled_data_multilabel import (
    DistilledMultilabelData,
    get_mean_distilled_data,
)
from multilabel.evaluator_multilabel import MultilabelEvaluator
from multilabel.trainer_multilabel import MultilabelTrainer
from trainer import TrainConfig
from utils import log_params_from_omegaconf_dict


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id")
    parser.add_argument("--artifacts_path", default="test_eval")
    args = parser.parse_args()

    mlflow.artifacts.download_artifacts(
        # artifact_uri=f"runs:/{args.run_id}/config.yaml",
        run_id=args.run_id,
        dst_path=args.artifacts_path,
    )

    with open(args.artifacts_path + "/overrides.yaml", "r") as stream:
        overrides = yaml.load(stream, Loader=yaml.Loader)

    with initialize(
        version_base=None, config_path="../" + args.artifacts_path, job_name="test_app"
    ):
        config = compose(config_name="config.yaml", overrides=overrides)

    # print(cfg)
    set_seed(config.base.seed)

    # DataModules
    data_modules = {}
    for dataset in config.data:
        logger.info(f"Loading datasets: (`{dataset.task_name}`)")
        data_module = DataModule(dataset)
        data_modules[dataset.task_name] = data_module

    learner_models = {}
    for task_name, data_module in data_modules.items():
        config.model.task_name = task_name
        learner_models[task_name] = LearnerModel(config.model, data_module.num_labels)

    for task_name, data_module in data_modules.items():
        data_module.run_preprocess(tokenizer=learner_models[task_name].tokenizer)

    evaluator = MultilabelEvaluator(config=config.evaluate, models=learner_models)

    val_loaders = {
        task_name: data_module.valid_loader()
        for task_name, data_module in data_modules.items()
        if data_module.has_test()
    }

    task_ckpts = []
    for ckpt_task_name in data_modules.keys():
        distilled_data = DistilledMultilabelData.from_pretrained(
            args.artifacts_path + f"/best_ckpt_{ckpt_task_name}"
        )
        task_ckpts.append(distilled_data)

    mean_ckpt = get_mean_distilled_data(task_ckpts)

    test_results = {}
    for task_name in data_modules.keys():
        if task_name in val_loaders:
            task_results = evaluator.evaluate(
                mean_ckpt,
                eval_loader=val_loaders[task_name],
                task_name=task_name,
                verbose=True,
                # n_eval_model=3,
            )

            task_results = {
                f"val_{task_name}.{k}": f"{v[0]}±{v[1]}"
                for k, v in task_results.items()
            }
            test_results[task_name] = task_results

    print(f"Val Results for best mean ckpt: {test_results}")
    logger.info(f"Val Results for best mean ckpt: {test_results}")
