import math
import os
import sys
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import evaluate
import numpy as np
import pandas as pd
import torch as t
from datasets import Dataset, DatasetDict, load_dataset
from einops import rearrange
from evaluate import EvaluationModule
from torch import nn
from torch.nn import functional as F
from torch.optim import adamw
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextDataset,
    Trainer,
    TrainingArguments,
)

import wandb
from general import device
from moet_experiment.hf_model import MoET_hf
from moet_experiment.moet_config import MoETConfig

EXPERIMENT_NAME = "test"
EXPERIMENT_GROUP = "test"
TRAIN = True
EVALUATE = True
DEBUG = False


def get_training_args(
    *, config: MoETConfig, debug: bool = False, deepspeed_config=None
):
    return TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        deepspeed=deepspeed_config,  #  type: ignore
        # max_grad_norm = 1.0,
        # Optimizations
        # gradient_checkpointing = True
        # deepspeed = "deepspeed_config.json"
        # PyTorch 2.0 settings
        # bf16=True if t.cuda.is_available() else False, # bfloat16 training
        torch_compile=True if t.cuda.is_available() else False,  # optimizations
        optim="adamw_torch_fused"
        if t.cuda.is_available()
        else "adamw_hf",  # improved optimizer
        # eval_accumulation_steps=32,
        # Save, log, eval
        # num_train_epochs=config.num_epochs,
        max_steps=2 if debug else config.max_steps,
        warmup_steps=config.warmup_steps,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=5,
        load_best_model_at_end=True,
        report_to=["wandb"],
    )


def get_trainer(
    *,
    model: nn.Module,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    config: MoETConfig,
    debug: bool = False,
    tokenizer: PreTrainedTokenizerBase,
    data_collator: DataCollatorForLanguageModeling,
):
    deepspeed_config = (
        {
            # "fp16": {
            #     "enabled": "auto",
            #     "loss_scale": 0,
            #     "loss_scale_window": 1000,
            #     "initial_scale_power": 16,
            #     "hysteresis": 2,
            #     "min_loss_scale": 1
            # },
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": "auto",
                    "betas": "auto",
                    "eps": "auto",
                    "weight_decay": "auto",
                },
            },
            "scheduler": {
                "type": "WarmupLR",
                "params": {
                    "warmup_min_lr": "auto",
                    "warmup_max_lr": "auto",
                    "warmup_num_steps": "auto",
                },
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "gradient_accumulation_steps": "auto",
            "gradient_clipping": "auto",
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
        }
        if t.cuda.is_available()
        else None
    )

    training_args = get_training_args(
        config=config,
        debug=debug,
        # deepspeed_config=deepspeed_config
    )

    def compute_metrics(eval_pred):
        """Calculates metrics for evaluation.
        Here we take in the LOGITS and labels and compute the perplexity and other metrics.
        We are not taking in word predictions per se.

        Parameters
        ----------
        eval_pred : EvalPrediction
            Tuple of logits and labels.

        Returns
        -------
        dict
            Computed metrics.
        """
        outputs, _labels = eval_pred

        loss, _logits, _ = outputs

        # Perplexity
        per_word_loss = sum(loss) / len(loss)
        perplexity = math.exp(per_word_loss)

        # print("perplexity", perplexity)

        # # Exact match
        # preds = np.argmax(logits, axis = -1)

        # print("preds", preds)

        # preds_str_list = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # labels_str_list = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # @lru_cache
        # def get_exact_match_metric() -> EvaluationModule:
        #     return evaluate.load("exact_match")

        # exact_match_metric = get_exact_match_metric()

        # try:
        #     exact_match_dict = exact_match_metric.compute(predictions = preds_str_list, references=labels_str_list)
        #     assert exact_match_dict is not None, "Unable to compute the exact_match"
        #     exact_match = exact_match_dict["exact_match"]
        # except:
        #     exact_match = None
        # return {"perplexity": perplexity, "exact_match": exact_match}
        return {"perplexity": perplexity}

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,  #  type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics = None,
    )
    return trainer


def hyperparameter_sweep(
    num_trials: int,
    config: MoETConfig,
    model_name_or_path: str,
    small_train_dataset: TextDataset,
    small_eval_dataset: TextDataset,
    data_collator: DataCollatorForLanguageModeling,
    training_args: TrainingArguments,
):
    def wandb_hp_space(trial):
        return {
            "method": "random",
            "metric": {"name": "loss", "goal": "minimize"},  # "objective"
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
                "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
            },
        }

    def model_init() -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
        )

    trainer = Trainer(
        # model=None,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        # compute_metrics=compute_metrics,
        model_init=model_init,
        data_collator=data_collator,
    )

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=num_trials,
        # compute_objective=compute_objective,
    )

    return best_trial


def get_df(path: str):
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(
            {
                "model_name": [],
                "experiment_name": [],
                "group": [],
                "train_loss": [],
                "eval_loss": [],
                "other_metrics": [],
            }
        )
        df.to_csv(path, index=False)

    return df


def train_model(trainer: Trainer, config: MoETConfig, runs_df: pd.DataFrame, wandb):
    with wandb.init(
        project=config.model_name,
        name=f"{config.model_name}_{EXPERIMENT_NAME}",
        group=EXPERIMENT_GROUP,
    ) as run:
        try:
            trainer.train()
            wandb.alert(title="Training complete", text="Training complete")
        except Exception as e:
            wandb.alert(title="Training failed", text="Training failed")
            raise e

        checkpoint_path = f"checkpoints/{config.model_name}_{EXPERIMENT_NAME}_{run.id}"

        trainer.save_model(checkpoint_path)
        wandb.save(checkpoint_path)

        print("Trainer state: ", trainer.state)

        row_dict = {
            "model_name": config.model_name,
            "experiment_name": EXPERIMENT_NAME,
            "group": EXPERIMENT_GROUP,
            "train_loss": None,
            "eval_loss": trainer.state.best_metric,
            "other_metrics": None,
        }

        df_row = pd.DataFrame(row_dict, index=[0])

        runs_df = pd.concat([runs_df, df_row], ignore_index=True)
        runs_df.to_csv("runs.csv")

        # wandb.save("runs.csv")

        print("Saved to runs.csv")


def evaluate_model(trainer: Trainer, config: MoETConfig, evals_df: pd.DataFrame, wandb):
    with wandb.init(
        project=config.model_name,
        name=f"{config.model_name}_{EXPERIMENT_NAME}",
        group=f"eval_{EXPERIMENT_GROUP}",
    ) as run:
        metrics = trainer.evaluate()
        wandb.log(metrics)

        print(metrics)

        row_dict = {
            "model_name": config.model_name,
            "experiment_name": EXPERIMENT_NAME,
            "group": EXPERIMENT_GROUP,
            "train_loss": None,
            "eval_loss": trainer.state.best_metric,
            "other_metrics": None,
        }

        df_row = pd.DataFrame(row_dict, index=[0])

        evals_df = pd.concat([evals_df, df_row], ignore_index=True)
        evals_df.to_csv("evals.csv")

        # wandb.save("evals.csv")

        print("Saved to evals.csv")


def main():
    config = MoETConfig()

    model = MoET_hf()  # hf wrapped model
    model.to(device)

    print(device)

    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
    tokenizer.pad_token = tokenizer.eos_token
    # model.load_model(CHECKPOINT_PATH)

    # train_dataset = TinyStoriesDataset(split="train", max_seq_len=config.block_size)
    # eval_dataset = TinyStoriesDataset(split="test", max_seq_len=config.block_size)

    dataset = load_dataset("roneneldan/TinyStories")
    assert isinstance(dataset, DatasetDict)
    print(f"Example from dataset: \n {dataset['train'][0]}")

    processed_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            padding="max_length",
            max_length=config.block_size,
            truncation=True,
        ),
        batched=True,
    )
    processed_dataset.set_format(type="torch", columns=["input_ids"])
    # processed_dataset = processed_dataset.remove_columns("attention_mask")

    print(processed_dataset.column_names)

    train_dataset: Dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["validation"]
    # Reduce size of eval dataset for testing
    small_eval_dataset = eval_dataset.select(list(range(100)))
    # print(eval_dataset[0])

    runs_df = get_df("runs.csv")
    evals_df = get_df("evals.csv")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = get_trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=small_eval_dataset,
        config=config,
        tokenizer=tokenizer,
        debug=DEBUG,
        data_collator=data_collator,
    )

    wandb.login()

    if TRAIN:
        train_model(trainer=trainer, config=config, runs_df=runs_df, wandb=wandb)

    if EVALUATE:
        evaluate_model(trainer=trainer, config=config, evals_df=evals_df, wandb=wandb)

    # prompt = eval_dataset.select([10])
    # output = trainer.predict(prompt) # type: ignore
    # print("Prompt: ", prompt)
    # print("Output: ", output)


if __name__ == "__main__":
    main()
