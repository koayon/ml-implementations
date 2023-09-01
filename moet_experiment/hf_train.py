import os
from typing import List, Optional

import evaluate
import pandas as pd
import torch as t
from datasets import Dataset, DatasetDict, load_dataset
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

import wandb
from mixture_of_experts.tiny_stories import TinyStoriesDataset
from moet_experiment.hf_model import MoET_hf, MoETHFConfig
from moet_experiment.model import MoET
from moet_experiment.moet_config import MoETConfig

EXPERIMENT_NAME = "test"
EXPERIMENT_GROUP = "test"
TRAIN = True
EVALUATE = True


def get_trainer(*, model: nn.Module, train_dataset: Dataset, eval_dataset: Dataset, config: MoETConfig, debug: bool=False, tokenizer: PreTrainedTokenizerBase):
    training_args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay = config.weight_decay,

        # Optimizations
        # gradient_checkpointing = True
        # deepspeed = "deepspeed_config.json"

        # PyTorch 2.0 settings
        bf16=True if t.cuda.is_available() else False, # bfloat16 training
        torch_compile=True, # optimizations
        optim="adamw_torch_fused", # improved optimizer
        # eval_accumulation_steps=32,

        # Save, log, eval
        # num_train_epochs=config.num_epochs,
        max_steps = 100 if debug else config.max_steps,
        warmup_steps = config.warmup_steps,
        logging_strategy = "steps",
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_total_limit=5,
        load_best_model_at_end = True,
        report_to=["wandb"],
    )

    exact_match_metric = evaluate.load("exact_match")
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
        logits, labels = eval_pred

        # Perplexity
        flattened_logits = rearrange(logits, "b s v -> (b s) v")
        flattened_labels = rearrange(labels, "b s -> (b s)")
        loss = F.cross_entropy(flattened_logits, flattened_labels)
        per_word_loss = loss / flattened_labels.shape[0]

        perplexity = t.exp(-per_word_loss).item()

        # Exact match
        preds = t.argmax(logits, dim=-1)

        preds_str_list = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_str_list = tokenizer.batch_decode(labels, skip_special_tokens=True)

        try:
            exact_match = exact_match_metric.compute(predictions = preds_str_list, references=labels_str_list)["exact_match"]
        except:
            exact_match = None
        return {"perxplexity": perplexity, "exact_match": exact_match}

    trainer = Trainer(
        model = model,
        args = training_args,
        tokenizer = tokenizer,
        # data_collator=data_collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics = compute_metrics,
        # preprocess_logits_for_metrics = None,
    )
    return trainer


def get_df(path: str ):
    if os.path.exists(path):
        runs_df = pd.read_csv(path)
    else:
        runs_df = pd.DataFrame({"model_name": [], "experiment_name": [], "group": [], "train_loss": [], "eval_loss": [], "other_metrics": []})
        runs_df.to_csv(path, index=False)

    return runs_df


def train_model(trainer: Trainer, config: MoETConfig, model: nn.Module, runs_df: pd.DataFrame):
    with wandb.init(project=config.model_name, name = f"{config.model_name}_{EXPERIMENT_NAME}", group = EXPERIMENT_GROUP) as run:
        try:
            trainer.train()
            wandb.alert(title="Training complete", text="Training complete")
        except Exception as e:
            raise e
            wandb.alert(title="Training failed", text="Training failed")

        checkpoint_path = f"checkpoints/{config.model_name}_{EXPERIMENT_NAME}_{run.id}"

        trainer.save_model(checkpoint_path)
        wandb.save(checkpoint_path)

        print("Trainer state: ", trainer.state)

        row_dict = {"model_name": config.model_name, "experiment_name": EXPERIMENT_NAME, "group": EXPERIMENT_GROUP, "train_loss": None, "eval_loss": trainer.state.best_metric, "other_metrics": None,}

        df_row = pd.DataFrame(row_dict, index = [0])

        runs_df = pd.concat([runs_df, df_row], ignore_index=True)
        runs_df.to_csv("runs.csv")

        # wandb.save("runs.csv")

        print("Saved to runs.csv")


def evaluate_model(trainer: Trainer, config: MoETConfig, evals_df: pd.DataFrame):

    with wandb.init(project=config.model_name, name = f"{config.model_name}_{EXPERIMENT_NAME}", group = f"eval_{EXPERIMENT_GROUP}") as run:
        metrics = trainer.evaluate()
        wandb.log(metrics)

        print(metrics)

        row_dict = {"model_name": config.model_name, "experiment_name": EXPERIMENT_NAME, "group": EXPERIMENT_GROUP, "train_loss": None, "eval_loss": trainer.state.best_metric, "other_metrics": None,}

        df_row = pd.DataFrame(row_dict, index = [0])

        evals_df = pd.concat([evals_df, df_row], ignore_index=True)
        evals_df.to_csv("evals.csv")

        # wandb.save("evals.csv")

        print("Saved to evals.csv")


def main():
    config = MoETConfig()

    hf_config = MoETHFConfig()
    model = MoET_hf(hf_config=hf_config) # hf wrapped model


    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
    tokenizer.pad_token = tokenizer.eos_token
    # model.load_model(CHECKPOINT_PATH)

    # train_dataset = TinyStoriesDataset(split="train", max_seq_len=config.block_size)
    # eval_dataset = TinyStoriesDataset(split="test", max_seq_len=config.block_size)

    dataset = load_dataset("roneneldan/TinyStories")
    print(dataset["train"][0])

    processed_dataset = dataset.map(lambda x: tokenizer(x["text"],
                                                        padding="max_length", max_length=config.block_size,
                                                        truncation=True), batched=True)
    processed_dataset.set_format(type="torch", columns=["input_ids"])
    # processed_dataset = processed_dataset.remove_columns("attention_mask")

    print(processed_dataset.column_names)

    train_dataset: Dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["validation"]
    eval_dataset = eval_dataset.select(list(range(100)))
    # print(eval_dataset[0])

    runs_df = get_df("runs.csv")
    evals_df = get_df("evals.csv")

    trainer = get_trainer(model = model, train_dataset = train_dataset, eval_dataset=eval_dataset, config = config, tokenizer = tokenizer, debug=False, )

    if TRAIN:
        train_model(trainer = trainer, config = config, model = model, runs_df = runs_df)

    if EVALUATE:
        evaluate_model(trainer = trainer, config = config, evals_df = evals_df)

    prompt = eval_dataset.select([100])
    output = trainer.predict(prompt)
    print("Prompt: ", prompt)
    print("Output: ", output)

if __name__ == "__main__":
    main()
