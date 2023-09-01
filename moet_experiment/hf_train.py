from doctest import debug
from transformers import Trainer, TrainingArguments, GenerationConfig, PreTrainedTokenizerBase, AutoTokenizer, PretrainedConfig, PreTrainedModel
from moet_experiment.moet_config import MoETConfig
from moet_experiment.model import MoET
import torch as t
import os
import pandas as pd
from torch import nn
from mixture_of_experts.tiny_stories import TinyStoriesDataset
import wandb
from typing import Optional, List
from datasets import Dataset, load_dataset
from torch.nn import functional as F
from einops import rearrange

EXPERIMENT_NAME = "test"
EXPERIMENT_GROUP = "test"
TRAIN = True
EVALUATE = True


def get_trainer(*, model: nn.Module, train_dataset: Dataset, eval_dataset: Dataset, config: MoETConfig, debug: bool=False, tokenizer: PreTrainedTokenizerBase):
    training_args = TrainingArguments(
        output_dir="checkpoints",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        # eval_accumulation_steps=32,
        fp16=True if t.cuda.is_available() else False,
        learning_rate=config.learning_rate,
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
        weight_decay = config.weight_decay,
        report_to="wandb",
        # deepspeed = "deepspeed_config.json"
        # gradient_checkpointing = True
    )

    def data_collator()

    trainer = Trainer(
        model = model,
        args = training_args,
        tokenizer = tokenizer,
        data_collator=data_collator,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        # compute_metrics = None,
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

class MoETHFConfig(PretrainedConfig):

    def __init__(
        self,
        block_type="MoE",
        layers: int = 8,
        **kwargs,
    ):

        self.block_type = block_type
        self.layers = layers
        super().__init__(**kwargs)

class MOETHFModel(PreTrainedModel):
    def __init__(self, hf_config: MoETHFConfig):
        super().__init__(hf_config)
        self.hf_config = hf_config

        self.model = MoET()

    def forward(self, input_ids: t.Tensor, attention_mask: t.Tensor, return_loss: bool = True, **kwargs):
        logits, _moe_cache = self.model(input_ids, attention_mask)
        if return_loss:
            labels = input_ids[:, 1:]
            pred_logits = logits[:, :-1, :]

            flattened_logits = rearrange(pred_logits, "b s v -> (b s) v")
            flattened_labels = rearrange(labels, "b s -> (b s)")

            loss = F.cross_entropy(flattened_logits, flattened_labels)

            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}



# TODO: There's a problem with the dataset. It's not working with the trainer.
# We might want to switch to Dataset over IterableDataset

def main():
    config = MoETConfig()

    # model = MoET(config = config)

    hf_config = MoETHFConfig()
    model = MOETHFModel(hf_config=hf_config) # hf wrapped model


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

    train_dataset = processed_dataset["train"]
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

    prompt = "Once upon a time, there was a "
    output = trainer.predict(prompt)
    print("Prompt: ", prompt)
    print("Output: ", output)

if __name__ == "__main__":
    main()
