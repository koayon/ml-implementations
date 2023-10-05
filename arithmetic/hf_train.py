from re import T

import torch as t
import torch.nn as nn
from einops import rearrange
from transformers import Trainer, TrainingArguments

from arithmetic.model import ArithmeticNet
from arithmetic.synthetic_data import get_dataset
from general.character_level_tokenizer import CharTokenizer

tokenizer = CharTokenizer()


class CustomTrainer(Trainer):
    def __init__(self, idk_penalty: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idk_penalty = idk_penalty

    def compute_loss(self, model, inputs, return_outputs=False):
        logits: t.Tensor

        # Separate the tuples into two lists: inputs and outputs
        input_ids = [item[0] for item in inputs]
        target_ids = [item[1] for item in inputs]

        # Stack the lists to create input and output tensors
        input_tensor = t.stack(input_ids)  # batch, seq
        output_tensor = t.stack(target_ids)  # batch, seq

        outputs = model(encoder_input_ids=input_tensor, decoder_input_ids=output_tensor)
        logits, full_cache, idk_logits, pre_idk_logits = outputs

        # TODO: Edit the generate function. What we need is for the model to autoregressively generate the whole output (which might be longer than the target and contain idk tokens)
        # Then we put this through the below loss function.

        print(logits.shape)
        print(target_ids)

        # Get strip out all idk tokens from the output and instead give a penalty as n*pen for the number of idk tokens

        # logits [batch, seq, vocab_size]
        criterion = nn.CrossEntropyLoss(reduction="none")
        flattened_logits = rearrange(logits, "b s v -> (b s) v")
        flattened_labels = rearrange(target_ids, "b s -> (b s)")

        skipped_idk_tokens = (
            flattened_logits[:, tokenizer.idk_token_id] > 2.0
        )  # batch*seq
        skipped_idk_tokens_list = skipped_idk_tokens.tolist()

        flattened_logits_list = flattened_logits.tolist()

        print(skipped_idk_tokens_list)
        assert False

        # Remove the columns in the logits that correspond to the idk tokens
        penalty = 0
        for i, skip_bool in enumerate(skipped_idk_tokens_list):
            if skip_bool:
                flattened_logits_list.pop(i)
                penalty += self.idk_penalty

        flattened_logits = t.tensor(flattened_logits_list)

        loss = criterion(flattened_logits, flattened_labels)
        loss = loss.sum() + penalty

        return (loss, outputs) if return_outputs else loss


def main():
    dataset = get_dataset()
    model = ArithmeticNet(training=True)

    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        save_total_limit=1,
        save_steps=10,
        eval_steps=10,
        no_cuda=False,
        seed=42,
        fp16=False,
        fp16_opt_level="O1",
    )

    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        idk_penalty=0.1,
    )

    trainer.train()

    print(trainer.state)


if __name__ == "__main__":
    main()
