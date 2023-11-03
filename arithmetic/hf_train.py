import numpy as np
import torch as t
import torch.nn as nn
from einops import rearrange
from transformers import Trainer, TrainingArguments

from arithmetic.model import ArithmeticNet
from arithmetic.synthetic_data import get_dataset
from general.character_level_tokenizer import CharTokenizer

tokenizer = CharTokenizer()


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        logits: t.Tensor  # [batch, seq, vocab_size]

        encoder_input_ids = inputs["encoder_input_ids"]
        target_ids = inputs["labels"]
        decoder_input_ids = target_ids[:, :-1]
        label_ids = target_ids[:, 1:]

        outputs = model(
            encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids
        )
        logits, full_cache, idk_logits, pre_idk_logits = outputs

        batch_size, seq_len, vocab_size = logits.shape

        criterion = nn.CrossEntropyLoss(reduction="none")
        flattened_logits = rearrange(logits, "b s v -> (b s) v")
        flattened_labels = rearrange(label_ids, "b s -> (b s)")

        # Get predictions
        # predictions = t.argmax(flattened_logits, dim=-1)
        # print("predictions", predictions)
        # print("labels", flattened_labels)

        loss = t.sum(criterion(flattened_logits, flattened_labels)) / batch_size

        # print("loss", loss)

        return (loss, logits) if return_outputs else loss

    def compute_generate_loss(self, model, inputs, return_outputs=False):
        # Edit the generate function. What we need is for the model to autoregressively generate the whole output (which might be longer than the target and contain idk tokens)
        # Then we put this through the below loss function.

        raise NotImplementedError
        skipped_idk_tokens = (
            flattened_logits[:, tokenizer.idk_token_id] > 2.0
        )  # batch*seq
        skipped_idk_tokens_list = skipped_idk_tokens.tolist()
        # Get strip out all idk tokens from the output and instead give a penalty as n*pen for the number of idk tokens

        flattened_logits_list = flattened_logits.tolist()

        print(skipped_idk_tokens_list)

        # Remove the columns in the logits that correspond to the idk tokens
        penalty = 0
        for i, skip_bool in enumerate(skipped_idk_tokens_list):
            if skip_bool:
                flattened_logits_list.pop(i)
                penalty += self.idk_penalty


def main(num_train_epochs=2_000, num_examples=1000):
    dataset = get_dataset(num_examples=num_examples)
    print(dataset)

    model = ArithmeticNet(training=True)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=num_examples,
        per_device_eval_batch_size=num_examples,
        learning_rate=1e-3,
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=num_train_epochs // 100,
        evaluation_strategy="steps",
        # load_best_model_at_end=True,
        save_total_limit=1,
        # save_steps=10,
        eval_steps=num_train_epochs // 10,
        no_cuda=False,
        seed=42,
        fp16=False,
        fp16_opt_level="O1",
    )

    trainer = CustomTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        # tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    print("Trained!")

    predict_outputs = trainer.predict(dataset)

    logits: np.ndarray = predict_outputs.predictions  # type: ignore # [batch, seq, vocab_size]
    print(logits.shape)
    print(logits)
    preds = np.argmax(logits, axis=-1)
    print("preds", preds)
    print("true labels", predict_outputs.label_ids[:, 1:])  # type: ignore

    # print(trainer.state)
    # TODO: Memorises 1 data point but has trouble with 10?


if __name__ == "__main__":
    main()
