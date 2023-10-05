import torch as t
import torch.nn as nn
from einops import rearrange
from transformers import Trainer, TrainingArguments

from general.character_level_tokenizer import CharTokenizer

tokenizer = CharTokenizer()


class CustomTrainer(Trainer):
    def __init__(self, idk_penalty: float = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idk_penalty = idk_penalty

    def compute_loss(self, model, inputs, return_outputs=False):
        logits: t.Tensor

        outputs = model(**inputs)
        logits, full_cache, idk_logits, pre_idk_logits = outputs

        # Get strip out all idk tokens from the output and instead give a penalty as n*pen for the number of idk tokens

        # logits [batch, seq, vocab_size]
        criterion = nn.CrossEntropyLoss(reduction="none")
        flattened_logits = rearrange(logits, "b s v -> (b s) v")
        flattened_labels = rearrange(inputs["labels"], "b s -> (b s)")

        skipped_idk_tokens = (
            flattened_logits[:, tokenizer.idk_token_id] > 2.0
        )  # batch*seq
        skipped_idk_tokens_list = skipped_idk_tokens.tolist()

        flattened_logits_list = flattened_logits.tolist()

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
