from nnsight import LanguageModel
from nnsight.contexts.Runner import Runner

from mamba.hf_model import MambaHFModel

CLEAN_PROMPT = "The city of Paris is in the country of"
CORRUPTED_PROMPT = "The city of Rome is in the country of"


def initial_clean_run(
    prompt: str, runner: Runner, correct_index: int, incorrect_index: int
) -> float:
    with runner.invoke(prompt) as invoker:
        clean_tokens = invoker.input["input_ids"][0]

        # Get hidden states of all layers in the network.
        # We index the output at 0 because it's a tuple where the first index is the hidden state.
        # No need to call .save() as we don't need the values after the run, just within the experiment run.
        clean_hs = [
            model.transformer.h[layer_idx].output[0]
            for layer_idx in range(len(model.transformer.h))
        ]

        # Get logits from the lm_head.
        clean_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (
            clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
        )

        return clean_logit_diff


def corrupted_run(
    prompt: str, runner: Runner, correct_index: int, incorrect_index: int
) -> float:
    with runner.invoke(CORRUPTED_PROMPT) as invoker:
        corrupted_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
        corrupted_logit_diff = (
            corrupted_logits[0, -1, correct_index]
            - corrupted_logits[0, -1, incorrect_index]
        )
        return corrupted_logit_diff


if __name__ == "__main__":
    hf_model = MambaHFModel()
    model = LanguageModel(
        repoid_path_model=hf_model, tokenizer=hf_model.tokenizer, custom_model=True
    )
    print(model)

    correct_index: int = model.tokenizer(" France")["input_ids"][0]  # type: ignore
    incorrect_index: int = model.tokenizer(" Italy")["input_ids"][0]  # type: ignore

    print(correct_index)

    prompt = "The city of Paris is in the country of"
    # Enter nnsight tracing context
    with model.forward() as runner:
        # Clean run
        clean_logit_diff = initial_clean_run(
            prompt, runner, correct_index, incorrect_index
        )
        clean_logit_diff.save()  # type: ignore

        # Corrupted run
        corrupted_logit_diff = corrupted_run(
            prompt, runner, correct_index, incorrect_index
        )
        corrupted_logit_diff.save()  # type: ignore

        all_ablating_results = []

        # Iterate through all the layers - define ablating method
        for layer_idx in range(len(model.transformer.h)):
            layer_ablating_results = []

            # Iterate through tokens in prompt
            for token_idx in range(len(clean_tokens)):
                # Apply the ablation to each hidden transition in turn
                with runner.invoke(prompt) as invoker:
                    # last_hidden_state = model.layers[-1].mixer.x_proj.output[0].save()
                    state_vector = (
                        model.mamba.layers[layer_idx]
                        .mamba_block.s6.ssm.h_updater.output[0]
                        .save()
                    )  # TODO: Overwrite
                    # final_y = model.mamba.layers[-1].mamba_block.s6.ssm.output[0].save()

                    # Alter the hidden state of the layer specified
                    # Get final output logit and compare diff against clean logit (normalised by the diff with the corrupted logit)
                    # Append results to list
                layer_ablating_results.append(result.save())

            all_ablating_results.append(layer_ablating_results)

    # Visualise results

    # print(final_state_vector.value.shape)
    # print(final_y.value.shape)
    # print("Done")

    # TODO: Write for loop to ablate each SSM block (across all layers and tokens)
