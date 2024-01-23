from nnsight import LanguageModel

from mamba.hf_model import MambaHFModel

CLEAN_PROMPT = "The city of Paris is in the country of"
CORRUPTED_PROMPT = "The city of Rome is in the country of"

if __name__ == "__main__":
    hf_model = MambaHFModel()
    model = LanguageModel(
        repoid_path_model=hf_model, tokenizer=hf_model.tokenizer, custom_model=True
    )
    print(model)

    prompt = "The city of Paris is in the country of"
    with model.invoke(prompt) as invoker:
        # last_hidden_state = model.layers[-1].mixer.x_proj.output[0].save()
        final_state_vector = (
            model.mamba.layers[-1].mamba_block.s6.ssm.h_updater.output[0].save()
        )
        final_y = model.mamba.layers[-1].mamba_block.s6.ssm.output[0].save()

    print(final_state_vector.value.shape)
    print(final_y.value.shape)
    print("Done")

    # TODO: Write for loop to ablate each SSM block (across all layers and tokens)
