from nnsight import LanguageModel

from mamba.hf_model import MambaHFModel

if __name__ == "__main__":
    hf_model = MambaHFModel()
    model = LanguageModel(
        repoid_path_model=hf_model, tokenizer=hf_model.tokenizer, custom_model=True
    )
    print(model)

    prompt = "I am become Death, the destroyer of worlds."
    with model.invoke(prompt) as invoker:
        # last_hidden_state = model.layers[-1].mixer.x_proj.output[0].save()
        print("Hi")
        last_hidden_state = (
            model.mamba.layers[-1].mamba_block.s6.ssm.h_updater.output[0].save()
        )

    print(last_hidden_state.value.shape)
    print("Done")
