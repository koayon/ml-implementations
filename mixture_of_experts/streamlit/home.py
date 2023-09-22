import streamlit as st
from transformers import AutoTokenizer

from mixture_of_experts.cache import ExpertChoiceFullCache
from mixture_of_experts.interp import tokens_processed_by_expert
from moet_experiment.model import MoET

st.set_page_config(
    layout="wide",
    page_title="Exploring Mixture of Experts",
    page_icon="🤖",
    menu_items={"About": "# Menu Item 1"},
)

MODEL_DICT = {"switch_transformer-small": None, "tiny_moe": None, "moet": MoET()}
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")


def set_model(model_name):
    st.session_state["model"] = MODEL_DICT[model_name]


st.session_state["model"] = MoET()
st.session_state["set_model"] = set_model

st.title("MoE Interp Playground")

with st.sidebar:
    st.write("🧑 vs 🤖")
    reset_button, reveal_button = [st.button("Reset game"), st.button("Reveal text")]
    with st.form("model_form"):
        # model_name = st.selectbox(
        #     label="Model",
        #     options=["switch_transformer-small", "tiny_moe", "moet"],
        #     on_change=st.session_state["set_model"],
        # )
        # model = model_dict[model_name]
        # submit_model = st.form_submit_button("Use this model")
        pass
    # st.image("assets/mechanical_hands.png")

    st.write("Powered by PyTorch")

input_str = st.text_input(label="Enter some text here")

submit_button = st.button("Submit")

if submit_button:
    # Forward model

    input_tokens = tokenizer(input_str, return_tensors="pt", padding=True)  # 1, seq_len

    model = st.session_state["model"]
    cache: ExpertChoiceFullCache
    _, cache = model(input_tokens)

    # Get tokens processed by expert
    token_indexes, tokens = tokens_processed_by_expert(
        cache=cache, layer_index="", expert_num=0
    )
    token_indexes = set(token_indexes)

    # Display the output
    coloured_text = ""
    assert tokens is not None

    for i, _ in enumerate(input_tokens):
        if i in token_indexes:
            coloured_text += f"<span style='color: red;'>{tokens[i]}</span>"
        else:
            coloured_text += tokens[i]

    st.subheader("Output")
    st.write(coloured_text, unsafe_allow_html=True)
