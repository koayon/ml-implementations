import os
import sys

import streamlit as st
import torch as t
from transformers import AutoTokenizer

# Add the root directory to the path
st_dir = os.path.dirname(os.path.abspath(__file__))
moe_dir = os.path.dirname(st_dir)
ROOT = os.path.dirname(moe_dir)
sys.path.append(ROOT)

from helpers import set_logging_level
from mixture_of_experts.cache import ExpertChoiceFullCache
from mixture_of_experts.interp import tokens_processed_by_expert
from moet_experiment.model import MoET

# Set up
set_logging_level("INFO")

st.set_page_config(
    layout="wide",
    page_title="Exploring Mixture of Experts",
    page_icon="🤖",
    menu_items={"About": "# Menu Item 1"},
)

MODEL_DICT = {
    "switch_transformer-small": None,
    "tiny_moe": None,
    "moet": MoET(use_expert_choice=True),
}
tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-8M")
st.session_state["model"] = MODEL_DICT["moet"]


def set_model(model_name) -> None:
    st.session_state["model"] = MODEL_DICT[model_name]


# Main
st.title("MoE Interp Playground")

with st.sidebar:
    st.write("MoE Interp Playground")

    model_name = st.selectbox(
        label="Model",
        options=["switch_transformer-small", "tiny_moe", "moet"],
        on_change=set_model,
    )
    # st.image("assets/mechanical_hands.png")

    st.write("Powered by PyTorch")

input_str = st.text_input(label="Enter some text here")

submit_button = st.button("Submit")

if submit_button:
    # Forward model

    input_tokens = t.tensor(
        tokenizer(input_str, return_tensors="pt")["input_ids"]
    )  # 1, seq_len

    # st.write(input_tokens)

    model = st.session_state["model"]
    cache: ExpertChoiceFullCache
    _, cache = model(input_tokens)

    # st.write(cache.routing_logits_tensor.shape)

    LAYER_INDEX = "moe_block_early2"

    # Get tokens processed by expert
    token_indexes, tokens = tokens_processed_by_expert(
        cache=cache, layer_index=LAYER_INDEX, expert_num=0
    )
    token_indexes = set(token_indexes)

    # Display the output
    coloured_text = ""

    for i, _ in enumerate(input_tokens.squeeze(0).tolist()):
        if i in token_indexes:
            coloured_text += f"<span style='color: red;'>{tokenizer.decode(input_tokens.squeeze(0)[i])}</span>"
        else:
            coloured_text += tokenizer.decode(input_tokens.squeeze(0)[i])

    st.subheader("Output")
    st.write(coloured_text, unsafe_allow_html=True)
