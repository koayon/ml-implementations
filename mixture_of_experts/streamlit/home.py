import os
import sys
from typing import List, Tuple

import streamlit as st
import torch as t

# Add the root directory to the path
st_dir = os.path.dirname(os.path.abspath(__file__))
moe_dir = os.path.dirname(st_dir)
ROOT = os.path.dirname(moe_dir)
sys.path.append(ROOT)

from helpers import set_logging_level
from mixture_of_experts.streamlit.funcs import generate_output_visual
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
    LAYER_INDEX = "moe_block_early2"
    coloured_text = generate_output_visual(
        expert1=(LAYER_INDEX, 0),
        expert2=(LAYER_INDEX, 1),
        model=st.session_state["model"],
        input_str=input_str,
    )

    st.subheader("Output")
    st.write(coloured_text, unsafe_allow_html=True)
