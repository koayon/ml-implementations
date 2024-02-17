import torch as t
from nnsight.models.Mamba import MambaInterp
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/gpt-neox-20b", padding_side="left"
)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = MambaInterp("state-spaces/mamba-130m", device="cuda", tokenizer=tokenizer)

# print(model)

prompt = "The capital of France is Paris"

# print(tokenizer.tokenize(prompt))
# assert False

NUM_LAYERS = len(model.backbone.layers)

with model.invoke(prompt) as invoker:
    all_hxs: list = [model.backbone.layers[i].mixer.ssm.hx for i in range(NUM_LAYERS)]
    all_h1s: list[t.Tensor] = [hx.next().output.save() for hx in all_hxs]  # The
    all_h2s: list[t.Tensor] = [hx.next().output.save() for hx in all_hxs]  # capital
    all_h3s: list[t.Tensor] = [hx.next().output.save() for hx in all_hxs]  # of
    for i in range(NUM_LAYERS):
        all_hxs[i].next().output = all_h3s[i]  # skip France
    all_h5s: list[t.Tensor] = [hx.next().output.save() for hx in all_hxs]  # is
    all_h6s: list[t.Tensor] = [hx.next().output.save() for hx in all_hxs]  # Paris

    output = model.lm_head.output.save()

    # hx = model.backbone.layers[0].mixer.ssm.hx
    # h1 = hx.next().output.save() # The
    # h2 = hx.next().output.save() # capital
    # h3 = hx.next().output.save() # of
    # hx.next().output = h3 # France (skipped)
    # h5 = hx.next().output.save() # is
    # h6 = hx.next().output.save()

t.softmax(output, dim=-1)

# 0 --> h1 --> h2 --> h2 --> h4_skipping_h3
#                     XXX
# Sentence
#      once  upon     a     time
