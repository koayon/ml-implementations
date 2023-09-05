import torch
from torch import backends

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore
# elif torch.backends.mps.is_available(): # type: ignore
#     device = torch.device("mps")
else:
    device = "cpu"
