# Creates some tensors in default dtype (here assumed to be float32)
import torch
from torch.cuda.amp import autocast

a_float32 = torch.rand((8, 8), device="cpu")
b_float32 = torch.rand((8, 8), device="cpu")
c_float32 = torch.rand((8, 8), device="cpu")
d_float32 = torch.rand((8, 8), device="cpu")

with autocast():
    # torch.mm is on autocast's list of ops that should run in bfloat16.
    # Inputs are float32, but the op runs in bfloat16 and produces bfloat16 output.
    # No manual casts are required.
    e_bfloat16 = torch.mm(a_float32, b_float32)
    # Also handles mixed input types
    f_bfloat16 = torch.mm(d_float32, e_bfloat16)

# After exiting autocast, calls f_float16.float() to use with d_float32
g_float32 = torch.mm(d_float32, f_bfloat16.float())
print(g_float32)