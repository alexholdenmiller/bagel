import torch
import torch.nn as nn
from torch.profiler import profile, record_function

from gumbel_vector_quantizer import GumbelVectorQuantizer

BSZ = 128
EDIM = 760
SEQLEN = 100

input = torch.rand(BSZ, SEQLEN, EDIM).cuda()


# Define your model and input
model = nn.Sequential(
    nn.Linear(EDIM, EDIM),
    nn.GELU(),
    nn.Linear(EDIM, EDIM),
    nn.GELU(),
    nn.Linear(EDIM, EDIM),
).cuda()

# Profile the forward pass
with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
    with record_function("model_forward"):
        output = model(input)

# Print the profiling results
print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

# Define your model and input
model = nn.Sequential(
    GumbelVectorQuantizer(
        dim=EDIM,  # input dimension
        num_vars=160,
        temp=(2, 0.1, 0.999995),
        groups=2,
        combine_groups=False,
        vq_dim=EDIM,  # output dimension
        time_first=True,
        weight_proj_depth=1,
        weight_proj_factor=1,
    )
).cuda()

# Profile the forward pass
with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
    with record_function("model_forward"):
        output = model(input)

# Print the profiling results
print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))