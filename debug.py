import torch
import torch.nn as nn

from time import time
from tqdm import tqdm
from torch.profiler import profile, record_function

from gb_new import GumbelVectorQuantizer as GBNew
from gb_orig import GumbelVectorQuantizer as GBOrig

BSZ = 128
EDIM = 512
SEQLEN = 100

N = 500

NVARS = 320
TEMP = (2, 0.1, 0.999995)
GROUPS = 2
COMB_GROUPS = False
VQ_DIM = 512
TF = True
WPD = 1
WPF = 1

CUDA = True

input1 = torch.rand(BSZ, SEQLEN, EDIM)
if CUDA:
    input1 = input1.cuda()

# Define your model and input
model1 = GBOrig(
    dim=EDIM,  # input dimension
    num_vars=NVARS,
    temp=TEMP,
    groups=GROUPS,
    combine_groups=COMB_GROUPS,
    vq_dim=EDIM,  # output dimension
    time_first=TF,
    weight_proj_depth=WPD,
    weight_proj_factor=WPF,
)
if CUDA:
    model1 = model1.cuda()

# Profile the forward pass
# with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
#     with record_function("model_forward"):
#         output = model1(input1)

# # Print the profiling results
# print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

model1(input1)
torch.cuda.synchronize()
t1 = time()
for _ in tqdm(range(N)):
    model1(input1)
    torch.cuda.synchronize()
t2 = time()

input2 = torch.rand(BSZ, SEQLEN, EDIM)
if CUDA:
    input2 = input2.cuda()

# Define your model and input
model2 = GBNew(
    dim=EDIM,  # input dimension
    num_vars=NVARS,
    temp=TEMP,
    groups=GROUPS,
    combine_groups=COMB_GROUPS,
    vq_dim=EDIM,  # output dimension
    time_first=TF,
    weight_proj_depth=WPD,
    weight_proj_factor=WPF,
)
if CUDA:
    model2 = model2.cuda()

# Profile the forward pass
# with profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
#     with record_function("model_forward"):
#         output = model2(input2)

# # Print the profiling results
# print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

model2(input2)
torch.cuda.synchronize()
t3 = time()
for _ in tqdm(range(N)):
    model2(input2)
    torch.cuda.synchronize()
t4 = time()

print(f"model1 aveerage {(t2 - t1)/N:.4f}s")
print(f"model2 aveerage {(t4 - t3)/N:.4f}s")