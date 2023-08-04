# %% [markdown]
# ## Patching Experiments

# %% [markdown]
# ### Load the dataset

# %%
import json
with open('l1.json', 'r') as f:
    l1_dict=json.load(f)
l1=list(l1_dict.keys())
neg_adj=list(l1_dict.values())
with open('l2.json', 'r') as f:
    l2_dict=json.load(f)
l2=list(l2_dict.keys())
with open('l3.json', 'r') as f:
    l3_dict=json.load(f)
l3=list(l3_dict.keys())
pos_adj=list(l3_dict.values())
with open('l4.json', 'r') as f:
    l4_dict=json.load(f)
l4=list(l4_dict.keys())
with open('l5.json', 'r') as f:
    l5_dict=json.load(f)
l5=list(l5_dict.keys())


# %% [markdown]
# ### Setting everything up for Trasnformer lens library

# %%
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
DEBUG_MODE = False
IN_COLAB = False
print("Running as a Jupyter notebook - intended for development only!")
from IPython import get_ipython
ipython = get_ipython()
# Code to automatically update the HookedTransformer code as its edited without restarting the kernel
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 2")

pio.renderers.default = "png"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader
from jaxtyping import Float, Int

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
torch.set_grad_enabled(False)
from neel_plotly import line, imshow, scatter
import transformer_lens.patching as patching
model = HookedTransformer.from_pretrained("gpt2-small")
prompts = ['The day is bright but the night is']
not_id=model.to_single_token(" not")
bright_id=model.to_single_token(" bright")
clean_tokens = model.to_tokens(prompts)
print("Clean string 0", model.to_string(clean_tokens[0]))
logits,cache=model.run_with_cache(clean_tokens)
id=logits[...,-1,:].argmax()
model.to_string(id)

import plotly.io as pio
pio.renderers.default = "vscode"


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %% [markdown]
# # Activation Patching

# %%
import transformer_lens.patching as patching
clean_tokens=model.to_tokens(l1)
clean_tokens.shape
corrupted_tokens=model.to_tokens(l5)
answers=[(i,j) for i,j in zip(neg_adj,pos_adj)]
answer_token_indices = torch.tensor([[model.to_single_token(" "+answers[i][j]) for j in range(2)] for i in range(len(answers))], device=model.cfg.device)


def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    if len(logits.shape)==3:
        # Get final logits only
        logits = logits[:, -2, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()
#The l1 dataset has te answer positive adjective
#The l5 dataset has te answer positive adjective
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()

# %%
clean_tokens=model.to_tokens(l1)
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
clean_logits[:, -2, :].argmax(dim=-1)


# %%
CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff
def sb_metric(logits, answer_token_indices=answer_token_indices):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (CLEAN_BASELINE  - CORRUPTED_BASELINE)

resid_act_mlp_out_results = patching.get_act_patch_mlp_out(model, corrupted_tokens, clean_cache, sb_metric)

# %%
imshow(resid_act_mlp_out_results[:,1:9], 
       yaxis="Layer", 
       xaxis="Position", 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0][1:9]))],
       title="resid_act_mlp_out_results Activation Patching")

# %%
ALL_HEAD_LABELS = [f"L{i}H{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]
if True:
    attn_head_out_act_patch_results = patching.get_act_patch_attn_head_out_by_pos(model, corrupted_tokens, clean_cache, sb_metric)
    attn_head_out_act_patch_results = einops.rearrange(attn_head_out_act_patch_results, "layer pos head -> (layer head) pos")


# %%
imshow(attn_head_out_act_patch_results[:,1:9], 
       yaxis="Layer", 
       xaxis="Position", 
       x=[f"{tok} {i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0][1:9]))],
       title="attn_head_out_act_patch_results Activation Patching")

# %%
#["Residual Stream", "Attn Output", "MLP Output"]

every_block_result = patching.get_act_patch_block_every(model, corrupted_tokens, clean_cache, sb_metric)


# %%
imshow(every_block_result, facet_col=0, title="Activation Patching Per Block", xaxis="Position", yaxis="Layer", zmax=1, zmin=-1, x= [f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(clean_tokens[0][0:9]))])

