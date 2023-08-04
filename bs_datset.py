# Define a positive negative adjective pair
# Define a positive negateice names pair
# Defien pair of semantic name/adj pairs both positive and adjectives
# Defien adjectives
# Define names
# Templates with 'but' --> (We can interchange with 'and')
# Templates without 'but' 
# Short and long template
# gen_prompt_uniform
# gen_flipped_prompts
# class BSDataset

# Make sure all prompts have the same length





import io
from logging import warning
from typing import Union, List
from site import PREFIXES
import warnings
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer
import random
import re
import matplotlib.pyplot as plt
import copy
import itertools
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache
import json
import pickle as pkl
model = HookedTransformer.from_pretrained("gpt2-small")

#---------------------------------------------



pos_nouns=["day","summer","victory","war","man","king"]
neg_nouns=["night","winter","defeat","peace","woman","queen"]
pos_adj=["bright","hot","sweet","bad","strong","good"]
neg_adj=["dark","cold","bitter","good","weak","bad"]

COMMON_NOUNS = [
    "apple", "ball", "cat", "dog", "elephant", "flower", "guitar", "hat",
    "ice", "jelly", "kite", "lemon", "mountain", "nest", "ocean", "pencil",
    "quilt", "river", "sun", "tree", "umbrella", "violin", "whale", "xylophone",
    "yarn", "zebra"
]
# without space [1, 1, 1, 1, 2, 1, 3, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 3, 2, 2]
# with left space [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 2]

adjective_pairs = [
    ("hot", "cold"), ("young", "old"), ("happy", "sad"), ("empty", "full"),
    ("fast", "slow"), ("hard", "soft"), ("light", "heavy"), ("early", "late"),
    ("rich", "poor"), ("strong", "weak"), ("high", "low"), ("dry", "wet"),
    ("sweet", "sour"), ("tight", "loose"), ("long", "short"), ("sharp", "dull")
]
# without space [[1, 1], [1, 1], [1, 2], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 2], [1, 2], [1, 2], [1, 1], [1, 2]]
# with left space [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]


common_noun_pairs = [
    ("day", "night"), ("win", "lose"), ("up", "down"), ("left", "right"),
    ("begin", "end"), ("open", "close"), ("inside", "outside"), ("buy", "sell")
]
# without space [[1, 1], [1, 2], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
# with left space [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]


gender_sensitive_noun_pairs = [
    ("king", "queen"), ("husband", "wife"), 
    ("waiter", "waitress"), ("prince", "princess"), ("father", "mother"),
    ("son", "daughter"), ("brother", "sister"), ("uncle", "aunt"),
    ("nephew", "niece"), ("gentleman", "lady"), ("boy", "girl"),
    ("male", "female"), ("grandfather", "grandmother"), ("god", "goddess"),
    ("hero", "heroine"), ("widower", "widow"), ("sir", "madam"),
    ("emperor", "empress"), ("baron", "baroness"), ("duke", "duchess")
]
# without spcae [[1, 2], [1, 1], [2, 2], [2, 3], [1, 1], [1, 1], [1, 2], [1, 1], [3, 2], [3, 2], [1, 1], [1, 1], [2, 2], [1, 3], [1, 2], [2, 2], [2, 2], [2, 2], [2, 3], [2, 3]]
# with left space [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [2, 1], [1, 2], [1, 2], [2, 3], [2, 3]]


CONJ_pair=[("but","and"),("while","so")]

CONJ_TEMPLATE_paired=[
    "The [A] is [B], [CONJ] the [a] is [b]"
]

CONJ_TEMPLATE_unpaired=[
        template.replace("[A]", "[a1]",1).replace("[a]", "[A]", 1).replace("a1","a",1)
        for template in CONJ_TEMPLATE_paired

]

CONJ_TEMPLATE=[
    "The [A1] is [B1], [CONJ] the [A2] is [B2]"
]

adj_nouns={"common_nouns":COMMON_NOUNS,"adjective_pairs":adjective_pairs,"common_noun_pairs":common_noun_pairs,"gender_sensitive_noun_pairs":gender_sensitive_noun_pairs}
with open("datasets/adj_nouns_lists.json","w") as f:
        json.dump(adj_nouns,f)

#Datasets for experiments
# Outter semantic relation-->Example: The day is bright but the night is dark.
# Outter semantic relation (gender sensitive)-->Example: The king is powerfull but the queen is weak.
# Inner semantic realtion of both noun and adjective-->Example: The  day is hard but the night is soft.
# Inner semantic realtion of both noun (gender sensitive) and adjective-->Example: The king is empty but the queen is full
# Inner semantic relation of adjectives-->Example: The day is bright but the king is dark.

# Metric: Logit differnece of induction vs extrapolation

# Patching: 
#----
# For all how does the performace change when changging the conjunction by the oposite.---> Example: The day is dark, and the night is 
#----
# For the outter semantic relation, interchange the nouns:-->Example: The day is dark but the night is bright.
# For the inner semantic relation of both nouns and adjectives, interchange one of the nouns by a random noun. The day is hard, but the king  




#-----------------------

# Functions to get only single token words
def length_tokenized_pairs(pair_list):
    lengths=list()
    for pair in pair_list:
        pair_tok=list()
        for term in pair:
            term=" "+term
            pair_tok.append(len(model.to_tokens(term,prepend_bos=False)[0]))
        lengths.append(pair_tok)
    return lengths

def length_tokenized_list(lista):
    lengths=list()
    for term in lista:
         term=" "+term
         lengths.append(len(model.to_tokens(term,prepend_bos=False)[0]))
    return lengths


def filter_pairs(list_pair):
    lengths= length_tokenized_pairs(list_pair)
    return [1 if l==[1,1] else 0 for l in lengths]

def filter_list(lista):
    lengths= length_tokenized_list(lista)
    return [1 if l==1 else 0 for l in lengths]


def get_filtered_pairs(list_pair):
    filtro=filter_pairs(list_pair)
    return [item for item, should_include in zip(list_pair, filtro) if filtro]
def get_filtered_list(lista):
    filtro=filter_list(lista)
    return [item for item, should_include in zip(lista, filtro) if filtro]

#-----------------------


def multiple_replace(dict, text):
    # Create a regular expression from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start() : mo.end()]], text)

def gen_outter_prompts(templates,pos_nouns,neg_nouns,pos_adj,neg_adj
        ):
    prompts=[]
    for cj in conj:
        for a,b,c,d in zip(pos_nouns,neg_nouns,pos_adj,neg_adj):
            prompts.append(multiple_replace({"[A]":a,"[B]":c,"[a]":b,"[b]":d,"[CONJ]":cj[0]},templates[0]))
    return prompts


def gen_inner_both_prompts(
        templates, noun_dict,adj_dict,conj):
    prompts=[]
    
    
    for n_dict in noun_dict: 
        for a_dict in adj_dict:
                for cj in conj:
                    prompts.append(multiple_replace({"[A]":n_dict[0],"[B]":a_dict[0],"[a]":n_dict[1],"[b]":a_dict[1],"[CONJ]":cj[0]},templates[0]))
    return prompts


def gen_inner_adj_prompts(templates,noun_list, adj_list,conj,N=4,seed=123):
    prompts=[]
    if len(noun_list) < 2 or len(adj_list) < 2:
        raise ValueError("Both lists must have at least two elements.")
    
    # Create all possible combinations of 2 elements for each list
    combinations1 = list(itertools.combinations(noun_list, 2))
    combinations2 = list(itertools.combinations(adj_list, 2))
    
    # Create a product of the two combinations lists
    all_combinations = list(itertools.product(combinations1, combinations2))
    
    # Randomly select a subset of the combinations
    selected_combinations = random.sample(all_combinations, min(N, len(all_combinations)))
    for ((a, b), (c, d)) in selected_combinations:
        for cj in conj:
            prompts.append(multiple_replace({"[A]":a,"[B]":c,"[a]":b,"[b]":d,"[CONJ]":cj[0]},templates[0]))
    return prompts


def gen_flipped_prompts_conj(prompts, conj):
    flipped_prompts=list()
    for prompt in prompts:
        if conj[0][0] in prompt:
            flipped_prompts.append(prompt.replace(conj[0][0],conj[1][0]))
        if conj[1][0] in prompt:
            flipped_prompts.append(prompt.replace(conj[1][0],conj[1][1]))
    return flipped_prompts


# Tokenizer function
def tokenize(prompts):
    tokenized=list()
    for prompt in prompts:
        toks = model.to_tokens(prompt)
        tokenized.append(toks)
    return tokenized


# Generate the prompts
#------------------------
# Outter semantic relation clean conjunction (but,while)
conj=CONJ_pair
pos_nouns=get_filtered_list(pos_nouns)
neg_nouns=get_filtered_list(neg_nouns)
pos_adj=get_filtered_list(pos_adj)
neg_adj=get_filtered_list(neg_adj)
outter_clean_conj=gen_outter_prompts(CONJ_TEMPLATE_paired,pos_nouns,neg_nouns,pos_adj,neg_adj)
# Outter semantic relation clean conjunction (and,so)
outter_corrupted_conj=gen_flipped_prompts_conj(outter_clean_conj,conj)
#--------------------
# Outter semantic relation clean 
conj=CONJ_pair
pos_nouns=get_filtered_list(pos_nouns)
neg_nouns=get_filtered_list(neg_nouns)
pos_adj=get_filtered_list(pos_adj)
neg_adj=get_filtered_list(neg_adj)
outter_clean_name=gen_outter_prompts(CONJ_TEMPLATE_paired,pos_nouns,neg_nouns,pos_adj,neg_adj)
# Outter semantic relation corrupted 
pos_nouns_altered=neg_nouns
neg_nouns_altered=pos_nouns
outter_corrupted_name=gen_outter_prompts(CONJ_TEMPLATE_paired,pos_nouns_altered,neg_nouns_altered,pos_adj,neg_adj)

#----------------------------------
# Inner semantic realtion of both noun and adjective Clean (but,while)
noun_dict=get_filtered_pairs(common_noun_pairs)
adj_dict=get_filtered_pairs(adjective_pairs)
conj=CONJ_pair
inner_sem_both_clean_conj=gen_inner_both_prompts(CONJ_TEMPLATE_paired,noun_dict,adj_dict,conj)
# Inner semantic realtion of both noun and adjective Corrupted (and,so)
inner_sem_both_corrupted_conj=gen_flipped_prompts_conj(inner_sem_both_clean_conj,conj)
#----------------------------------
# Inner semantic realtion of both noun and adjective Clean
noun_dict=get_filtered_pairs(common_noun_pairs)
adj_dict=get_filtered_pairs(adjective_pairs)
conj=CONJ_pair
inner_sem_both_clean_name=gen_inner_both_prompts(CONJ_TEMPLATE_paired,noun_dict,adj_dict,conj)
# Inner semantic realtion of both noun and adjective Corrupted 
noun_dict=get_filtered_pairs(common_noun_pairs)
noun_dict_invert=[tuple(reversed(t)) for t in noun_dict]
inner_sem_both_corrupted_name=gen_inner_both_prompts(CONJ_TEMPLATE_paired,noun_dict_invert,adj_dict,conj)
#-----------------------------------
# Outter semantic relation (gender sensitive) TODO
# Inner semantic realtion of both noun (gender sensitive) and adjective Clean TODO
# Inner semantic realtion of both noun (gender sensitive) and adjective Corrupted TODO
# Inner semantic relation of adjectives Clean TODO
# Inner semantic relation of adjectives Corrupted TODO

#------------------
with open("datasets/inner_conj_clean.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_clean_conj)],f)
with open("datasets/inner_conj_corrupt.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_corrupted_conj)],f)
with open("datasets/inner_name_clean.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_clean_name)],f)
with open("datasets/inner_name_corrupt.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_corrupted_name)],f)

print([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_clean_conj)])
print([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_corrupted_conj)])
print([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_clean_name)])
print([[p[0][:-1],p[0][-1]] for p in tokenize(inner_sem_both_corrupted_name)])

with open("datasets/outter_conj_clean.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(outter_clean_conj)],f)
with open("datasets/outter_conj_corrupt.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(outter_corrupted_conj)],f)
with open("datasets/outter_name_clean.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(outter_clean_name)],f)
with open("datasets/outter_name_corrupt.pkl","wb") as f:
    pkl.dump([[p[0][:-1],p[0][-1]] for p in tokenize(outter_corrupted_name)],f)

print([[p[0][:-1],p[0][-1]] for p in tokenize(outter_clean_conj)])
print([[p[0][:-1],p[0][-1]] for p in tokenize(outter_corrupted_conj)])
print([[p[0][:-1],p[0][-1]] for p in tokenize(outter_clean_name)])
print([[p[0][:-1],p[0][-1]] for p in tokenize(outter_corrupted_name)])
# Save the tokenized prompts as a dict where the key is the prompt and the last adjective is the key







