from knowledge_neurons import (
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
)
from data import mpararel_expanded
import json
import collections
from tqdm import tqdm

# first initialize some hyperparameters
MODEL_NAME = "bert-base-multilingual-cased"

# these are some hyperparameters for the integrated gradients step
BATCH_SIZE = 20
STEPS = 20  # number of steps in the integrated grad calculation
ADAPTIVE_THRESHOLD = 0.3  # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
P = 0.5  # the threshold for the sharing percentage
LANG = "fr"  # language to probe the LM
REL = "P101"  # relation to probe the LM

# load dataset
# each item in pararel is the same 'fact' (head/relation/tail) expressed in different ways
mPARAREL = mpararel_expanded()

# setup model & tokenizer
model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

# initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert')
kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))


def get_neurons(fact):
    PROMPTS, GROUND_TRUTH, RELATION_NAME = (
        fact["sentences"],
        fact["obj_label"],
        fact["relation_name"],
    )
    PROMPTS = [
        p[0].upper() + p[1:] + "." if not p.endswith(".") else p[0].upper() + p[1:]
        for p in PROMPTS
    ]
    refined_neurons, coarse_neurons, prompts_info = kn.get_refined_neurons(
        prompts=PROMPTS,
        ground_truth=GROUND_TRUTH,
        p=P,
        batch_size=BATCH_SIZE,
        steps=STEPS,
        coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
        quiet=True,
    )
    return [prompts_info, coarse_neurons, refined_neurons]


FACTS = mPARAREL[LANG][REL]
RESULTS = collections.defaultdict(lambda: collections.defaultdict(dict))
RESULTS[LANG][REL] = []

for FACT in tqdm(FACTS, f"probing the LM for analyzing relations {REL} in {LANG}"):
    RESULTS[LANG][REL].append(get_neurons(FACT))

with open(f"mpararel_{LANG}_{REL}.json", "w") as f:
    json.dump(RESULTS, f, indent=4)
