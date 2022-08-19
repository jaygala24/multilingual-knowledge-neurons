# knowledge-neurons

An open source repository (forked from [EluetherAI/knowledge-neurons](https://raw.githubusercontent.com/EleutherAI/knowledge-neurons)) replicating the 2021 paper *[Knowledge Neurons in Pretrained Transformers](https://arxiv.org/abs/2104.08696)* by Dai et al., and extending the technique to autoregressive models, as well as MLMs.

The Huggingface Transformers library is used as the backend, so any model you want to probe must be implemented there. 

Currently integrated models:
```python
BERT_MODELS = [
    "bert-base-uncased", "bert-base-cased", 
    "bert-base-multilingual-uncased", "bert-base-multilingual-cased", 
    "bert-large-uncased", "bert-large-cased"
]
ROBERTA_MODELS = [
    "xlm-roberta-base", "xlm-roberta-large",
    "facebook/xlm-roberta-xl", "facebook/xlm-roberta-xxl"
]
# GPT2_MODELS = ["gpt2"]
# GPT_NEO_MODELS = [
#     "EleutherAI/gpt-neo-125M",
#     "EleutherAI/gpt-neo-1.3B",
#     "EleutherAI/gpt-neo-2.7B",
# ]
```

## Setup

Clone the github, and run scripts from there:

```bash
git clone https://github.com/jaygala24/multilingual-knowledge-neurons.git
cd multilingual-knowledge-neurons
```

## Data

```bash
!gdown --id 1Nz4q3hIdwvs82ErILR1jaBzTuU3tMR8c
!unzip datasets.zip -d datasets/
```

## Usage & Examples

An example using `xlm-robert-base`:

```python
from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
import json
import random
import collections
from tqdm import tqdm

# first initialize some hyperparameters
MODEL_NAME = "xlm-roberta-large"

# these are some hyperparameters for the integrated gradients step
BATCH_SIZE = 20
STEPS = 20 # number of steps in the integrated grad calculation
ADAPTIVE_THRESHOLD = 0.3 # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
P = 0.5 # the threshold for the sharing percentage (p% prompts sharing the neurons)
GROUND_TRUTH = "Paris"
ENG_TEXTS = [
    "Sarah was visiting <mask>, the capital of France",
    "The capital of France is <mask>",
    "<mask> is the capital of France",
    "France's capital <mask> is a hotspot for romantic vacations",
    "The eiffel tower is situated in <mask>",
    "<mask> is the most populous city in France",
    "<mask>, France's capital, is one of the most popular tourist destinations in the world",
]
FRENCH_TEXTS = [
    "Sarah visitait <mask>, la capitale de la France",
    "La capitale de la France est <mask>",
    "<mask> est la capitale de la France",
    "La capitale de la France <mask> est un haut lieu des vacances romantiques",
    "La tour eiffel est située à <mask>",
    "<mask> est la ville la plus peuplée de France",
    "<mask>, la capitale de la France, est l'une des destinations touristiques les plus prisées au monde",
]
TEXTS = ENG_TEXTS + FRENCH_TEXTS

# setup model & tokenizer
model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

# initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert' / 'roberta')
kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

refined_neurons, coarse_neurons, prompts_info = kn.get_refined_neurons(
    prompts=ENG_TEXTS,
    ground_truth=GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
    quiet=True,
)
refined_neurons

refined_neurons, coarse_neurons, prompts_info = kn.get_refined_neurons(
    prompts=FRENCH_TEXTS,
    ground_truth=GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
    quiet=True,
)
refined_neurons

refined_neurons, coarse_neurons, prompts_info = kn.get_refined_neurons(
    prompts=TEXTS,
    ground_truth=GROUND_TRUTH,
    p=P,
    batch_size=BATCH_SIZE,
    steps=STEPS,
    coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
    quiet=True,
)
refined_neurons
```

## References

https://github.com/EleutherAI/knowledge-neurons

## Citations
```bibtex
@article{Dai2021KnowledgeNI,
  title={Knowledge Neurons in Pretrained Transformers},
  author={Damai Dai and Li Dong and Y. Hao and Zhifang Sui and Furu Wei},
  journal={ArXiv},
  year={2021},
  volume={abs/2104.08696}
}
```
