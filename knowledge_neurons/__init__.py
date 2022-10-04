from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM
from .knowledge_neurons import KnowledgeNeurons
from .data import pararel, pararel_expanded
from .constants import LANGUAGES, PARAREL_RELATION_NAMES

BERT_MODELS = [
    "bert-base-uncased",
    "bert-base-multilingual-uncased",
]
ROBERTA_MODELS = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "facebook/xlm-roberta-xl",
    "facebook/xlm-roberta-xxl",
]
GPT2_MODELS = ["gpt2"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
ALL_MODELS = BERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS


def initialize_model_and_tokenizer(model_name: str):
    if model_name in BERT_MODELS + ROBERTA_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif model_name in GPT_NEO_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval()

    return model, tokenizer


def model_type(model_name: str):
    if model_name in BERT_MODELS:
        return "bert"
    elif model_name in ROBERTA_MODELS:
        return "roberta"
    elif model_name in GPT2_MODELS:
        return "gpt2"
    elif model_name in GPT_NEO_MODELS:
        return "gpt_neo"
    else:
        raise ValueError("Model {model_name} not supported")
