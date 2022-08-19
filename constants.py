LANGUAGES = [
    'af', 'ar', 'az', 'bg', 'bn', 'ca', 'cs', 'cy', 'da', 'de', 'el', 'en', 
    'es', 'et', 'fa', 'fi', 'fr', 'ga', 'gl', 'he', 'hr', 'hu', 'hy', 'id', 
    'is', 'it', 'ja', 'ko', 'lt', 'lv', 'ms', 'nl', 'pl', 'pt', 'ro', 'ru', 
    'sk', 'sl', 'sq', 'sv', 'th', 'tr', 'uk', 'vi', 'zh-hans', 'zh-hant'
]

PARAREL_RELATION_NAMES = [
    'P937', 'P1412', 'P127', 'P103', 'P159', 'P140', 'P136', 'P495', 'P276',
    'P17', 'P361', 'P36', 'P740', 'P264', 'P407', 'P138', 'P30', 'P131',
    'P176', 'P449', 'P279', 'P19', 'P101', 'P364', 'P106', 'P1376', 'P178',
    'P37', 'P413', 'P27', 'P20', 'P190', 'P1303', 'P39', 'P108', 'P463',
    'P530', 'P47'
]

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
#     "EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"
# ]
ALL_MODELS = BERT_MODELS + ROBERTA_MODELS
