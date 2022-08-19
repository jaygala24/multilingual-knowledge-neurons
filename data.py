import json
import urllib.request
from urllib.error import HTTPError
import collections
from tqdm import tqdm
from pathlib import Path
import os
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from constants import LANGUAGES, PARAREL_RELATION_NAMES


def mpararel(data_path: str = "datasets/mpararel.json"):
    parent_dir = Path(data_path).parent
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        mPARAREL = collections.defaultdict(lambda: collections.defaultdict(dict))
        # download relations from github
        for lang in tqdm(LANGUAGES, "downloading mpararel dataset"):
            for rel in PARAREL_RELATION_NAMES:
                try:
                    with urllib.request.urlopen(
                        f"https://raw.githubusercontent.com/coastalcph/mpararel/master/data/mpararel_reviewed/patterns/{lang}/{rel}.jsonl",
                        timeout=None,
                    ) as url:
                        mPARAREL[lang][rel]["patterns"] = [
                            json.loads(d.strip())["pattern"]
                            for d in url.read().decode().split("\n")
                            if d
                        ]
                except HTTPError:
                    continue

                try:
                    with urllib.request.urlopen(
                        f"https://raw.githubusercontent.com/coastalcph/mpararel/master/data/mpararel_reviewed/patterns/{lang}/{rel}.jsonl",
                        timeout=None,
                    ) as url:
                        mPARAREL[lang][rel]["vocab"] = [
                            json.loads(d.strip())
                            for d in url.read().decode().split("\n")
                            if d
                        ]
                except HTTPError:
                    del mPARAREL[lang][rel]

        with open(data_path, "w") as f:
            json.dump(mPARAREL, f)
        return mPARAREL


def mpararel_expanded(
    tokenizer: PreTrainedTokenizerBase = None,
    data_path: str = "datasets/mpararel_expanded.json",
):
    parent_dir = Path(data_path).parent
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(data_path) and tokenizer is None:
        with open(data_path, "r") as f:
            return json.load(f)
    else:
        mPARAREL = mpararel()
        mPARAREL_EXPANDED = collections.defaultdict(
            lambda: collections.defaultdict(dict)
        )
        # expand relation templates into sentences
        for lang, rels in tqdm(
            mPARAREL.items(), "expanding mpararel dataset into full sentences"
        ):
            for rel, value in rels.items():
                mPARAREL_EXPANDED[lang][rel] = []
                for vocab in value["vocab"]:
                    full_sentences = []
                    for pattern in value["patterns"]:
                        mask_token_count = len(tokenizer.tokenize(vocab["obj_label"]))
                        full_sentences.append(
                            pattern.replace("[X]", vocab["sub_label"]).replace(
                                "[Y]",
                                " ".join([tokenizer.mask_token] * mask_token_count),
                            )
                        )
                    mPARAREL_EXPANDED[lang][rel].append(
                        {
                            "sentences": full_sentences,
                            "sub_label": vocab["sub_label"],
                            "obj_label": vocab["obj_label"],
                            "relation_name": rel,
                        }
                    )
        with open(data_path, "w") as f:
            json.dump(mPARAREL_EXPANDED, f)
        return mPARAREL_EXPANDED


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    data = mpararel_expanded(tokenizer)
