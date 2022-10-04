import os
import json
import argparse
from pathlib import Path
from transformers import set_seed
from knowledge_neurons import (
    KnowledgeNeurons,
    initialize_model_and_tokenizer,
    model_type,
    mpararel,
    ALL_MODELS,
    LANGUAGES,
    PARAREL_RELATION_NAMES,
)


def main(args):
    RESULTS_DIR = Path(args.results_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(args.seed)

    # load dataset
    # each item in pararel is the same 'fact' (head/relation/tail) expressed in different ways
    mPARAREL = mpararel()

    # initialize results dicts
    RESULTS = {}
    NEURONS = {}

    # setup model + tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model_name)
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(args.model_name))

    for rel in PARAREL_RELATION_NAMES:
        print("-" * 50)
        print(f"Probing facts for relation: {rel}")
        for int_tuple in mPARAREL[rel][args.int_lang]["vocab"]:
            uuid = f"{int_tuple['sub_uri'].lower()}-{int_tuple['obj_uri'].lower()}"
            # currently we're only doing the analysis for single mask tokens
            mask_token_count = len(tokenizer.tokenize(int_tuple["obj_label"]))
            if mask_token_count > 1:
                continue

            for obs_tuple in mPARAREL[rel][args.obs_lang]["vocab"]:
                # ignore if we don't find the same tuple in other language
                if (
                    uuid
                    != f"{obs_tuple['sub_uri'].lower()}-{obs_tuple['obj_uri'].lower()}"
                ):
                    continue

                # currently we're only doing the analysis for single mask tokens
                mask_token_count = len(tokenizer.tokenize(obs_tuple["obj_label"]))
                if mask_token_count > 1:
                    continue
                
                print(f"Fact identifier: {uuid}")

                # generate prompts for the facts in the intervened language
                int_obj_label = int_tuple["obj_label"]
                int_sentences = []
                for pattern in mPARAREL[rel][args.int_lang]["patterns"]:
                    int_sentences.append(
                        pattern.replace("[X]", int_tuple["sub_label"]).replace(
                            "[Y]", " ".join([tokenizer.mask_token] * mask_token_count),
                        )
                    )

                # generate prompts for the facts in the observed language
                obs_obj_label = obs_tuple["obj_label"]
                obs_sentences = []
                for pattern in mPARAREL["P101"]["fr"]["patterns"]:
                    obs_sentences.append(
                        pattern.replace("[X]", obs_tuple["sub_label"]).replace(
                            "[Y]", " ".join([tokenizer.mask_token] * mask_token_count),
                        )
                    )

                results_this_uuid = {
                    "suppression": {
                        "related": {
                            "pct_change": [],
                            "correct_before": [],
                            "correct_after": [],
                            "intervene_n_prompts": len(int_sentences),
                            "observe_n_prompts": len(obs_sentences),
                            "intervene_lang": "en",
                            "observe_lang": "fr",
                        }
                    },
                    "enhancement": {
                        "related": {
                            "pct_change": [],
                            "correct_before": [],
                            "correct_after": [],
                            "intervene_n_prompts": len(int_sentences),
                            "observe_n_prompts": len(obs_sentences),
                            "intervene_lang": "en",
                            "observe_lang": "fr",
                        }
                    },
                }

                print(f"Discovering the neurons for the fact")
                # get the knowledge for the same fact in English
                neurons = kn.get_refined_neurons(
                    prompts=int_sentences,
                    ground_truth=int_obj_label,
                    p=args.p,
                    batch_size=args.batch_size,
                    steps=args.steps,
                    coarse_adaptive_threshold=args.adaptive_threshold,
                    quiet=True,
                )

                for obs_sentence in obs_sentences:
                    print(f"Suppressing and Enhancing the neurons for the fact")
                    # enhance and supress the information at neuron level and evaluate
                    # the effect of same fact elicited in a different language
                    suppression_results, _ = kn.suppress_knowledge(
                        obs_sentence, obs_obj_label, neurons, quiet=True
                    )
                    enhancement_results, _ = kn.enhance_knowledge(
                        obs_sentence, obs_obj_label, neurons, quiet=True
                    )
                    
                    # get the pct change in probability of the ground truth string being produced before and after suppressing knowledge
                    suppression_prob_diff = (suppression_results["after"]["gt_prob"] - suppression_results["before"]["gt_prob"]) / suppression_results["before"]["gt_prob"]
                    results_this_uuid["suppression"]["related"]["pct_change"].append(suppression_prob_diff)
                    enhancement_prob_diff = (enhancement_results["after"]["gt_prob"] - enhancement_results["before"]["gt_prob"]) / enhancement_results["before"]["gt_prob"]
                    results_this_uuid["enhancement"]["related"]["pct_change"].append(enhancement_prob_diff)
                    
                    # check whether the answer was correct before/after suppression
                    results_this_uuid["suppression"]["related"]["correct_before"].append(
                        suppression_results["before"]["argmax_completion"] == obs_obj_label
                    )
                    results_this_uuid["suppression"]["related"]["correct_after"].append(
                        suppression_results["after"]["argmax_completion"] == obs_obj_label
                    )

                    results_this_uuid["enhancement"]["related"]["correct_before"].append(
                        enhancement_results["before"]["argmax_completion"] == obs_obj_label
                    )
                    results_this_uuid["enhancement"]["related"]["correct_after"].append(
                        enhancement_results["after"]["argmax_completion"] == obs_obj_label
                    )
                
                results_this_uuid["n_refined_neurons"] = len(neurons)
                results_this_uuid["relation_name"] = rel
                RESULTS[uuid] = results_this_uuid
                NEURONS[uuid] = neurons
        
        print("-" * 50, sep="\n")
        
    # save results + neurons to json file
    with open(RESULTS_DIR / f"{args.model_name}_pararel_neurons.json", "w") as f:
        json.dump(NEURONS, f, indent=4)
    with open(RESULTS_DIR / f"{args.model_name}_pararel_results.json", "w") as f:
        json.dump(RESULTS, f, indent=4)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        "Use the Pararel dataset to extract knowledge neurons from a Language Model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help=f"name of the LM to use - choose from {ALL_MODELS}",
    )
    parser.add_argument(
        "--int_lang",
        type=str,
        default="en",
        help=f"language in which the knowledge neurons to be discovered - choose from {LANGUAGES}",
    )
    parser.add_argument(
        "--obs_lang",
        type=str,
        default="fr",
        help=f"language in which the update operations should be performed - choose from {LANGUAGES}",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="bert_base_uncased_neurons",
        help="directory in which to save results",
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of steps to run the integrated gradient calculation for",
    )
    parser.add_argument(
        "--adaptive_threshold",
        type=int,
        default=0.3,
        help="A setting used to determine the score threshold above which coarse neurons are selected - the paper uses 0.3",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.3,
        help="the threshold for the sharing percentage - we retain neurons that are shared by p% of prompts (p here is a decimal fraction, i.e between 0 and 1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    main(args)
