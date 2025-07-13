import os
import json
import torch
import argparse
import openai
import numpy as np
import sys

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from methods.prefixes import *

from utils.constants import *
from utils.metrics import *
from utils.utils import *

from methods.method_fudge.constants import fudge_models

from generate_control_text import generate_control_text


def process_prompt(prompt,
                   task,
                   method,
                   prefix_setting,
                   strength,
                   max_tokens,
                   top_k,
                   top_p,
                   base_model_string,
                   fudge_model_string,
                   display):
    original_text = json.loads(prompt)["origin_text"] if method == "calib" else None
    label = json.loads(prompt)["label"]
    prompt = json.loads(prompt)["text"]
    if len(prompt) < 3:
        return None

    output = generate_control_text(
        task=task,
        method=method,
        prompt=prompt,
        prefix_setting=prefix_setting,
        strength=strength,
        max_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        base_model_string=base_model_string,
        fudge_model_string=fudge_model_string
    )

    generated_text = output["full_text"]
    raw_text = output["raw_text"]
    original_text = output["original_text"] if original_text is None else original_text

    if generated_text.startswith('</s>'):
        generated_text = generated_text[4:].strip()

    if display:
        print("Prompt:", prompt, "\n")
        print(f"Generated Text by {method}:", generated_text, "\n")

    return generated_text, raw_text, original_text, {
        "raw_text": raw_text,
        "content": generated_text,
        "label": label
    }


@torch.no_grad()
def generate_sentiment_evals(save_dir,
                            prompts_setting,
                            method,
                            prefix_setting,
                            strength,
                            max_tokens,
                            top_k,
                            top_p,
                            base_model_string,
                            fudge_model_string,
                            task,
                            display):
    prompts = prompt_datasets[prompts_setting]

    generations = []
    generated_pairs = []
    outputs = []

    assert prompts.endswith('.jsonl')
    with open(prompts, "r") as f:
        prompts_list = f.readlines()

    max_workers = 2 if args.method == "fudge" else 4
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        print(f"{max_workers}路并行")
        future_to_prompt = {
            executor.submit(
                process_prompt,
                prompt, task, method, prefix_setting, strength,
                max_tokens, top_k, top_p, base_model_string, fudge_model_string, display
            ): prompt for prompt in prompts_list
        }

        with tqdm(total=len(prompts_list), file=sys.stdout) as pbar:
            for future in as_completed(future_to_prompt):
                pbar.update(1)
                generated_text, raw_text, original_text, result = future.result()
                try:
                    if result is not None:
                        outputs.append(result)
                        if generated_text is not None:
                            generations.append(generated_text)
                            # generations.append(raw_text)
                            generated_pairs.append((original_text, raw_text))
                except Exception as e:
                    print(f"a Task failed with exception: {e}")

    # Perplexity & Grammaticality
    ppl, _, _ = perplexity(generations, model_name='gpt2-chinese-cluecorpussmall')
    judge_results = sentiment_judge(generations)
    cos = cosine_similarity(generated_pairs)

    for i in range(len(outputs)):
        outputs[i]["perplexity"] = ppl[i]
        outputs[i]["cosine_similarity"] = cos[i]
        outputs[i]["judge_result"] = judge_results[i]
        outputs[i]["success"] = 1 if outputs[i]["label"] == judge_results[i] else 0
    write_eval_output_file(outputs, save_dir, prompts_setting, method, prefix_setting, strength, top_k, top_p)

    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--prompts_setting', required=True,
                        type=str, choices=sentiment_prompts)
    parser.add_argument('--method', required=True, type=str, choices=methods)
    parser.add_argument('--prefix_setting', type=str,
                        default="pos", choices=list(sentiment_prefixes.keys()))
    parser.add_argument('--strength', type=float, default=-1.0)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=0)  # default is no top_k
    parser.add_argument('--top_p', type=float, default=1)  # default is no top_p
    parser.add_argument('--base_model_string', type=str,
                        default='facebook/opt-125m', choices=models)
    parser.add_argument('--auxiliary_model_string', type=str,
                        default='facebook/opt-125m', choices=models)
    parser.add_argument('--fudge_model_string', type=str,
                        default='facebook/opt-125m', choices=fudge_models)
    parser.add_argument('--display', action='store_true', default=False)
    args = parser.parse_args()

    # Evaluate model text generations
    generate_sentiment_evals(save_dir=args.save_dir,
                            prompts_setting=args.prompts_setting,
                            method=args.method,
                            prefix_setting=args.prefix_setting,
                            strength=args.strength,
                            max_tokens=args.max_tokens,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            base_model_string=args.base_model_string,
                            fudge_model_string=args.fudge_model_string,
                            task="sentiment",
                            display=args.display)




