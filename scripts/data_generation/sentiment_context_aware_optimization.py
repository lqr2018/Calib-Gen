import argparse
import json
import sys
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelWithLMHead

from methods.calib_gen import context_aware_optimization
from utils.constants import *
from utils.engine_util import *
from utils.metrics import *
from methods.prefixes import *

def main(args):
    prompts = prompt_datasets[args.prompts_setting]
    output_prompts = prompt_datasets[args.output_prompts_setting]
    assert prompts.endswith(".jsonl")
    assert output_prompts.endswith(".jsonl")

    with open(prompts, "r") as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    lines = []

    # for i, line in tqdm(enumerate(data), file=sys.stdout):
    for i, line in tqdm(enumerate(data), total=len(data), desc="Processing", file=sys.stdout):
        result = context_aware_optimization(line['text'], args.prefix_setting, args.strength, task="sentiment")
        line['text'] = result['full_text']
        line['origin_text'] = result['original_text']

        lines.append(line)

    with open(output_prompts, "w") as f:
        for line in lines:
            json.dump(line, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompts_setting', required=True, type=str, choices=sentiment_prompts)
    parser.add_argument('--prefix_setting', type=str, default="pos", choices=list(sentiment_prefixes.keys()))
    parser.add_argument('--output_prompts_setting', required=True, type=str, choices=sentiment_prompts)
    parser.add_argument('--strength', default=-0.2, type=float)

    args = parser.parse_args()

    main(args)