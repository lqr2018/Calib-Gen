import os
import json
import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from sentence_transformers import util
from transformers import AutoTokenizer, AutoModel

from utils.constants import *

aux_to_base_map = None

def unpack_scores(response):
    """Extracts Perspective API scores from request response"""

    if not response:
        return None

    attribute_scores = response['attributeScores'].items()

    summary_scores = {}
    span_scores = {}
    for attribute, scores in attribute_scores:
        # Save summary score
        assert scores['summaryScore']['type'] == 'PROBABILITY'
        summary_scores[attribute] = scores['summaryScore']['value']

        # Save span scores
        for span_score_dict in scores['spanScores']:
            assert span_score_dict['score']['type'] == 'PROBABILITY'
            span = (span_score_dict['begin'], span_score_dict['end'])
            span_scores.setdefault(span, {})[
                attribute] = span_score_dict['score']['value']

    return summary_scores, span_scores

def make_filename(save_dir, prompts_setting, method, prefix_setting, strength, top_k, top_p):
    decode_str = ""
    if top_k > 0:
        decode_str += f"top_k={top_k}"
    if top_p < 1:
        decode_str += f"top_p={top_p}"
    if not os.path.exists(f"{save_dir}/{prompts_setting}"):
        os.makedirs(f"{save_dir}/{prompts_setting}")
    if method == "calib":
        filename = f"{save_dir}/{prompts_setting}/{method}_{prefix_setting}_{strength}_{decode_str}_outputs.jsonl"
    else:
        filename = f"{save_dir}/{prompts_setting}/{method}_outputs.jsonl"
    return filename

def write_eval_output_file(outputs, save_dir, prompts_setting, method, prefix_setting, strength, top_k, top_p):
    filename = make_filename(save_dir, prompts_setting, method, prefix_setting, strength, top_k, top_p)

    print(f"output is in {filename}")
    with open(filename, "w") as f:
        for output in outputs:
            json.dump(output, f)
            f.write("\n")

def top_k_top_p_filtering(
    logits: Tensor,
    control_strength,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -1e8,
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with the highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    Taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def find_corresponding_logprobs(logprobs, indices):
    return torch.gather(logprobs, dim=-1, index=indices)