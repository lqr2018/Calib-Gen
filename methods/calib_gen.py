import argparse
from torch.nn import functional as F

from utils.utils import *
from utils.constants import *
from utils.engine_util import *
from methods.prefixes import all_prefixes
from methods.base_method import BaseMethod

def context_aware_optimization(prompt, prefix_setting, strength=-0.2, task="toxicity"):
    model_string = 'THUDM/chatglm3-6b'
    prefix = all_prefixes[task][prefix_setting]

    tokenizer = get_tokenizer(model_string)

    tokens = tokenizer.encode(prompt)[:64]

    prompt_length = len(tokens)
    tokens = tokens[:int(prompt_length/2)]

    prompt = tokenizer.decode(tokens).replace("[gMASK] sop ", "")

    calib = CalibGen(prompt=prompt,
                     prefix=prefix,
                     strength=strength,
                     max_tokens=int(prompt_length/2),
                     model_string=model_string)

    result = calib.muse()

    # print(result['full_text'])
    return result

class CalibGen(BaseMethod):
    def __init__(self, prompt, prefix, strength=0.0, max_tokens=32, temperature=0.8, top_k=100, top_p=0.8, model_string='facebook/opt-125m'):
        super().__init__(prompt, prefix, max_tokens, temperature, model_string)
        self.prompt = prompt
        self.prefix = prefix
        self.strength = strength
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model_string = model_string
        self.tokenizer = get_tokenizer(model_string)

    def _top_p(self, logits, control_strength, filter_value, min_tokens_to_keep=1):
        filter_value = filter_value if control_strength >= 0.5 else 0

        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > self.top_p
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

    def _modified_sampling_methods(self, base_logprobs, prefix_logprobs, filter_value=-1e8, min_tokens_to_keep=1):
        base_logprobs = self._top_p(base_logprobs, control_strength=1 - self.strength, top_k=self.top_k, top_p=self.top_p)[0]
        top_logprobs, top_indices = torch.topk(base_logprobs, self.top_k)
        prefix_logprobs = self._top_p(prefix_logprobs, control_strength=self.strength, top_k=self.top_k, top_p=self.top_p)[0]

        base_logprobs = top_logprobs
        prefix_logprobs = find_corresponding_logprobs(prefix_logprobs, top_indices)

        return base_logprobs, prefix_logprobs, top_indices


    def muse(self):
        return self.generate()

    def generate(self):
        str = "This sentence is written in English."
        # str = ""

        prompt = self.tokenizer.encode(str + " ") + self.tokenizer.encode(self.prompt)[:64]
        str_length = len(self.tokenizer.encode(str + " "))
        prompt_length = len(prompt)
        prompt_cache_id = None

        prefix = self.tokenizer.encode(self.prefix + " ") + prompt
        prefix_cache_id = None

        for i in range(self.max_tokens):
            prompt_output = get_next_logprobs(
                prompt, self.model_string, cache_id=prompt_cache_id)
            prompt_cache_id = prompt_output['cache_id']

            prefix_output = get_next_logprobs(
                prefix, self.model_string, cache_id=prefix_cache_id)
            prefix_cache_id = prefix_output['cache_id']

            base_logprobs = torch.Tensor(prompt_output['logits'])[None, :]
            prefix_logprobs = torch.Tensor(prefix_output['logits'])[None, :]
            base_logprobs, prefix_logprobs, top_indices = self._modified_sampling_methods(base_logprobs, prefix_logprobs)

            diff = prefix_logprobs - base_logprobs

            final_logprobs = (base_logprobs + diff * self.strength) / self.temperature

            next_token = top_indices[torch.multinomial(torch.softmax(final_logprobs, dim=0), 1)].item()

            prompt.append(next_token)
            prefix.append(next_token)

        return {
            "full_text": self.tokenizer.decode(prompt[str_length:]).replace("[gMASK] sop ", ""),
            "raw_text": self.tokenizer.decode(prompt[prompt_length:]).replace("[gMASK] sop ", ""),
            "original_text": self.prompt
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_prompt', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_string', type=str, default='facebook/opt-125m', choices=models)
    args = parser.parse_args()
    calib = CalibGen(args.control_prompt, max_tokens=args.max_tokens, temperature=args.temperature, model_string=args.model_string)
    print(calib.muse())