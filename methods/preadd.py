import argparse
from utils.utils import *
from utils.constants import *
from utils.engine_util import *
from methods.base_method import BaseMethod


class PreAdd(BaseMethod):
    def __init__(self, prompt, prefix="", strength=0, max_tokens=32, temperature=0.8, top_k=0, top_p=1, model_string="facebook/opt-125m"):
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

            base_logprobs = \
            top_k_top_p_filtering(torch.Tensor(prompt_output['logits'])[None, :], control_strength=1 - self.strength,
                                  top_k=self.top_k, top_p=self.top_p)[0]  # top-k/top-p filtering
            prefix_logprobs = \
            top_k_top_p_filtering(torch.Tensor(prefix_output['logits'])[None, :], control_strength=self.strength,
                                  top_k=self.top_k, top_p=self.top_p)[0]  # top-k/top-p filtering

            diff = prefix_logprobs - base_logprobs

            final_logprobs = (base_logprobs + diff * self.strength) / self.temperature

            next_token = torch.multinomial(torch.softmax(final_logprobs, dim=0), 1).item()

            prompt.append(next_token)
            prefix.append(next_token)

        return {
            "full_text": self.tokenizer.decode(prompt[str_length:]).replace("[gMASK] sop ", ""),
            "raw_text": self.tokenizer.decode(prompt[prompt_length:]).replace("[gMASK] sop ", ""),
            "original_text": self.prompt
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--strength', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1)
    parser.add_argument('--thresh', type=float, default=0)
    parser.add_argument('--model_string', type=str,
                        default='facebook/opt-125m', choices=models)
    args = parser.parse_args()
    preadd = PreAdd(args.prompt, args.prefix, strength=args.strength,
          max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, thresh=args.thresh, model_string=args.model_string)
    print(preadd.generate())
