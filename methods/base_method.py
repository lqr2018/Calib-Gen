import argparse
from utils.utils import *
from utils.constants import *
from utils.engine_util import *


class BaseMethod:
    def __init__(self, prompt, prefix="", max_tokens=32, temperature=0.8, model_string="facebook/opt-125m"):
        self.prompt = prompt
        self.prefix = prefix + " " if prefix != "" else ""
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_string = model_string
        self.tokenizer = get_tokenizer(model_string)

    def generate(self):
        prompt = self.tokenizer.encode(self.prefix) + self.tokenizer.encode(self.prompt)[:64]
        prefix_length = len(self.tokenizer.encode(self.prefix))
        prompt_length = len(prompt)
        prompt_cache_id = None

        for i in range(self.max_tokens):
            prompt_output = get_next_logprobs(prompt, self.model_string, cache_id=prompt_cache_id)
            prompt_cache_id = prompt_output['cache_id']

            final_logprobs = torch.Tensor(prompt_output['logits']) / self.temperature

            next_token = torch.multinomial(torch.softmax(final_logprobs, dim=0), 1).item()
            prompt.append(next_token)

        return {
            "full_text": self.tokenizer.decode(prompt[prefix_length:]).replace("[gMASK] sop ", ""),
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
    base_method = BaseMethod(args.control_prompt, max_tokens=args.max_tokens, temperature=args.temperature, model_string=args.model_string)
    print(base_method.generate())
