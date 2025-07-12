import argparse
import torch
from transformers import AutoTokenizer

from utils.utils import *
from utils.constants import *
from utils.engine_util import *
from methods.method_fudge.util import load_fudge_model
from methods.base_method import BaseMethod

"""
Controlled Generation with FUDGE

https://arxiv.org/abs/2104.05218
"""

tokenizer_from = None
tokenizer_to = None

class Fudge(BaseMethod):
    def __init__(self, prompt, strength=0.0, max_tokens=32, temperature=1, model_string='facebook/opt-125m', control_model_string='facebook/opt-125m', task=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(prompt, "", max_tokens, temperature, model_string)
        self.prompt = prompt
        self.strength = strength
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model_string = model_string
        self.control_model_string = control_model_string
        self.task = task
        self.device = device
        self.tokenizer = get_tokenizer(model_string)
        self.control_tokenizer = get_tokenizer(control_model_string)

    def _convert_tokens_between_models(self, tokens_from):
        tokens_to = []

        min_length = 1e8
        for _ in tokens_from:
            text = self.tokenizer.decode(_).replace("</s>", "")
            tokens_to.append(self.control_tokenizer.encode(text))
            min_length = min(len(self.control_tokenizer.encode(text)), min_length)

        tokens_tensor = []

        for i, tokens in enumerate(tokens_to):
            tokens_tensor.append(tokens[:min_length])

        tokens_to = torch.tensor(tokens_tensor)
        tokens_to = tokens_to.to(tokens_from.device)

        return tokens_to

    def _find_control_model(self, ckpt_folder):
        if self.task == 'toxicity':
            control_model = load_fudge_model(
                f'./methods/method_fudge/{ckpt_folder}/toxicity_model_best.pth', self.task, self.control_model_string)
        elif self.task == 'sentiment_neg':
            control_model = load_fudge_model(
                f'./methods/method_fudge/{ckpt_folder}/sentiment_neg_model_best.pth', self.task, self.control_model_string)
        elif self.task == 'sentiment_pos':
            control_model = load_fudge_model(
                f'./methods/method_fudge/{ckpt_folder}/sentiment_pos_model_best.pth', self.task, self.control_model_string)
        else:
            raise NotImplementedError

        return control_model

    def generate(self):
        # prompt should be a string
        # tokenizer = get_tokenizer(model_string)
        # control_tokenizer = AutoTokenizer.from_pretrained(control_model_string)

        prompt = self.tokenizer.encode(self.prompt)[:64]
        prompt_length = len(prompt)
        prompt_cache_id = None

        control_prompt = self.control_tokenizer.encode(self.tokenizer.decode(prompt).replace("[gMASK] sop ", ""))
        # control_prompt_length = len(control_prompt)

        # load in FUDGE control model
        if self.control_model_string == 'facebook/opt-125m':
            ckpt_folder = 'ckpt'
        else:
            raise NotImplementedError

        control_model = self._find_control_model(ckpt_folder)

        for i in range(self.max_tokens):
            prompt_output = get_next_logprobs(prompt, self.model_string, cache_id=prompt_cache_id)
            prompt_cache_id = prompt_output['cache_id']

            base_logits = (torch.Tensor(prompt_output['logits']) / self.temperature).to(self.device)
            topk = 100

            # get logprobs and indices corresponding to the topk tokens in prompt_output
            top_logits, top_indices = torch.topk(base_logits, topk)  # dim = topk

            # form input to control model by trying to append each topk candidate to current sequence
            prompt_topk_candidates = torch.cat([torch.LongTensor(control_prompt).to(self.device).unsqueeze(
                0).expand(topk, -1), top_indices.unsqueeze(1)], dim=1)  # dim = topk x (seq+1)

            # if the base model and the control model use the same tokenizer, this step should be omitted and use the next line
            fudge_prompt_candidates = self._convert_tokens_between_models(prompt_topk_candidates)
            # fudge_prompt_candidates = prompt_topk_candidates

            # plug into control model
            if self.strength == 0:
                control_offset = torch.zeros_like(top_logits).float()
            else:
                control_logits = control_model(fudge_prompt_candidates)[:, -1, 0]

                negative_logprobs = torch.log(1 - torch.sigmoid(control_logits))
                control_offset = negative_logprobs

            # combine logprobs to get final ones corresponding to top indices
            final_logits = top_logits + self.strength * control_offset

            # Decoding Options:
            next_token = top_indices[torch.multinomial(
                torch.softmax(final_logits, dim=0), 1)].item()

            prompt.append(next_token)
            control_prompt = self.control_tokenizer.encode(self.tokenizer.decode(prompt).replace("[gMASK] sop ", ""))

        return {
            "full_text": self.tokenizer.decode(prompt).replace("[gMASK] sop ", ""),
            "raw_text": self.tokenizer.decode(prompt[prompt_length:]).replace("[gMASK] sop ", ""),
            "original_text": self.prompt
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_prompt', type=str, required=True)
    parser.add_argument('--strength', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--model_string', type=str, default='facebook/opt-125m',
                        choices=[model for model in models if "facebook/opt" in model])
    parser.add_argument('--task', type=str, default=None,
                        choices=["toxicity", "bias", None])
    args = parser.parse_args()
    fudge = Fudge(args.control_prompt, strength=args.strength, max_tokens=args.max_tokens,
          temperature=args.temperature, model_string=args.model_string, task=args.task)
    print(fudge.generate())
