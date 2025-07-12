import math
import time
from typing import Dict, Optional, Union, Sequence

from scipy.special import expm1
import openai
from transformers import AutoTokenizer
import torch
import requests


server_tokenizer = None
aux_server_tokenizer = None

def get_tokenizer(model_string):
    global server_tokenizer
    if server_tokenizer is None:
        server_tokenizer = AutoTokenizer.from_pretrained(model_string, trust_remote_code=True)
    return server_tokenizer

def get_aux_tokenizer(model_string):
    global aux_server_tokenizer
    if aux_server_tokenizer is None:
        aux_server_tokenizer = AutoTokenizer.from_pretrained(model_string, trust_remote_code=True)
    return aux_server_tokenizer


def get_next_logprobs(prompt, model_string, cache_id=None, include_indices=[]):
    # prompt should be a list of tokens
    assert type(prompt) == list
    if len(prompt) > 0:
        assert type(prompt[0]) == int
    return server_next_logprobs(prompt, model_string, cache_id=cache_id)

def server_next_logprobs(prompt, model_string, cache_id=None, url='http://localhost:9741/logits/base_model'):
    # prompt is just a list of ints, just doing 1 at a time for now
    data = {'prompt': [prompt], 'cache_id': cache_id}
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            r = requests.post(url, json=data)
            r.raise_for_status()  # Check if the request was successful (status code 200)
            response = r.json()
            return {'logits': response['logits'][0],
                    'cache_id': response['cache_id']}
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                break
            time.sleep(2)

def get_masked_logprobs(prompt, model_string, masked_position, cache_id=None, include_indices=[]):
    assert type(prompt) == list
    if len(prompt) > 0:
        assert type(prompt[0]) == int
    return server_masked_logprobs(prompt, model_string, masked_position, cache_id=cache_id)

def server_masked_logprobs(prompt, model_string, masked_position, cache_id=None, url='http://localhost:9741/logits/mask_filling'):
    # prompt is just a list of ints, just doing 1 at a time for now
    data = {'prompt': [prompt],
            'cache_id': cache_id,
            'mask_position': masked_position}
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            r = requests.post(url, json=data)
            r.raise_for_status()  # Check if the request was successful (status code 200)
            response = r.json()
            return {'logits': response['logits'][0],
                    'cache_id': response['cache_id']}
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                return {'logits': response['error'],
                        'cache_id': response['error']}
            time.sleep(2)
