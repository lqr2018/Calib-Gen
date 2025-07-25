import argparse
import uuid
import time

import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

CACHE_TIME_LIMIT = 60 * 10  # ten minutes
CACHE_SIZE_LIMIT = 100

cache = {}
base_model = None
tokenizer = None
bos_token = None
device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'


app = Flask(__name__, template_folder='service')


@app.route("/logits/base_model", methods=["POST"])
def base_model_logits():
    with torch.no_grad():
        print("Received request")
        global cache
        global device
        global base_model
        global bos_token
        global tokenizer
        try:
            prompt = request.json['prompt']  # should be a list of lists of ints
            assert all([type(p) == list for p in prompt])
            assert all([all([type(t) == int for t in p]) for p in prompt])
            assert all([len(p) == len(prompt[0]) for p in prompt])  # same length
            if len(prompt[0]) == 0 and bos_token is not None:
                prompt = [[bos_token] for _ in prompt]
            cache_id = request.json['cache_id'] if 'cache_id' in request.json else None

            past = cache[cache_id][1] if cache_id is not None else None
            input_ids = torch.LongTensor(prompt).to(device)
            length = input_ids.shape[1]
            if past is not None:
                input_ids = input_ids[:, -1:]
            output = base_model(input_ids, past_key_values=past, attention_mask=torch.ones(
                1, length).to(device).long() if past is not None else None)
            if cache_id is None:
                cache_id = str(uuid.uuid4())
            cache[cache_id] = (time.time(), output.past_key_values)
            clean_cache()
            logits = output.logits[:, -1].cpu().tolist()
            logits[0][2] = -1e8  # don't allow end of sentence token
            logits[0][13] = -1e8  # don't allow newlines
            logits[0][1037] = -1e8  # don't allow "|"

            print("Completed request")
            return jsonify({'cache_id': cache_id, 'logits': logits})
        except Exception as e:
            print(f"server error: {e}")
            return jsonify({'error': str(e)})

def clean_cache():
    global cache
    keys = list(cache.keys())
    for key in keys:
        if time.time() - cache[key][0] > CACHE_TIME_LIMIT:
            del cache[key]
    keys = sorted(list(cache.keys()), key=lambda x: cache[x][0], reverse=True)
    keys_to_keep = keys[:CACHE_SIZE_LIMIT]
    for key in keys:
        if key not in keys_to_keep:
            del cache[key]


def load_model(args):
    global base_model
    global tokenizer
    global device
    global bos_token
    base_model_string = args.base_model_string
    if base_model is None:
        tokenizer = AutoTokenizer.from_pretrained(base_model_string, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_string, torch_dtype=torch.float16, trust_remote_code=True).to(device)
        bos_token = args.bos_token_id
        base_model.eval()
        print(f"base model loaded: {base_model_string}")
    return base_model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_string', type=str,
                        default="THUDM/chatglm3-6b")
    parser.add_argument('--bos_token_id', type=int, default=None)
    parser.add_argument('--port', default=9741, type=int,
                        help='port to run the service on')
    args = parser.parse_args()

    load_model(args)

    app.run(host="0.0.0.0", port=args.port, threaded=True)
    # app.run(host="0.0.0.0", port=args.port, threaded=False)
