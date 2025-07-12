import sys

PAD_TOKEN = '[PAD]'
EOT_TOKEN = '<|endoftext|>'
SEP = 50256  # just use the weird eot token

fudge_models = ['facebook/opt-125m']
# HIDDEN_DIM = 1024

SAVE_PATH = "./methods/fudge/ckpt"
TOXICITY_DATA_PATH = "./data/jigsaw_toxic/toxicity_prompts/toxic_training_prompts.jsonl"
SENTIMENT_DATA_PATH_POS = "./data/sentiment_prompts/train_pos.jsonl"
SENTIMENT_DATA_PATH_NEG = "./data/sentiment_prompts/train_neg.jsonl"
