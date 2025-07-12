########################################
# General Generation Parameters
########################################

methods = ["preadd",
           "raw_prompting",
           "neg_prompting",
           "fudge",
           "calib"]

tasks = ["toxicity", "sentiment"]

small_models = ["facebook/opt-125m"]
big_models = ["THUDM/chatglm3-6b"]
models = small_models + big_models

toxicity_prompts_dir = "./data/jigsaw_toxic/toxicity_prompts"
sentiment_prompts_dir = "./data/sentiment_prompts"
prompt_datasets = {"toxicity_random": f"{toxicity_prompts_dir}/random_prompts.jsonl",
                   "toxicity_toxic": f"{toxicity_prompts_dir}/toxic_prompts.jsonl",
                   # "toxicity_random_small": f"{toxicity_prompts_dir}/random_prompts_small.jsonl",
                   # "toxicity_toxic_small": f"{toxicity_prompts_dir}/toxic_prompts_small.jsonl",

                   "toxicity_random_cao": f"{toxicity_prompts_dir}/random_prompts_cao.jsonl",
                   "toxicity_toxic_cao": f"{toxicity_prompts_dir}/toxic_prompts_cao.jsonl",
                   # "toxicity_random_small_cao": f"{toxicity_prompts_dir}/random_prompts_small_cao.jsonl",
                   # "toxicity_toxic_small_cao": f"{toxicity_prompts_dir}/toxic_prompts_small_cao.jsonl",

                   "sentiment_positive": f"{sentiment_prompts_dir}/positive_dataset.jsonl",
                   "sentiment_negative": f"{sentiment_prompts_dir}/negative_dataset.jsonl",

                   "sentiment_positive_cao": f"{sentiment_prompts_dir}/positive_dataset_cao.jsonl",
                   "sentiment_negative_cao": f"{sentiment_prompts_dir}/negative_dataset_cao.jsonl",
                   }
toxicity_prompts = [p for p in prompt_datasets.keys() if "toxicity" in p]
sentiment_prompts = [p for p in prompt_datasets.keys() if "sentiment" in p]
all_prompts = toxicity_prompts + sentiment_prompts

########################################
# Metrics Parameters
########################################

PAD_TOKEN = '[PAD]'
EOT_TOKEN = '<|endoftext|>'

########################################
# Perspective API

# All attributes can be found here:
# https://github.com/conversationai/perspectiveapi/blob/master/2-api/models.md
########################################

# Input your PerspectiveAPI key here:
PERSPECTIVE_API_KEY = "xxx"

PERSPECTIVE_API_ATTRIBUTES = (
    'TOXICITY',
    'SEVERE_TOXICITY',
    'IDENTITY_ATTACK',
    'INSULT',
    'THREAT',
    'PROFANITY',
)
PERSPECTIVE_API_ATTRIBUTES_LOWER = tuple(a.lower() for a in PERSPECTIVE_API_ATTRIBUTES)

########################################
# OpenAI
########################################

# Input your OpenAI api key here
OPENAI_API_KEY = "xxx"

