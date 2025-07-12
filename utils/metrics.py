import sys
import time
import json
import numpy as np
import openai
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer
from .constants import OPENAI_API_KEY, PERSPECTIVE_API_KEY, PERSPECTIVE_API_ATTRIBUTES, EOT_TOKEN
from .utils import unpack_scores
from model_train.bert_train import predict_sentiment

openai.api_key = OPENAI_API_KEY

def get_perspective_api_scores(content, display=False, retries=3, delay=2):
    # 配置日志
    logging.basicConfig(
        filename='perspective_api.log',
        level=logging.ERROR,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )
    for attempt in range(retries):
        try:
            analyze_request = {
                'comment': {'text': content},
                'requestedAttributes': dict([(attribute, {}) for attribute in PERSPECTIVE_API_ATTRIBUTES]),
                'languages': ["en"]
            }

            # print("content_length: ", len(content))

            # 执行 API 请求
            response = client.comments().analyze(body=analyze_request).execute()

            # 解析结果
            summary_scores, span_scores = unpack_scores(response)
            return summary_scores, span_scores

        except HttpError as e:
            logging.error(f"HTTP error occurred: {e}")
            if display:
                print(f"HTTP error: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                print("thread failed!!!")
                return None, None
        except ValueError as e:
            logging.error(f"Invalid response format: {e}")
            if display:
                print(f"Response format error: {e}")
            print("thread failed!!!")
            return None, None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            if display:
                print(f"An unexpected error occurred: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                print("thread failed!!!")
                return None, None


def perplexity(sentences, device='cuda', model_name='gpt2-chinese-cluecorpussmall'):
    ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ppl_model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
    ppl_model.eval()

    with torch.no_grad():
        ppl = []
        sos_token = ppl_tokenizer.decode([0])
        for sentence in tqdm(sentences, total=len(sentences), file=sys.stdout):
            full_tensor_input = ppl_tokenizer.encode(
                sos_token + sentence.replace(EOT_TOKEN, ' ').strip(), return_tensors='pt').to(device)[:512]
            full_loss = ppl_model(full_tensor_input, labels=full_tensor_input)[0].mean()
            ppl.append(torch.exp(full_loss).flatten().cpu().item())

    return ppl, np.mean(ppl), np.std(ppl)

def cosine_similarity(sentence_pairs, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    from sentence_transformers import SentenceTransformer

    if not sentence_pairs or not all(isinstance(pair, (list, tuple)) and len(pair) == 2 for pair in sentence_pairs):
        raise ValueError("sentence_pairs must be a non-empty list of tuples/lists with exactly two elements.")

    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

    sentences_a, sentences_b = zip(*sentence_pairs)

    embeddings_a = model.encode(sentences_a, batch_size=8, device=device)
    embeddings_b = model.encode(sentences_b, batch_size=8, device=device)

    dot_product = np.einsum("ij, ij->i", embeddings_a, embeddings_b)
    norm_a = np.linalg.norm(embeddings_a, axis=1)
    norm_b = np.linalg.norm(embeddings_b, axis=1)

    similarities = dot_product / (norm_a * norm_b)
    similarities = similarities.tolist()

    return similarities

def sentiment_judge(sentences, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    model_dir = "./bert-base-multilingual-cased/bert_movie_review_sentiment"
    model = BertForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    results = []
    for content in sentences:
        result = predict_sentiment(content, model, tokenizer, device)
        results.append(result)

    return results
