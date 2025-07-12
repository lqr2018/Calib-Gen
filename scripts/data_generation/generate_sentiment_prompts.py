import pandas as pd
import json
import os
from random import sample

def sample_imdb_data(parquet_path, output_pos="pos_samples.json", output_neg="neg_samples.json", pos_train="train_pos.json", neg_train="train_neg.json", sample_size=1000):
    try:
        df = pd.read_parquet(parquet_path)

        if 'label' not in df.columns:
            raise ValueError("The dataset does not contain a 'label' column. Please verify the data format.")

        pos_samples = df[df['label'] == 1]
        neg_samples = df[df['label'] == 0]

        samples_for_pos_train = df
        samples_for_neg_train = df.assign(label=1-df['label'])

        if len(pos_samples) < sample_size or len(neg_samples) < sample_size:
            raise ValueError(f"Insufficient samples in dataset: positive samples {len(pos_samples)}, negative samples {len(neg_samples)}")

        pos_selected = pos_samples.sample(n=sample_size, random_state=42)
        neg_selected = neg_samples.sample(n=sample_size, random_state=42)

        pos_selected.to_json(output_pos, orient='records', lines=True, force_ascii=False)
        neg_selected.to_json(output_neg, orient='records', lines=True, force_ascii=False)

        samples_for_pos_train.to_json(pos_train, orient='records', lines=True, force_ascii=False)
        samples_for_neg_train.to_json(neg_train, orient='records', lines=True, force_ascii=False)

        print(f"Sampling completed. Positive samples saved to: {output_pos}")
        print(f"Sampling completed. Negative samples saved to: {output_neg}")
        print(f"Number of positive samples: {len(pos_selected)}")
        print(f"Number of negative samples: {len(neg_selected)}")

    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")


if __name__ == "__main__":
    input_parquet = "./data/imdb/plain_text/train-00000-of-00001.parquet"
    positive_output_json = "./data/sentiment_prompts/positive_dataset.jsonl"
    negative_output_json = "./data/sentiment_prompts/negative_dataset.jsonl"
    pos_train_json = "./data/sentiment_prompts/train_pos.jsonl"
    neg_train_json = "./data/sentiment_prompts/train_neg.jsonl"


    if not os.path.exists(input_parquet):
        print(f"Error: Input file {input_parquet} does not exist")
    else:
        sample_imdb_data(input_parquet, positive_output_json, negative_output_json, pos_train_json, neg_train_json)