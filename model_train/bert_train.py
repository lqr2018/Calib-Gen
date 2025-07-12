import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse


class MovieReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = np.mean(losses)

    return avg_loss, accuracy


def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    avg_loss = np.mean(losses)

    return avg_loss, accuracy

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits
    _, prediction = torch.max(logits, dim=1)

    return prediction.item()

def load_imdb_data(train_path: str, test_path: str):
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    return train_df, test_df

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    if args.train:
        print("Training model...")
        model = BertForSequenceClassification.from_pretrained(
            './bert-base-multilingual-cased',
            num_labels=2
        ).to(device)

        train_path = "./data/imdb/plain_text/train-00000-of-00001.parquet"
        test_path  = "./data/imdb/plain_text/test-00000-of-00001.parquet"
        train_df, val_df = load_imdb_data(train_path, test_path)

        train_dataset = MovieReviewDataset(
            texts=train_df['text'].values,
            labels=train_df['label'].values,
            tokenizer=tokenizer,
            max_length=args.max_length
        )

        val_dataset = MovieReviewDataset(
            texts=val_df['text'].values,
            labels=val_df['label'].values,
            tokenizer=tokenizer,
            max_length=args.max_length
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        optimizer = AdamW(model.parameters(), lr=2e-6, correct_bias=False)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for epoch in range(args.epochs):
            print(f'Epoch {epoch + 1}/{args.epochs}')
            print('-' * 10)

            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, device, scheduler
            )
            print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}')

            val_loss, val_acc = eval_model(model, val_loader, device)
            print(f'Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}\n')

        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        print(f"Model saved to {args.model_dir}")

    elif args.predict:
        print("Prediction mode...")
        try:
            model = BertForSequenceClassification.from_pretrained(args.model_dir).to(device)
            sentiment = predict_sentiment(args.predict, model, tokenizer, device, args.max_length)
            print(f'Movie Review: "{args.predict}"')
            print(f'Sentiment: {sentiment}')
        except:
            print("Error: Model not found. Please train the model first with --train")
    else:
        print("Please specify either --train or --predict <text>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Text to predict sentiment")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--model_dir", type=str, default="./bert-base-multilingual-cased/bert_movie_review_sentiment",
                        help="Directory to save/load model")
    args = parser.parse_args()
    main(args)

# model train
# python scripts/model_train/bert_train.py --train --epochs 10 --batch_size 64
