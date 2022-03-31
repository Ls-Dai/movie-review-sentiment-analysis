import argparse
import torch
from torch.utils.data import RandomSampler
import os
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from dataset import Dataset
from model import BertEstimator 


BERT_MODEL = 'bert-base-uncased'
NUM_LABELS = 2  # negative and positive reviews


def train(input_path, epochs=10, output_path="./weights/"):
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)

    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

    dataset = Dataset(tokenizer=tokenizer)
    dataloader = dataset.prepare_dataloader(input_path, sampler=RandomSampler)
    estimator = BertEstimator()
    estimator.train(
        tokenizer=tokenizer,
        dataloader=dataloader,
        model=model,
        epochs=epochs, 
    )
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


def evaluate(input_path, model_path="./weights/"):
    estimator = BertEstimator()
    estimator.load(model_dir=model_path)

    dataset = Dataset(tokenizer=estimator.tokenizer)
    dataloader = dataset.prepare_dataloader(input_path)
    score = estimator.evaluate(dataloader)
    print(score)


def main():

    parser = argparse.ArgumentParser(prog='main')

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--path", default="", type=str)
    parser.add_argument("--epochs", default=10, type=int)

    args = parser.parse_args()

    if args.train:
        if os.path.exists(args.path):
            train(input_path=args.path, epochs=args.epochs)
        else:
            print("Not a valid data path.")
    elif args.evaluate:
        if os.path.exists(args.path):
            evaluate(input_path=args.path)
        else:
            print("Not a valid data path.")


if __name__ == "__main__":
    main()
