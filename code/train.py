import argparse
import os

from datasets import load_from_disk
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import cuda
from collections import defaultdict
import json
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

def compute_TD(model, device, data_loader, TD):
    for _,data in enumerate(tqdm(train_loader)):
        ids = torch.stack(data['ids']).to(device, dtype = torch.long).T
        mask = torch.stack(data['masks']).to(device, dtype = torch.long).T
        targets = data['label'].to(device, dtype = torch.long)
        model.eval()
        with torch.no_grad():
            outputs = model(ids, mask).logits
        model.train()
        outputs = F.softmax(outputs, dim = 1)
        for i in range(len(data['id'])):
            TD[data['id'][i].item()].append(outputs[i,targets[i]].item())

def train(model, device, epochs, loss_function, optimizer, train_loader, model_path, TD_path = None):
    TD = defaultdict(list)
    model.train()
    for e in range(epochs):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for _,data in enumerate(tqdm(train_loader)):
            ids = torch.stack(data['ids']).to(device, dtype = torch.long).T
            mask = torch.stack(data['masks']).to(device, dtype = torch.long).T
            targets = data['label'].to(device, dtype = torch.long)
            outputs = model(ids, mask).logits
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)

            n_correct += (big_idx==targets).sum().item()
            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

        loss_step = tr_loss/nb_tr_steps
        accu_step = (n_correct*100)/nb_tr_examples
        print(f"Training Loss for epoch {e}: {loss_step}")
        print(f"Training Accuracy for epoch {e}: {accu_step}")
        if TD_path is not None:
            compute_TD(model, device, train_loader, TD)

    model.save_pretrained(model_path)
    if TD_path is not None:
        with open(TD_path, 'w') as outfile:
            json.dump(TD, outfile)


def main():
    #argument for training
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="roberta_base", help="The base model used for training.")
    parser.add_argument("--input_path", type=str, default="data", help="The directory of the input data.")
    parser.add_argument("--output_path", type=str, default="model", help="The directory to output the model")
    parser.add_argument("--TD_training", action="store_true", default=False, help='help to define the name of the input data and output model.')
    parser.add_argument("--num_epochs", type=int, default=4, help="The base model used for training.")
    parser.add_argument("--lr", type=float, default=1e-6, help="The directory of the input data.")
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument("--TD_path", type=str, default=None, help="The directory to output the model")


    args = parser.parse_args()

    base_model = args.base_model
    input_path = args.input_path
    output_path = args.output_path
    TD_path = args.TD_path

    device = 'cuda' if cuda.is_available() else 'cpu'

    for source in ['open_qa', 'reddit_eli5', 'finance', 'medicine', 'wiki_csai', 'all']:
        torch.cuda.empty_cache()
        print(f'training for {source}')
        if source == 'all':
            if args.TD_training:
                train_dataset = load_from_disk(os.path.join(input_path, "HC3_selected_all_train"))
            else:
                train_dataset = load_from_disk(os.path.join(input_path, "HC3_processed_all_train"))
        else:
            if args.TD_training:
                dataset = load_from_disk(os.path.join(input_path, f'HC3_selected_{source}'))
            else:
                dataset = load_from_disk(os.path.join(input_path, f'HC3_processed_{source}'))
            train_dataset = dataset['train']
        model_checkpoint = base_model
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2).to(device)
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.lr)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        if TD_training:
            model_path = os.path.join(output_path, f'model_HC3_{source}_selected')
        else:
            model_path = os.path.join(output_path, f'model_HC3_{source}')
        if TD_path is not NOne:
            TD_path = os.path.join(TD_path, f'TD_HC3_{source}')
        train(model, device, args.num_epochs, loss_function, optimizer, train_loader, model_path, TD_path = TD_path)

if __name__ == '__main__':
    main()
    