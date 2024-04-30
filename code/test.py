import argparse
import os

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import cuda
from collections import defaultdict
import json
from torch.utils.data import DataLoader, Dataset
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def test(model,device,test_loader):
    n_correct = 0
    nb_examples = 0
    y_true = []
    y_pred = []
    model.eval()
    for _,data in enumerate(tqdm(test_loader)):
        ids = torch.stack(data['ids']).to(device, dtype = torch.long).T
        mask = torch.stack(data['masks']).to(device, dtype = torch.long).T
        targets = data['label'].to(device, dtype = torch.long)
        with torch.no_grad():
        outputs = model(ids, mask).logits
        big_val, big_idx = torch.max(outputs.data, dim=1)
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(big_idx.cpu().numpy().tolist())

        n_correct += (big_idx==targets).sum().item()
        nb_examples+=targets.size(0)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    #precision = precision_score(y_true, y_pred, average='macro')
    #recall = recall_score(y_true, y_pred, average='macro')
    return acc, f1

def main():
    #argument for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model", help="The directory of the model")
    parser.add_argument("--input_path", type=str, default="data", help="The directory of the input data.")
    parser.add_argument("--output_path", type=str, default="result", help="The directory to output the test result.")
    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")

    args = parser.parse_args()

    model_path = args.model_path
    input_path = args.input_path
    output_path = args.output_path

    device = 'cuda' if cuda.is_available() else 'cpu'

    for source in ['open_qa', 'reddit_eli5', 'finance', 'medicine', 'wiki_csai', 'all']:
        torch.cuda.empty_cache()
        print(f'testing for {source}')
        model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, f'model_HC3_{source}')).to(device)
        output = f'test result for full-training using dataset {source}'
        for test_source in ['finance', 'medicine','open_qa', 'reddit_eli5',  'wiki_csai', 'all']:
            if test_source == 'all':
                test_dataset = load_from_disk(os.path.join(input_path, f'HC3_processed_all_test'))
            else:
                dataset = load_from_disk(os.path.join(input_path, f'HC3_processed_{test_source}'))
                test_dataset = dataset['test']
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
            acc,f1 = test(model,device,test_loader)
            output += f'\n{test_source} accuracy: {acc}'
            output += f'\n{test_source} f1: {f1}'
        print(output)
        with open(os.path.join(output_path, f'test_{model_path}.txt'), 'w') as f:
            f.write(output)

if __name__ == '__main__':
    main()
