import argparse
import os

import json
import numpy as np
from datasets import load_dataset, load_from_disk, concatenate_datasets

def main():
    #argument for testing
    parser = argparse.ArgumentParser()
    parser.add_argument("--TD_path", type=str, default="TD", help="The directory of the training dynamic")
    parser.add_argument("--output_path", type=str, default="data", help="The directory to output the test result.")
    parser.add_argument("--data_path", type=str, default="data", help="The directory of the data_set")
    args = parser.parse_args()

    TD_path = args.TD_path
    output_path = args.output_path
    data_path = args.data_path


    for source in ['open_qa', 'reddit_eli5', 'finance', 'medicine', 'wiki_csai', 'all']:
        TD_path = os.path.join(TD_path, f'TD_HC3_{source}.json')
        if source == 'all':
            dataset = load_from_disk(os.path.join(input_path, f'HC3_processed_all_train'))
            output_name = 'HC3_selected_all_train'
        else:
            dataset = load_from_disk(os.path.join(input_path, f'HC3_processed_{source}'))
            output_name = f'HC3_selected_{source}'
        with open(TD_path) as infile:
            data = json.load(infile)
        id = list(zip(*data.items()))[0]
        id = np.asarray(id, dtype=int)
        obs = list(zip(*data.items()))[1]
        confidence = np.mean(obs, axis=1)
        variability = np.std(obs, axis=1)

        confidence_q = np.quantile(confidence, 0.01)
        confidence_q_low = np.quantile(confidence, 0.005)
        variability_q = np.quantile(variability, 0.99)
        drop_conf_low = id[np.where(confidence <= confidence_q_low)]
        drop_conf = id[np.where(confidence <= confidence_q)]
        drop_var = id[np.where(variability >= variability_q)]

        drop = np.union1d(drop_conf_low, np.intersect1d(drop_conf, drop_var))
        select_dataset = dataset.filter(lambda x : x['id'] not in drop)

        select_dataset.save_to_disk(os.path.join(output_path, output_name))