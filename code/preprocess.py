# Preprocess HC3 datasets
import argparse
import os

from datasets import load_dataset, load_from_disk, concatenate_datasets

def preprocess_HC3(args):
    output_path = args.output_path
    test_portion = args.test_portion
    base_model = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, padding='max_length', pad_to_max_length=True)
    dataset = load_dataset("Hello-SimpleAI/HC3", 'all')
    dataset = dataset['train']

    # tokenize data and create id for each example
    dataset = dataset.map(lambda x : map_HC3(x, tokenizer), batched=True, remove_columns=dataset.column_names)
    
    # create train and test split for each source
    filterd_dataset = []
    for source in ['reddit_eli5', 'finance', 'medicine', 'open_qa', 'wiki_csai']:
        source_dataset = (map_train_dataset.filter(lambda x: x['source'] == source))
        filterd_dataset.append(source_dataset.train_test_split(test_size=test_portion))
        file_name = f"HC3_processed_{source}"
        filterd_dataset[-1].save_to_disk(os.path.join(output_path, file_name))

    all_dataset_train = concatenate_datasets([d['train'] for d in filterd_dataset])
    all_dataset_train.save_to_disk(os.path.join(output_path, "HC3_processed_all_train"))
    all_dataset_test = concatenate_datasets([d['test'] for d in filterd_dataset])
    all_dataset_test.save_to_disk(os.path.join(output_path, "HC3_processed_all_test"))




def map_HC3(examples, tokenizer):
    kwargs = dict(max_length=512, truncation=True, padding='max_length')

    ids = []
    masks = []
    labels = []
    sources = []
    for question, example, source in zip(examples['question'], examples["human_answers"], examples['source']):
        for sentence in example:
        tok = tokenizer(question, sentence, **kwargs)
        ids += [tok['input_ids']]
        masks += [tok['attention_mask']]
        labels += [0]
        sources += [source]
    for question, example, source in zip(examples['question'],examples["chatgpt_answers"], examples['source']):
        for sentence in example:
        tok = tokenizer(question, sentence, **kwargs)
        ids += [tok['input_ids']]
        masks += [tok['attention_mask']]
        labels += [1]
        sources += [source]
    return {'ids': ids, 'masks': masks, 'label' : labels, 'source': sources}

def main():
    #argument for preprocess
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="data", help="The directory to output the preprocessed data")
    parser.add_argument("--test_portion", type=float, default=0.2, help="The portion of data used as test set.")
    parser.add_argument("--base_model", type=str, default="roberta_base", help="The base model used for training.")
    args = parser.parse_args()

    preprocess_HC3(args)

if __name__ == '__main__':
    main()
    