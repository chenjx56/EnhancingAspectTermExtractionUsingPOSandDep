import torch
import os
import pickle
import numpy as np


class ATEExample(object):
    def __init__(self, context, labels):
        self.context = context
        self.labels = labels

class ATEDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class ATEProcessor():
    def read_example(self, file_name):
        with open(file_name, 'r', encoding="utf8") as fin:
            examples = []
            for line in fin:
                context, label_str, _ = line.strip().split("\t")
                labels = label_str.split(" ")
                assert len(context.split(" ")) == len(labels)
                example = ATEExample(context, labels)
                examples.append(example)
        return examples

    def convert_examples_to_features(self, tokenizer, examples, max_lens):
        
        encodings = {}
        encodings["input_ids"] = []
        encodings["attention_mask"] = []
        encodings["token_type_ids"] = []
        labels = []

        label_to_id = {
            "O": 0,
            "B-AS": 1,
            "I-AS": 2,
        }
        for example in examples:
            tokenized_inputs = tokenizer(
                example.context.split(" "),
                padding='max_length',
                max_length=max_lens, # 102
                is_split_into_words=True,
                )
            # align word level labels to tokens  
            label = example.labels
            
            word_ids = tokenized_inputs.word_ids() # list of word index of each subword
            label_ids = []
            last_word_idx = -1
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx == last_word_idx and label_to_id[label[last_word_idx]] == 1:
                    label_ids.append(2)
                else:
                    label_ids.append(label_to_id[label[word_idx]])
                last_word_idx = word_idx
            # label_ids[0] = 0 # adapt to the evaluator
            
            encodings["input_ids"].append(tokenized_inputs["input_ids"])
            encodings["attention_mask"].append(tokenized_inputs["attention_mask"])
            encodings["token_type_ids"].append(tokenized_inputs["token_type_ids"])
            labels.append(label_ids)
        dataset = ATEDataset(encodings, labels)
        return dataset

def load_dataset(args, tokenizer, mode):
    processor = ATEProcessor()
    file_path = args.data_path
    file_path = file_path.replace("file", mode)
    examples = processor.read_example(file_path)
    return processor.convert_examples_to_features(tokenizer, examples, args.max_seq_length)