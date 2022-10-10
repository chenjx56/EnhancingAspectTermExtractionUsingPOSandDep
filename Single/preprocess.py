import torch
import os
import pickle
import numpy as np


class ATEExample(object):
    def __init__(self, context, labels, pos_tag, dependency_graph):
        self.context = context
        self.labels = labels
        self.pos = pos_tag
        self.dependency_graph = dependency_graph

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
        fgraph = open(file_name + ".graph", "rb")
        idx2graph = pickle.load(fgraph)
        fgraph.close()
        with open(file_name, 'r', encoding="utf8") as fin:
            examples = []
            for line in fin:
                context, label_str, pos_tag = line.strip().split("\t")
                labels = label_str.split(" ")
                pos_tag = pos_tag.split(" ")
                assert len(context.split(" ")) == len(labels)
                assert len(context.split(" ")) == len(pos_tag)
                example = ATEExample(context, labels, pos_tag, idx2graph[context])
                examples.append(example)
        return examples

    def convert_examples_to_features(self, tokenizer, examples, max_lens):
        
        encodings = {}
        encodings["input_ids"] = []
        encodings["attention_mask"] = []
        encodings["token_type_ids"] = []
        encodings["dependency_graph"] = []
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
            pos_tag = example.pos
            pos_tag = [tokenizer("["+pos.lower()+"]", add_special_tokens=False)["input_ids"] for pos in pos_tag]
            for p in pos_tag:
                try:
                    assert len(p) == 1
                except:
                    print(example.pos)
                    print(pos_tag)
                    print(p)
                    raise
            word_ids = tokenized_inputs.word_ids() # list of word index of each subword
            label_ids = []
            pos_ids = []
            last_word_idx = -1
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx == last_word_idx and label_to_id[label[last_word_idx]] == 1:
                    label_ids.append(2)
                    pos_ids.append(pos_tag[word_idx][0])
                else:
                    label_ids.append(label_to_id[label[word_idx]])
                    pos_ids.append(pos_tag[word_idx][0])
                last_word_idx = word_idx
            # label_ids[0] = 0 # adapt to the evaluator
            pos_lens = len(pos_ids)
            pad_pos = (max_lens - 2 - pos_lens) * [0]
            sep_idx = tokenized_inputs["input_ids"].index(102)
            tokenized_inputs["input_ids"][sep_idx] = 0
            tokenized_inputs["attention_mask"][sep_idx] = 0
            tokenized_inputs["attention_mask"][-1] = 1
            tokenized_inputs["input_ids"][-1] = 102
            input_ids = tokenized_inputs["input_ids"] + pos_ids
            input_ids += pad_pos
            tokenized_inputs["token_type_ids"][-1] = 1
            tokenized_inputs["token_type_ids"] += (max_lens-2) * [1]
            tokenized_inputs["attention_mask"] += tokenized_inputs["attention_mask"][1:sep_idx]
            tokenized_inputs["attention_mask"] += pad_pos
            
            d_graph = np.pad(example.dependency_graph, ((1,max_lens-2-len(example.dependency_graph)), (1,max_lens-2-len(example.dependency_graph))), 'constant')
            encodings["dependency_graph"].append(d_graph)

            assert len(input_ids) == len(tokenized_inputs["attention_mask"]) == len(tokenized_inputs["token_type_ids"])
            encodings["input_ids"].append(input_ids)
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