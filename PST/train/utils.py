# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" BERT classification fine-tuning: utilities to work with GLUE tasks """

from __future__ import absolute_import, division, print_function

import csv
import logging
import os
from collections import defaultdict
import sys
sys.path.append("../")
sys.path.append("../../")
import numpy as np
import pickle
import json
from random import shuffle
import re


logger = logging.getLogger(__name__)

import spacy
from spacy.tokens import Doc
from transformers import BertTokenizerFast

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

class InputExample(object):
    def __init__(self, guid, text_a, label_a, text_b=None, label_b=None, pos=None, graph=None):
        self.guid = guid
        self.text_a = text_a
        self.label_a = label_a
        self.text_b = text_b
        self.label_b = label_b
        if pos is not None and type(pos) == str:
            self.pos = pos.split()
        else:
            self.pos = pos
        self.graph = graph


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, tokens_len=None, pos=None, graph=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.tokens_len = tokens_len
        self.pos = pos
        self.graph = graph


class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        raise NotImplementedError()
    def get_labels(self):
        raise NotImplementedError()
    @classmethod
    def _read_txt(cls, input_path, quotechar=None, start_index=None, end_index=None):
        lines = []
        if start_index is None and end_index is None:
            with open(os.path.join(input_path, 'sentence.txt'), 'r', encoding="utf-8-sig") as data_sent, \
                open(os.path.join(input_path, 'target.txt'), 'r', encoding="utf-8-sig") as data_tagt, \
                open(os.path.join(input_path, 'pos.txt'), 'r', encoding="utf-8-sig") as data_pos, \
                open(os.path.join(input_path, 'sentence.txt.graph'), 'rb') as data_graph:
                idx2graph = pickle.load(data_graph)
                logger.info("load file {}/{}".format(os.path.join(input_path, 'sentence.txt'), os.path.join(input_path, 'target.txt')))
                for sent, tagt, pos in zip(data_sent.readlines(), data_tagt.readlines(), data_pos.readlines()):
                    sent = sent.strip()
                    tagt = tagt.strip()
                    pos = pos.strip()
                    graph = idx2graph[sent]
                    lines.append((sent, tagt, pos, graph))

        elif start_index is not None and end_index is not None:
            nlp = spacy.load('en_core_web_sm')
            nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
            tokenizer = BertTokenizerFast.from_pretrained("/data/jxchen/PLM/bert-base-uncased")
            # amazon/cell_phones_and_accessory_128.txt
            if "lap" in input_path:
                external_data = '../../amazon/cell_phones_and_accessory_sort_4w_128.txt'
            if "res" in input_path:
                external_data = '../../yelp/sentence.txt'

            with open(os.path.join(input_path, external_data), 'r', encoding="utf-8-sig") as data_sent:
                logger.info("load file {}/{}-{}".format(os.path.join(input_path, external_data), start_index, end_index))
                for k, sent in enumerate(data_sent.readlines()):
                    if start_index <= k < end_index:
                        sent = sent.strip()
                        tagt = ' '.join(['0'] * len(sent.split()))
                        pos, graph = getPosAndGraph(nlp, tokenizer, sent)
                        lines.append((sent, tagt, pos, graph))
        
        else:
            raise ValueError("start_index:{}, end_index:{}".format(start_index, end_index))
        return lines

    @classmethod
    def _read_csv(cls, input_path, file='discriminate.csv', quotechar=None):
        lines = []
        with open(os.path.join(input_path, file)) as csvfile:
            logger.info("load file {}".format(os.path.join(input_path, file)))
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(spamreader)
            # ["text", "aspect", "label", "error type"]
            for row in spamreader:
                lines.append((row[0], row[1], row[2]))

        return lines

        '''
        mode = input_path.split('/')[-1]
        with open(os.path.join(input_path.replace(mode, 'train'), 'sentence.txt'), 'r', encoding="utf-8-sig") as data_sent, \
            open(os.path.join(input_path.replace(mode, 'train'), 'target.txt'), 'r', encoding="utf-8-sig") as data_tagt:
                for sent, tagt in zip(data_sent.readlines(), data_tagt.readlines()):
                    trainLines.append((sent.strip(), tagt.strip()))

        mapDict = {'train':'bleu_text_train_train.txt', 'test':'bleu_text_test_train.txt'}
        with open(os.path.join(input_path, mapDict[mode]), 'r', encoding="utf-8-sig") as data_index:
            for k, (line, indx) in enumerate(zip(lines, data_index.readlines())):
                for topIdx in indx.strip().split()[:K]:
                    if topIdx == str(k) and mode == 'train':
                        continue
                    tmp_lines.append(line + trainLines[int(topIdx)])
        return tmp_lines
        '''

def getPosAndGraph(nlp, tokenizer, text, align_pos=False):
    tokens = nlp(text)
    tokenized = tokenizer(text.split(" "), is_split_into_words=True, add_special_tokens=False)
    word_ids = tokenized.word_ids()
    words = text.split()

    matrix1 = np.zeros((len(word_ids), len(word_ids))).astype('float32')
    
    assert len(words) == len(list(tokens))
    assert (len(tokens) - 1) == max(word_ids)

    pos = [token.tag_ for token in tokens]
    assert (len(pos) - 1) == max(word_ids)
    if align_pos:
        pos = [pos[idx] for idx in word_ids]
    
    for i, idx in enumerate(word_ids):
        matrix1[i][i] = 1
        for j, id in enumerate(word_ids):
            if tokens[id] in tokens[idx].children or word_ids[j] == word_ids[i]:
                matrix1[i][j] = 1
                matrix1[j][i] = 1

    return pos, matrix1   

class AspectProcessor(DataProcessor):
    def get_train_examples(self, data_dir, trainList):
        lines = []
        for t in trainList.split(' '):
            lines += self._read_txt(os.path.join(data_dir, t))
        return self._create_examples(lines, "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "validation.json")), "dev")
    def get_test_examples(self, data_dir, start_index, end_index):
        return self._create_examples(self._read_txt(os.path.join(data_dir, "test"), start_index=start_index, end_index=end_index), "test")
    def get_labels(self, data_dir=None):
        return {'O':0, 'B-AS':1, 'I-AS':2}
        # return {"O":0, "B-address":1, "I-address":2, "B-book":3, "I-book":4, "B-company":5, "I-company":6, "B-game":7, "I-game":8, "B-government":9, "I-government":10, "B-movie":11, "I-movie":12, "B-name":13, "I-name":14, "B-organization":15, "I-organization":16, "B-position":17, "I-position":18, "B-scene":19, "I-scene":20}
    def _create_examples(self, lines, set_type):
        examples = []
        id2label = {str(v):k for k, v in self.get_labels().items()}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            sent, tagt, pos, graph = line
            tagt = '  '.join([id2label[t] for t in tagt.split()])
            examples.append(InputExample(guid=guid, text_a=sent, label_a=tagt, text_b=None, label_b=None, pos=pos, graph=graph))
        return examples


class DiscriminateProcessor(DataProcessor):
    def get_train_examples(self, data_dir, trainList):
        lines = []
        for t in trainList.split(' '):
            lines += self._read_csv(os.path.join(data_dir, t))
        return self._create_examples(lines, "train")
    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "validation.json")), "dev")
    def get_test_examples(self, data_dir, eval_directory_name):
        file = "discri.csv" if 'train' in eval_directory_name else "discriminate.csv"
        return self._create_examples(self._read_csv(os.path.join(data_dir, eval_directory_name), file), "test")
    def get_labels(self, data_dir=None):
        return {'0':0, '1':1}
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            sent, target, label = line
            examples.append(InputExample(guid=guid, text_a=sent, label_a=label, text_b=target, pos=None, graph=None))
        return examples


def convert_examples_to_features_readcompre(examples, label2id, max_seq_length, tokenizer,
                                            cls_token_at_end=False, pad_on_left=False,
                                            cls_token='[CLS]', sep_token='[SEP]',
                                            pad_token_id=0,
                                            sequence_a_segment_id=0,
                                            sequence_b_segment_id=1,
                                            cls_token_segment_id=1,
                                            pad_token_segment_id=0,
                                            pad_token_label_id=-1,
                                            mask_padding_with_zero=True,
                                            output_dir=None):
    def _reseg_token_label(tokens, labels, poss, tokenizer):
        try:
            assert len(tokens) == len(labels) == len(poss)
        except:
            print(tokens)
            print(labels)
            print(poss)
            raise
        ret_tokens, ret_labels, ret_poss = [], [], []
        for token, label, pos in zip(tokens, labels, poss):
            sub_token = tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            ret_labels.append(label)
            ret_poss.append(pos)
            if len(sub_token) == 1:
                continue
            if label.startswith("B") or label.startswith("I"):
                sub_label = "I-" + label[2:]
                ret_labels.extend([sub_label] * (len(sub_token) - 1))
                ret_poss.extend([pos] * (len(sub_token) - 1))
            elif label.startswith("O"):
                sub_label = label
                ret_labels.extend([sub_label] * (len(sub_token) - 1))
                ret_poss.extend([pos] * (len(sub_token) - 1))
            else:
                raise ValueError
        assert len(ret_tokens) == len(ret_labels) == len(ret_poss)
        return ret_tokens, ret_labels, ret_poss

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = example.text_a.split()
        labels_a = example.label_a.split()
        pos = example.pos
        graph = example.graph

        def inputIdMaskSegment(tmp_tokens, tmp_labels, tmp_pos, graph, tmp_segment_id):
            tokens, labels, pos = _reseg_token_label(tmp_tokens, tmp_labels, tmp_pos, tokenizer)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
                labels = labels[:(max_seq_length - 2)]
                pos = pos[:(max_seq_length - 2)]
                graph = graph[:(max_seq_length - 2)][:(max_seq_length - 2)]

            label_ids = [label2id[label] for label in labels]
            pad_label_id = label2id["O"] if pad_token_label_id == -1 else pad_token_label_id

            pos = ["[" + p.lower() + "]" for p in pos]
            tokens = [cls_token] + tokens + [sep_token]
            pos = [cls_token] + pos + [sep_token]
            graph = np.pad(graph, ((1, 1), (1, 1)), 'constant')
            segment_ids = [tmp_segment_id] * len(tokens)
            label_ids = [pad_label_id] + label_ids + [pad_label_id]

            tokens_len = len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            pos = tokenizer.convert_tokens_to_ids(pos)
            assert 100 not in pos
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += ([pad_token_id] * padding_length)
            pos += ([pad_token_id] * padding_length)
            graph = np.pad(graph, ((0, padding_length), (0, padding_length)), 'constant')
            input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids += ([tmp_segment_id] * padding_length)
            label_ids += ([pad_label_id] * padding_length)
            assert len(input_ids) == len(input_mask) == len(segment_ids) == len(label_ids) == max_seq_length == len(pos) == len(graph)
            return input_ids, input_mask, segment_ids, label_ids, tokens_len, pos, graph

        input_ids, input_mask, segment_ids, label_ids, tokens_len, pos, graph = inputIdMaskSegment(tokens_a, labels_a, pos, graph, sequence_a_segment_id)

        '''
        if example.text_b is not None:
            tokens_b = example.text_b.split()
            labels_b = example.label_b.split()
            tokenb_input_ids, tokenb_mask, tokenb_segment_ids, tokenb_label_ids, _ = inputIdMaskSegment(tokens_b, labels_b, sequence_b_segment_id)
            
            input_ids += tokenb_input_ids
            input_mask += tokenb_mask
            segment_ids += tokenb_segment_ids 
            label_ids += tokenb_label_ids
        
        query_tokens = tokenizer.tokenize("what are aspects ?") + [sep_token]
        query_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        input_ids += query_ids
        input_mask += [1] * len(query_ids)
        segment_ids += [1] * len(query_ids)
        assert len(input_ids) == len(input_mask) == len(segment_ids)
        '''

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens_a]))
            logger.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, 
                                      label_ids=label_ids, tokens_len=tokens_len, pos=pos, graph=graph))
    return features


def convert_examples_to_features_discriminate(examples, label2id, max_seq_length, tokenizer,
                                              cls_token_at_end=False, pad_on_left=False,
                                              cls_token='[CLS]', sep_token='[SEP]',
                                              pad_token_id=0,
                                              sequence_a_segment_id=0,
                                              sequence_b_segment_id=1,
                                              cls_token_segment_id=1,
                                              pad_token_segment_id=0,
                                              pad_token_label_id=-1,
                                              mask_padding_with_zero=True,
                                              output_dir=None):

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokens_a = example.text_a.split()
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens_a = [cls_token] + tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens_a)

        query_tokens = tokenizer.tokenize('Is " {} " an aspect term in the sentence ?'.format(example.text_b)) + [sep_token]
        # query_tokens = tokenizer.tokenize(' " {} " 是 一 个 中 文 实 体 吗 ?'.format(example.text_b)) + [sep_token]
        input_ids = tokenizer.convert_tokens_to_ids(tokens_a + query_tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        segment_ids += [sequence_b_segment_id] * len(query_tokens)

        padding_length = max_seq_length + 20 - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += ([sequence_a_segment_id] * padding_length)

        label_id = label2id[example.label_a]
        assert len(input_ids) == len(input_mask) == len(segment_ids)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens_a + query_tokens]))
            logger.info("label_ids: %s" % str(label_id))

        features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


processors = {
    "aspect": AspectProcessor,
    "discriminate": DiscriminateProcessor
}
