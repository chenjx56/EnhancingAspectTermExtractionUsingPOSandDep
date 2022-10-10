# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import codecs
import os
import random
import time
import sys
sys.path.append('../')
sys.path.append("../../")

import numpy as np
import csv
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
SPIECE_UNDERLINE = u'▁'


from transformerss.modeling_bert import BertForTextClassification
# from transformerss.tokenization_bert import BertTokenizer
from transformers import BertTokenizerFast
from transformerss.configuration_bert import BertConfig
from transformerss.optimization import AdamW, WarmupLinearSchedule, Warmup
from utils import convert_examples_to_features_discriminate, processors

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logger = logging.getLogger(__name__)
torch.set_num_threads(12)
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    tb_writer = SummaryWriter(args.tensorboard_dir)
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    bert_param = [p for p in list(model.named_parameters()) if 'bert.' in p[0]]
    param_optimizer = [p for p in list(model.named_parameters()) if 'bert.' not in p[0]]
    # bert_param = [p for p in list(model.named_parameters())]
    # print([n for n, p in param_optimizer])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'lr': args.learning_rate * 10, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'lr': args.learning_rate * 10, 'weight_decay': 0.0},
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", num_train_optimization_steps)

    model.zero_grad()
    model.train()
    set_seed(args)
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs, masks, segments, labels = batch
            loss = model(inputs, token_type_ids=segments, attention_mask=masks, labels=labels, mode="train")

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if global_step % args.logging_global_step == 0:
                logger.info("Epoch:{}, Global Step:{}/{}, Loss:{:.5f}".format(epoch, 
                                                                              global_step, num_train_optimization_steps,
                                                                              loss.item()))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                
                global_step += 1
                if args.evaluate_during_training and global_step % args.eval_logging_steps == 0: 
                    model.eval()
                    evaluate(args, model, tokenizer, dataname="test")
                    model.train()
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', loss.item(), global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            break
        torch.cuda.empty_cache()
    tb_writer.close()


def evaluate(args, model, tokenizer, prefix="", dataname="dev"):
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, dataname=dataname)
    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    out_preds, out_labes = [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            predicts = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids, mode=dataname)

            out_preds.append(predicts.detach().cpu().numpy())
            out_labes.append(label_ids.detach().cpu().numpy())
        torch.cuda.empty_cache()
    
    y_true = np.concatenate(tuple(out_labes), axis=0)
    y_pred = np.concatenate(tuple(out_preds), axis=0)
    if args.do_train:
        logger.info("accuracy: {:.4}; precision:{:.4}; recall:{:.4}; f1:{:.4}".format(accuracy_score(y_true, y_pred), 
                                                                                      precision_score(y_true, y_pred, average='macro'),
                                                                                      recall_score(y_true, y_pred, average='macro'),
                                                                                      f1_score(y_true, y_pred, average='macro')))
    else:
        with open(os.path.join(args.output_dir, 'discriResult.txt'), 'w') as f:
            for p in y_pred:
                f.write(str(p) + '\n')


def load_and_cache_examples(args, tokenizer, evaluate=False, dataname="train"):
    processor = processors[args.task_name]()
    max_seq_length = args.train_max_seq_length if dataname=="train" else args.eval_max_seq_length
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(dataname,
                                        list(filter(None, args.bert_model.split('/'))).pop(), str(max_seq_length)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file {}".format(cached_features_file))
        with open(cached_features_file, "rb") as reader:
            features = pickle.load(reader)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if dataname == "train" and not evaluate:
            examples = processor.get_train_examples(args.data_dir, args.trainList)
        elif dataname == "dev" and evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif dataname == "test" and evaluate:
            examples = processor.get_test_examples(args.data_dir, args.eval_directory_name)
            args.test_examples = examples
        else:
            raise ValueError("(evaluate and dataname) parameters error !")
        features = convert_examples_to_features_discriminate(examples, args.label2id, max_seq_length, tokenizer, 
                                                              cls_token_at_end = bool('xlnet' in args.bert_model),
                                                              pad_on_left = False,
                                                              cls_token = tokenizer.cls_token,
                                                              sep_token = tokenizer.sep_token,
                                                              pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                              cls_token_segment_id=2 if 'xlnet' in args.bert_model else 0,
                                                              pad_token_segment_id=4 if 'xlnet' in args.bert_model else 0,
                                                              pad_token_label_id=args.pad_token_label_id,
                                                              output_dir=args.output_dir)

    all_inputs = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segments = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_inputs, all_masks, all_segments, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--tensorboard_dir", default="", type=str, help="Where do you want to store the tensorboard")
    parser.add_argument("--eval_results_dir", default="eval_results.txt", type=str,
                        help="Where do you want to store the eval results")
    parser.add_argument("--train_max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--eval_max_seq_length", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--max_grad_norm", default=10.0, type=float, help="Max gradient norm.")
    parser.add_argument("--schedule", default="WarmupLinearSchedule", type=str,
                        help="Can be `'WarmupLinearSchedule'`, `'warmup_constant'`, `'warmup_cosine'` , `'none'`,"
                             " `None`, 'warmup_cosine_warmRestart' or a `warmup_cosine_hardRestart`")
    parser.add_argument("--no_cuda", action='store_true', help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--pretrained_vocab', type=str, default='', help="to load pretrain vocab (txt file)")
    parser.add_argument('--pretrained_config', type=str, default='', help="to load config (config file)")
    parser.add_argument('--pretrained_params', type=str, default='', help='to load pretraining model params')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--pad_token_label_id", default=-1, type=int, help="id of pad token .")
    parser.add_argument('--logging_global_step', type=int, default=100, help="Log every X updates steps.")
    parser.add_argument('--eval_logging_steps', type=int, default=300, help="Log every X evalution steps.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument('--save_steps', type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--eval_directory_name', type=str, default="test")
    parser.add_argument('--trainList', type=str, default="train")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory {} exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if not args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)
    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.label2id = processor.get_labels(data_dir=args.data_dir)
    logger.info("LABEL : {}".format(args.label2id))
    args.num_labels = len(args.label2id)

    tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_params)
    args.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    model = BertForTextClassification.from_pretrained(args.pretrained_params, num_labels=args.num_labels, device=args.device)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, dataname="train")
        train(args, train_dataset, model, tokenizer)

        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), output_model_file)
        tokenizer.save_vocabulary(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))

    if args.do_eval:
        evaluate(args, model, tokenizer, dataname="test")


if __name__ == "__main__":
    main()
