# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at


from __future__ import absolute_import, division, print_function
from turtle import pos
from apex import amp
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


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
SPIECE_UNDERLINE = u'▁'


from transformerss.modeling_bert import BertForSequenceLabeling
# from transformerss.tokenization_bert import BertTokenizer
from transformers import BertTokenizerFast
from transformerss.configuration_bert import BertConfig
from transformerss.optimization import AdamW, WarmupLinearSchedule, Warmup
from utils import (convert_examples_to_features_readcompre, processors)

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
    scheduler = Warmup[args.schedule](optimizer, warmup_steps=args.warmup_steps, t_total=num_train_optimization_steps)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

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
            # batch = tuple(t.to(args.device) for t in batch)
            inputs, masks, segments, labels, lens, poss, graphs = batch
            inputs = torch.cat((inputs, poss), dim=-1)
            masks = torch.cat((masks, masks), dim=-1)
            segments = torch.cat((segments, torch.ones_like(segments)), dim=-1)
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            segments = segments.to(args.device)
            labels = labels.to(args.device)
            graphs = graphs.to(args.device)
            loss = model(inputs, token_type_ids=segments, attention_mask=masks, labels=labels, graphs=graphs, mode="train")

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                
            # loss.backward()
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
                    # results = evaluate(args, model, tokenizer, dataname="dev")
                    evaluate(args, model, tokenizer, dataname="test")
                    model.train()
                tb_writer.add_scalar('lr', scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalar('loss', loss.item(), global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            break
        # torch.cuda.empty_cache()
    tb_writer.close()


def evaluate(args, model, tokenizer, prefix="", dataname="dev"):
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, dataname=dataname)
    args.eval_batch_size = args.eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    out_preds, out_label_ids, out_lens = [], [], []
    for batch in eval_dataloader:
        with torch.no_grad():
            # batch = tuple(t.to(args.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids, lens, poss, graphs = batch
            # batch = tuple(t.to(args.device) for t in batch)
            inputs, masks, segments, labels, lens, poss, graphs = batch
            inputs = torch.cat((inputs, poss), dim=-1)
            masks = torch.cat((masks, masks), dim=-1)
            segments = torch.cat((segments, torch.ones_like(segments)), dim=-1)
            inputs = inputs.to(args.device)
            masks = masks.to(args.device)
            segments = segments.to(args.device)
            labels = labels.to(args.device)
            graphs = graphs.to(args.device)
            predicts = model(inputs, token_type_ids=segments, attention_mask=masks, labels=labels, graphs=graphs, mode=dataname)

            out_preds.append(predicts.detach().cpu().numpy())
            out_label_ids.append(labels.detach().cpu().numpy())
            out_lens.append(lens.detach().cpu().numpy())
        torch.cuda.empty_cache()

    # result = compute_metrics_sequence_labeling(out_preds, out_label_ids, out_lens, args.label2id, args.bert_model)
    # logger.info(" {} : {} ".format(dataname, result))

    test_result = []
    for numpy_result in out_preds:
        test_result.extend(numpy_result.tolist())
    # test_examples = [exam for k, exam in enumerate(args.test_examples) if k % K == 0]
    eval_conlleval(args, args.test_examples, tokenizer, test_result, os.path.join(args.output_dir, args.eval_test_file_name))


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
            examples = processor.get_test_examples(args.data_dir, args.start_index, args.end_index)
            args.test_examples = examples
        else:
            raise ValueError("(evaluate and dataname) parameters error !")
        try:
            features = convert_examples_to_features_readcompre(examples, 
                                                          args.label2id, 
                                                          max_seq_length,
                                                          tokenizer, 
                                                          cls_token_at_end = bool('xlnet' in args.bert_model),
                                                          pad_on_left = False,
                                                          cls_token = tokenizer.cls_token,
                                                          sep_token = tokenizer.sep_token,
                                                          pad_token_id=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                          cls_token_segment_id=2 if 'xlnet' in args.bert_model else 0,
                                                          pad_token_segment_id=4 if 'xlnet' in args.bert_model else 0,
                                                          pad_token_label_id=args.pad_token_label_id,
                                                          output_dir=args.output_dir)
        except:
            print(dataname)
            raise
        # logger.info("Saving features into cached file %s", cached_features_file)
        # with open(cached_features_file, "wb") as writer:
        #     pickle.dump(features, writer)

    all_inputs = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segments = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.tokens_len for f in features], dtype=torch.long)
    all_poss = torch.tensor([f.pos for f in features], dtype=torch.long)
    all_graphs = torch.tensor([f.graph for f in features], dtype=torch.float)
    dataset = TensorDataset(all_inputs, all_masks, all_segments, all_labels, all_lens, all_poss, all_graphs)
    return dataset


def eval_conlleval(args, examples, tokenizer, result, convall_file):
    import traceback
    id2label = {index:label for label, index in args.label2id.items()}
    def test_result_to_pair(writer):
        for example, prediction in zip(examples, result):
            line = ''
            # line_token = example.text_a.split(u"")
            line_token = example.text_a.split()
            label_token = example.label_a.split()
            len_seq = len(label_token)
            if len(line_token) != len(label_token):
                logger.info(example.text_a)
                logger.info(example.label_a)
                break

            step = 0 if bool('xlnet' in args.bert_model)  else 1
            for index in range(len_seq):
                if index >= args.eval_max_seq_length - 2:
                    break
                cur_token = line_token[index]
                cur_label = label_token[index]
                sub_token = tokenizer.tokenize(cur_token)
                try:
                    if bool('xlnet' in args.bert_model):
                        sub_token = [st.replace(SPIECE_UNDERLINE, '') for st in sub_token]
                        try:
                            sub_token.remove('')
                        except ValueError:
                            pass

                    if len(sub_token) == 0:
                        raise ValueError
                    elif len(sub_token) == 1:                            
                        line += cur_token + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                        step += 1
                    elif len(sub_token) > 1:
                        if cur_label.startswith("B-"):
                            line += sub_token[0] + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                            step += 1
                            cur_label = "I-" + cur_label[2:]
                            sub_token = sub_token[1:]
                        for t in sub_token:
                            line += t + ' ' + cur_label + ' ' + id2label[prediction[step]] + '\n'
                            step += 1

                except Exception as e:
                    logger.warning(e)
                    logger.warning(example.text_a)
                    logger.warning(example.label_a)
                    line = ''
                    # traceback.print_exc()
                    break
            writer.write(line + '\n')

    with codecs.open(convall_file, 'w', encoding='utf-8') as writer:
        test_result_to_pair(writer)
    from conlleval import return_report
    eval_result = return_report(convall_file)
    logger.info(''.join(eval_result))
    with open(args.datasets + str(args.seed) + ".txt", "a+", encoding="utf8") as report:
        report.write(''.join(eval_result))
    


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--datasets", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default=None, type=str, required=True, help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--tensorboard_dir", default="logs/", type=str, help="Where do you want to store the tensorboard")
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
    parser.add_argument('--start_index', type=int, default=None)
    parser.add_argument('--end_index', type=int, default=None)
    parser.add_argument('--eval_test_file_name', type=str, default="eval_test_label.txt")
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
    import spacy
    nlp = spacy.load('en_core_web_sm')
    tags = list(nlp.get_pipe("tagger").labels)
    tags = ["[" + i.lower() + "]" for i in tags]
    tags += ["[#]"]

    args.pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    model = BertForSequenceLabeling.from_pretrained(args.pretrained_params, num_labels=args.num_labels, device=args.device, max_lens=args.train_max_seq_length)
    number_ = tokenizer.add_tokens(tags)
    assert number_ == 0
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
        tokenizer.save_pretrained(args.output_dir)

    ####################################
    if args.do_eval:
        evaluate(args, model, tokenizer, dataname="test")
    #########


if __name__ == "__main__":
    main()
