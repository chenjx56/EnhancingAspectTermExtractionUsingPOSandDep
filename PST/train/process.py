import pickle
import os
import csv
import nltk
from collections import defaultdict
import random
import argparse


def writeDiscriminateWithSentenceTarget(directory):
    reviews = []
    N = 0
    with open(os.path.join(directory, 'sentence.txt'), 'r', encoding="utf-8-sig") as data_sent, open(os.path.join(directory, 'target.txt'), 'r', encoding="utf-8-sig") as data_tagt:
        for sent, tagt in zip(data_sent.readlines(), data_tagt.readlines()):
            sent = sent.strip()
            tagt = tagt.strip()
            
            tags = set()
            tmp = []
            for token, label in zip(sent.split(), tagt.split()):
                if label == '0':
                    if len(tmp) != 0:
                        tags.add(' '.join(tmp))
                        tmp = []
                elif label == '1' or int(label) % 2 ==1:
                    N += 1
                    if len(tmp) != 0:
                        tags.add(' '.join(tmp))
                        tmp = []
                    tmp.append(token)
                elif label == '2'  or int(label) % 2 ==0:
                    tmp.append(token)
                else:
                    raise ValueError
            reviews.append((sent, tags))
    print("Aspect number in {}/sentence&target : {} \n".format(directory, N))

    with open('{}/discriminate.csv'.format(directory), "w") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
        spamwriter.writerow(["text", "aspect", "label", "error type"])
        for sent, tags in reviews:
            for tag in tags:
                spamwriter.writerow([sent, tag, "1", "none"])
                tlen = len(tag.split())
                if tlen != 1:
                    # boundary sample
                    spamwriter.writerow([sent, ' '.join(tag.split()[1:]), "0", "boundary error"])
                    spamwriter.writerow([sent, ' '.join(tag.split()[:-1]), "0", "boundary error"])

                tokens = nltk.word_tokenize(sent)
                tokens = sent.split()
                poss = [p for w, p in nltk.pos_tag(tokens)]
                token2pos = {}
                pos2token = defaultdict(set)
                for t, p in zip(tokens, poss):
                    token2pos[t] = p
                    pos2token[p].add(t)
                try:
                    tagpos = set([token2pos[t] for t in tag.split()])
                    tp = random.sample(tagpos, 1)[0]
                    token = random.sample(pos2token[tp], 1)[0]
                    if all([token not in tag for tag in tags]):
                        spamwriter.writerow([sent, token, "0", "pos error"])
                except:
                    print(tokens, poss)
                    raise ValueError


def writeDiscriWithEvalSequence(directory, eval_file):
    reviews = []
    N = 0
    with open(os.path.join(directory, eval_file), 'r', encoding="utf-8-sig") as data:
        tokens, preds = [], []
        for i, line in enumerate(data.readlines()):
            line = line.strip()
            if len(line.split()) == 0:
                if len(tokens) != 0:
                    if preds != ['O'] * len(tokens):
                        reviews.append((tokens, preds))
                        N += 1
                tokens, preds = [], []
            elif len(line.split()) == 3:
                token, _, pred = line.split()
                tokens.append(token)
                preds.append(pred)
            else:
                print(i)
                raise ValueError
    print("Valide Sentence number in {}/ according source sentence&target : {} \n".format(directory, N))

    with open('{}/discri.csv'.format(directory), "w") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"')
        spamwriter.writerow(["text", "aspect", "label"])
        # B-AS, I-AS, O
        for sent, tags in reviews:
            text = ' '.join(sent).replace(' ##', '')
            tmp = []
            for token, pred in zip(sent, tags):
                if pred == 'O':
                    if len(tmp) != 0:
                        spamwriter.writerow([text, ' '.join(tmp).replace(' ##', ''), "0"])
                        tmp = []
                elif pred == 'B-AS' or pred[0] == 'B':
                    if len(tmp) != 0:
                        spamwriter.writerow([text, ' '.join(tmp).replace(' ##', ''), "0"])
                        tmp = []
                    tmp.append(token)
                elif pred == 'I-AS' or pred[0] == 'I':
                    if len(tmp) == 0:
                        continue
                    tmp.append(token)
                else:
                    raise ValueError


from utils import getPosAndGraph
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



def writeSentenceTargetWithDiscriResult(directory, eval_file, nlp, tokenizer):
    reviews = []
    N = 0
    with open(os.path.join(directory, eval_file), 'r', encoding="utf-8-sig") as data:
        tokens, preds = [], []
        for i, line in enumerate(data.readlines()):
            line = line.strip()
            if len(line.split()) == 0:
                if len(tokens) != 0:
                    if preds != ['O'] * len(tokens):
                        reviews.append((tokens, preds))
                        N += 1
                tokens, preds = [], []
            elif len(line.split()) == 3:
                token, _, pred = line.split()
                tokens.append(token)
                preds.append(pred)
            else:
                print(i)
                raise ValueError
    print("Valide Sentence number in {}/ according source sentence&target : {} \n".format(directory, N))

    sent2discri = defaultdict(set)
    with open('{}/discri.csv'.format(directory)) as csvfile, open('{}/discriResult.txt'.format(directory)) as txtfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(spamreader)
        for row, line in zip(spamreader, txtfile.readlines()):
            sent2discri[row[0]].add(line.strip())

    tag2id = {'O':0, 'B-AS':1, 'I-AS':2}
    # tag2id = {"O":0, "B-address":1, "I-address":2, "B-book":3, "I-book":4, "B-company":5, "I-company":6, "B-game":7, "I-game":8, "B-government":9, "I-government":10, "B-movie":11, "I-movie":12, "B-name":13, "I-name":14, "B-organization":15, "I-organization":16, "B-position":17, "I-position":18, "B-scene":19, "I-scene":20}
    N = 0
    with open('{}/sentence.txt'.format(directory), 'w') as sentfile, \
        open('{}/target.txt'.format(directory), 'w') as tarfile, \
        open('{}/pos.txt'.format(directory), 'w') as posfile, \
        open('{}/sentence.txt.graph'.format(directory), 'wb') as graphfile:
        idx2graph = {}
        for sent, tags in reviews:
            text = ' '.join(sent).replace(' ##', '')
            poss, graphs = getPosAndGraph(nlp, tokenizer, text, True)
            if set(sent2discri[text]) == set(['1']):
                sentfile.write(text + '\n')
                N += 1
                tmptags = []
                tmpposs = []
                for token, tag, pos in zip(sent, tags, poss):
                    if token[:2] == '##' and len(token) > 2:
                        continue
                    tmptags.append(str(tag2id[tag]))
                    tmpposs.append(pos)
                try:
                    assert len(tmptags) == len(text.split())
                    assert len(tmpposs) == len(text.split())
                except:
                    print(sent)
                    print(text)
                    print(tags)
                    print(tmptags)
                    print(poss)
                    print(tmpposs)
                    raise
                tarfile.write(' '.join(tmptags) + '\n')
                posfile.write(" ".join(tmpposs) + "\n")
                idx2graph[text] = graphs
        pickle.dump(idx2graph, graphfile)
    print("Finally Sentence number in {}/ according discriResult file : {} \n".format(directory, N))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sent2discriminate", default=None, type=str)
    parser.add_argument("--evalResult2discri", default=None, type=str)
    parser.add_argument("--discri2sent", default=None, type=str)
    parser.add_argument("--eval_file", default=None, type=str)

    parser.add_argument("--pretrained_params", default=None, type=str)
    parser.add_argument("--do_lower_case", default=True, type=str)
    args = parser.parse_args()

    if args.sent2discriminate:
        writeDiscriminateWithSentenceTarget(args.sent2discriminate)
    
    if args.evalResult2discri and args.eval_file:
        writeDiscriWithEvalSequence(args.evalResult2discri, args.eval_file)

    if args.discri2sent and args.eval_file:
        nlp = spacy.load('en_core_web_sm')
        nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
        tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_params)
        tags = list(nlp.get_pipe("tagger").labels)
        tags = ["[" + i.lower() + "]" for i in tags]
        tags += ["[#]"]
        number_ = tokenizer.add_tokens(tags, special_tokens=True)

        writeSentenceTargetWithDiscriResult(args.discri2sent, args.eval_file, nlp, tokenizer)