from torch.utils.data import RandomSampler
import fitlog
import logging
import random
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import BertTokenizerFast
from base_process import load_dataset
from torch.utils.data import DataLoader
from base_model import BertForTokenClassification
from torch import optim
from transformers import get_linear_schedule_with_warmup
from metric_write import evaluate_for_individual_sequence, evaluate_for_individual_sequence_report
from config import args

if args.fp16:
    from apex import amp
else:
    amp = None

# fitlog.commit(__file__)             # auto commit your codes
fitlog.set_log_dir('logs/')         # set the logging directory
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)  # record your hyper-parameters

isExisits = os.path.exists(args.log_path)
if not isExisits:
    os.makedirs(args.log_path)
logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename=args.log_path + '/train.log',  # 将日志内容写入这个文件
                    filemode='w')
logger = logging.getLogger(__name__)

def setup_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True 

def data_restore(inputs, golds, predicteds):
    res_inputs = list()
    res_glods = list()
    res_pres = list()
    for inpt1, gold1, pre1 in zip(inputs, golds, predicteds):
        for step, (inpt2, gold2, pre2) in enumerate(zip(inpt1, gold1, pre1)):
            if step == 0 and gold2 == -100:
                continue
            if gold2 == -100: # -100 represent the paddings/SEP/CLS
                res_glods.append(-100)
                res_pres.append(-100)
                res_inputs.append(101)
                break
            if pre2 == -100:
                break
            res_glods.append(gold2)
            res_pres.append(pre2)
            res_inputs.append(inpt2)
    return res_inputs, res_glods, res_pres

def train(configs):
    output_dir = configs.output_dir + str(configs.seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tokenizer = BertTokenizerFast.from_pretrained(configs.plm)
    
    print("Loading Train Data......")
    train_dataset = load_dataset(configs, tokenizer, "train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=configs.batch_size)

    print("Loading Dev Data......")
    dev_dataset = load_dataset(configs, tokenizer, "dev")
    dev_dataloader = DataLoader(dev_dataset, batch_size=configs.batch_size)

    print("Loading Test Data......")
    test_dataset = load_dataset(configs, tokenizer, "test")
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size)

    model = BertForTokenClassification.from_pretrained(configs.plm, num_labels=3)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    t_total = len(train_dataloader) * configs.epochs
    optimizer = optim.Adam(model.parameters(), lr=configs.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1), num_training_steps=t_total)
    if configs.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=configs.fp_type)

    t_loss = 0.0
    cur_best_F1 = 0.0
    global_steps = 1
    print('Load Successfully!')
    print('Train Model.........!')
    logger.info("***** Running training *****")
    for epoch in range(1, configs.epochs + 1):
        for step, batch in tqdm(enumerate(train_dataloader), desc="Traning", total=len(train_dataloader)):
            model.train()
            model.zero_grad()
            inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "token_type_ids": batch["token_type_ids"].to(device),
                    "labels": batch["labels"].to(device)
                }

            loss, logits = model(**inputs) # loss, logits

            if configs.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            scheduler.step()

            t_loss += loss.item()
            print("****Epoch: %d/%d | step:  %d/%d | total_loss: %f | learning_rate: %f" % (epoch, configs.epochs, step+1, len(train_dataloader)+1, loss.item(), scheduler.get_lr()[0]))
            logger.info("****Epoch: %d/%d | step:  %d/%d | total_loss: %f | learning_rate: %f" % (epoch, configs.epochs, step+1, len(train_dataloader)+1, loss.item(), scheduler.get_lr()[0]))
            if global_steps % configs.valid_step == 0:
                # Dev
                fitlog.add_loss(loss.item(), name="Loss", step=global_steps)
                p, r, f = valid(model, dev_dataloader, device)
                print("Dev results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
                logger.info("Dev results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
                fitlog.add_metric({"Dev": {"F1": f, "Precision": p, "Recall": r, "epoch": epoch}}, step=global_steps)
                if epoch > 0 and f >= cur_best_F1:
                    print("Saving ckpt...")
                    logger.info("Saving ckpt...")
                    output_file = output_dir + "/" + configs.datasets + '.bin' # output/baseline/$seed/${datasets}.bin
                    torch.save({'epoch': epoch + 1, 'sate_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, output_file)
                    cur_best_F1 = f
                    best_p = p
                    best_r = r
                    logger.info("Saving model checkpoint to %s", output_dir)
                    fitlog.add_best_metric({"Dev": {"F1": cur_best_F1, "Precision": best_p, "Recall": best_r, "step": global_steps}})
                    # Test
                    p, r, f = valid(model, test_dataloader, device)
                    print("Test results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
                    logger.info("Test results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
                    print("***************************************************************************************************")
                    print("Precision：", p, "Recall", r, "F1:", f)
                    print("***************************************************************************************************")
                    fitlog.add_best_metric({"Test": {"F1": f, "Precision": p, "Recall": r}})
            global_steps += 1

    # Dev
    fitlog.add_loss(loss.item(), name="Loss", step=global_steps)
    p, r, f = valid(model, dev_dataloader, device)
    print("Dev results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
    logger.info("Dev results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
    fitlog.add_metric({"Dev": {"F1": f, "Precision": p, "Recall": r, "epoch": epoch}}, step=global_steps)
    if epoch > 0 and f >= cur_best_F1:
        print("Saving ckpt...")
        logger.info("Saving ckpt...")
        output_file = output_dir + "/" + configs.datasets + '.bin' # output/baseline/$seed/${datasets}.bin
        torch.save({'epoch': epoch + 1, 'sate_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, output_file)
        cur_best_F1 = f
        best_p = p
        best_r = r
        logger.info("Saving model checkpoint to %s", output_dir)
        fitlog.add_best_metric({"Dev": {"F1": cur_best_F1, "Precision": best_p, "Recall": best_r, "step": global_steps}})
        # Test
        p, r, f = valid(model, test_dataloader, device)
        print("Test results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
        logger.info("Test results:  |    Precision:%f   |    Recall:%f    |    F1:%f" % (p, r, f))
        print("***************************************************************************************************")
        print("Precision：", p, "Recall", r, "F1:", f)
        print("***************************************************************************************************")
        fitlog.add_best_metric({"Test": {"Precision": p, "Recall": r, "F1": f}})
                    
    return test_dataloader, tokenizer, output_file

def valid(model, dev_dataloader, device, tokenizer=None):
    predicted_labels = list()
    real_labels = list()
    input_context = list()
    model.eval()
    for step, batch in tqdm(enumerate(dev_dataloader), desc="Validing...", total=len(dev_dataloader)):
        inputs = {
                    "input_ids": batch["input_ids"].to(device), # 0
                    "attention_mask": batch["attention_mask"].to(device), # 1
                    "token_type_ids": batch["token_type_ids"].to(device), # 2
                }
        labels = batch["labels"]
        logits = model(**inputs)
        predicted_label = torch.argmax(logits[0], -1)
        # predicted_label = logits[0].squeeze(0)

        predicted_labels.append(predicted_label.cpu().numpy().tolist())
        real_labels.append(labels.numpy().tolist())
        input_context.append(batch["input_ids"])

    input_context, real_labels, predicted_labels = data_restore(input_context, real_labels, predicted_labels) # (seqlen, 1)
    input_context, real_labels, predicted_labels = data_restore(input_context, real_labels, predicted_labels) # (seqlen)

    if tokenizer is not None:
        input_context = [tokenizer.decode(ids) for ids in input_context]
        path = "results/" + args.types + "/" + args.datasets + "_" + str(args.seed)
        if not os.path.exists("results/" + args.types):
            os.mkdir("results/" + args.types)
        p, r, f = evaluate_for_individual_sequence_report(path, input_context, real_labels, predicted_labels)
        # 输出训练集的badcase
        # "results/lap14_0.txt"
        if not os.path.exists("badcase/" + args.types):
            os.mkdir("badcase/" + args.types)
        with open(path, "r", encoding="utf8") as f1:
            fout = open("badcase/" + args.types + "/" + args.datasets + "_" + str(args.seed), "w+", encoding="utf8")
            sentence = []
            labs = []
            prs = []
            for line in f1.readlines():
                if line == "\n":
                    if labs != prs:
                        fout.write("\t".join(sentence))
                        fout.write("\n")
                        fout.write("\t".join(labs))
                        fout.write("\n")
                        fout.write("\t".join(prs))
                        fout.write("\n")
                    sentence = []
                    labs = []
                    prs = []
                else:
                    token, label, pred = line.strip().split(" ")
                    sentence.append(token)
                    labs.append(label)
                    prs.append(pred)
            fout.close()
    else:
        p, r, f = evaluate_for_individual_sequence(real_labels, predicted_labels)
    return p, r, f

if __name__ == "__main__":
    setup_seed(args.seed)
    test_dataloader, tokenizer, output_file = train(args)
    model = BertForTokenClassification.from_pretrained(args.plm , num_labels=3)

    model_CKPT = torch.load(output_file)
    model_state_dict = model_CKPT['sate_dict']
    model.load_state_dict(model_state_dict)
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device1)
    p, r, f = valid(model, test_dataloader, device1, tokenizer)
    print("***************************************************************************************************")
    print("Precision：", p, "Recall", r, "F1:", f)
    print("***************************************************************************************************")
    fitlog.add_best_metric({"Test": {"Precision": p, "Recall": r, "F1": f}})

    fitlog.finish() 