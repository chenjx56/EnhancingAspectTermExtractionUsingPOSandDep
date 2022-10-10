import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, default="data/Process_data/lap14_file.txt")
parser.add_argument("--log_path", type=str, default="log")
parser.add_argument("--output_dir", type=str, default="output/test/")
parser.add_argument("--plm", type=str, default="PLM/bert-base-uncased")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_seq_length", type=int, default=102)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--valid_step", type=int, default=50)

parser.add_argument("--fp16",action = "store_true")
parser.add_argument("--fp_type", type=str, default="O2")

parser.add_argument("--datasets", type=str, default="lap14")
parser.add_argument("--types", type=str, default="test")

args = parser.parse_args()