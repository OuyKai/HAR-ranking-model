import re
import tqdm
import argparse
from THULAC import THULAC

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", "-f",
                    dest="file",
                    type=str,
                    required=True)
args = parser.parse_args()


def tokenize_data(file_path,tokenizer):
    data = ''
    with open(file_path, "r") as f:
        for line in f:
            data += line.rstrip("/n")
    return ' '.join(tokenizer.seg(data))


if __name__ == "__main__":
    tokenizer = THULAC()
    tokenized_data = tokenize_data(args.file, tokenizer)
    print(tokenized_data[:100])

