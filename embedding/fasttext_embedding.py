import sys
sys.path.append("../")
import tqdm
import re
import argparse
import fasttext
from embedding.THULAC.thulac import THULAC

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", "-f",
                    dest="file",
                    type=str,
                    required=True)
args = parser.parse_args()


def tokenize_data(file_path, tokenizer):
    data = ""
    with open(file_path) as f:
        for line in f:
            data += line

    seg = re.compile("\n")
    punc = re.compile('。')
    data = seg.split(data)
    new_data = []
    for d in tqdm.tqdm(data):

        if len(d) > 1000:
            new_line = []
            splited = punc.split(d)
            for splice in splited:
                new_line.extend(tokenizer.seg(splice))
                if splice:
                    new_line.extend(['。'])
            new_data.append(' '.join(new_line))
        elif d:
            new_data.append(' '.join(tokenizer.seg(d)))

    return "\n".join(new_data)


def to_file(data):
    import os
    if not os.path.exists("model"):
        os.mkdir("model")
    with open("model/data.txt", "w") as f:
        f.write(data)


def embedding(model_path):
    model = fasttext.train_unsupervised(model_path+"/data.txt")
    model.save_model(model_path+"/model.bin")
    return model


if __name__ == "__main__":
    tokenizer = THULAC()
    tokenized_data = tokenize_data(args.file, tokenizer)
    to_file(tokenized_data)
    tokenizer.clear()
    embedding("model")

