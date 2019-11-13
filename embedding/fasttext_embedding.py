import sys
sys.path.append("../")
import tqdm
import re
import argparse
import fasttext
import torch
from embedding.THULAC.thulac import THULAC

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", "-f",
                    dest="file",
                    type=str)
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


class Embedder():
    def __init__(self, max_num_sent, max_seq_len):
        self.model = fasttext.load_model("model/model.bin")
        self.dim = self.model.get_dimension()
        self.tokenizer = THULAC()
        self.max_seq_len = max_seq_len
        self.max_num_sent = max_num_sent

    def embed(self, sentence):
        container = torch.zeros(self.max_seq_len, self.dim)
        mask = torch.zeros(self.max_seq_len).short()
        words = self.tokenizer.seg(sentence)[:self.max_seq_len]  # truncating while segmented len > max_seq_len
        span_words = min(len(words), self.max_seq_len)
        for i in range(span_words):
            container[i, :] = torch.tensor(self.model.get_word_vector(words[i]))
            mask[i] = 1
        return container, words, mask

    def batch_embed(self, sentences):
        container = torch.zeros(self.max_num_sent, self.max_seq_len, self.dim)
        mask = torch.zeros(self.max_num_sent, self.max_seq_len)
        span_sents = min(len(sentences), self.max_num_sent)
        words = []
        for i in range(span_sents):
            sent_container, sent_words, sent_mask = self.embed(sentences[i])
            container[i, :, :] = sent_container
            mask[i, :] = sent_mask
            words.append(sent_words)
        return container, words, mask




if __name__ == "__main__":
    # tokenizer = THULAC()
    # tokenized_data = tokenize_data(args.file, tokenizer)
    # to_file(tokenized_data)
    # tokenizer.clear()
    # embedding("model")
    e = Embedder(6, 30)
    print(e.batch_embed(["我在说什么！", "你好啊！", "是的呢！"]))
    print(e.embed("你们在说什么额！"))

