import json
import sys

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append("../")
from embedding.fasttext_embedding import Embedder
import pandas as pd


class HarData(Dataset):
    def __init__(self, data_path, max_num_sent, max_seq_length, embedding_path):
        super(HarData, self).__init__()
        self.max_num_sent = max_num_sent
        self.max_seq_len = max_seq_length
        self.embedder = Embedder(max_num_sent, max_seq_length, model_path=embedding_path)
        self.data = pd.read_csv(data_path)
        self.q = self.data["queries"].tolist()  # query list
        # docs list => list of list, 5 docs with 1 positive doc and 4 negative docs
        # positive doc should be in first position
        # per doc also should be a list of strings
        self.docs = [json.loads(i) for i in self.data["docs"].tolist()]

    def __getitem__(self, index):
        selected_q = self.q[index]
        selected_docs = self.docs[index]
        E_q, q_words, q_mask = self.embedder.embed(selected_q)
        span = len(selected_docs)  # should be 5 docs
        E_d_container = torch.zeros(span, self.max_num_sent, self.max_seq_len, self.embedder.dim)
        d_mask_container = torch.zeros(span, self.max_num_sent, self.max_seq_len).short()
        s_mask_container = torch.zeros(span, self.max_num_sent).short()
        words_list = []
        for i in range(span):
            E_d, d_words, d_mask, s_mask = self.embedder.batch_embed(selected_docs[i])
            E_d_container[i, :, :, :] = E_d
            d_mask_container[i, :, :] = d_mask
            s_mask_container[i, :] = s_mask
            words_list.append(d_words)
        return E_d_container, E_q, d_mask_container, q_mask, words_list, q_words

    def __len__(self):
        return len(self.q)


if __name__ == "__main__":
    data = HarData("sample/sample.csv", 6, 30, "../embedding/model/model.bin")
    dataloader = DataLoader(dataset=data, batch_size=2, shuffle=True)
    for batch_idx, batch_data in enumerate(dataloader):
        # remember here E_d shape is (batch_size, doc_number, max_num_sent, max_seq_len, hidden_size)
        print(batch_data[0].shape)
        # E_q is (batch_size, max_seq_len, hidden_size)
        # while HarEncoder only computes query-doc, however dataset gives query-{docs_candidate}
        print(batch_data[1].shape)
