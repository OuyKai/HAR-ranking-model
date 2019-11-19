import torch.nn as nn
from torch.utils.data import DataLoader

from data.har_data import HarData
from har.model import HarEncoder


def train():
    train_data = HarData("data/sample/sample.csv", 6, 30, "embedding/model/model.bin")
    embedding_size = 100
    hidden_size = 60
    sentence_num = 6
    query_length = 30
    document_length = 30
    d_attn_size = 50
    q_attn_size = 50
    model = HarEncoder(embedding_size, hidden_size, sentence_num, query_length, document_length,
                       d_attn_size, q_attn_size, 0.1)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    criterion = nn.MultiMarginLoss()
    for batch_idx, batch_data in enumerate(train_loader):
        E_d, E_q, d_mask, q_mask, s_mask, labels, d_words, q_words = batch_data
        out = model(E_d, E_q, d_mask, q_mask, s_mask, use_pair=True)
        print(criterion(out, labels))


if __name__ == "__main__":
    train()
