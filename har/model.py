import sys

import torch.nn as nn

sys.path.append("..")
from har.attentions import InnerAttention, DocumentHierachicalInnerAttention
from utils.score_layer import ScoreLayer
from har.encoders import GRUEncoder
from embedding.fasttext_embedding import Embedder
from data.har_data import HarData
from torch.utils.data import DataLoader


class Har(nn.Module):
    def __init__(self, hidden_size, sentence_num, query_length, document_length, d_attn_size, q_attn_size):
        super(Har, self).__init__()
        self.hidden_size = hidden_size
        self.sentence_num = sentence_num
        self.query_length = query_length
        self.document_length = document_length
        self.d_attn_size = d_attn_size
        self.q_attn_size = q_attn_size
        self.q_inner_attention = InnerAttention(hidden_size, q_attn_size)
        self.doc_inner_attention = DocumentHierachicalInnerAttention(
            hidden_size,
            query_length,
            document_length,
            d_attn_size
        )
        self.score_layer = ScoreLayer(hidden_size)

    def forward(self, U_d, U_q, d_mask=None, q_mask=None, sent_mask=None):
        z_q = self.q_inner_attention(U_q, q_mask)
        y_d = self.doc_inner_attention(U_d,
                                       U_q,
                                       d_mask=d_mask,
                                       q_mask=q_mask,
                                       sent_mask=sent_mask)
        score = self.score_layer(z_q, y_d)
        return score


class HarEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size,
                 sentence_num, query_length,
                 document_length, d_attn_size,
                 q_attn_size, dropout):
        super(HarEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.document_length = document_length
        self.query_length = query_length
        self.sentence_num = sentence_num
        assert hidden_size % 2 == 0
        self.q_encoder = GRUEncoder(embedding_size, hidden_size//2, dropout)
        self.d_encoder = GRUEncoder(embedding_size, hidden_size//2, dropout)
        self.har = Har(hidden_size, sentence_num, query_length,
                       document_length, d_attn_size, q_attn_size)

    def forward(self, E_d, E_q, d_mask=None, q_mask=None, sent_mask=None, use_pair=False):
        if use_pair:
            # E_d dimension 5, shape (batch_size, doc_number, max_num_sent, max_seq_len, embedding_size)
            # E_q dimension 3, shape (batch_size, max_seq_len, embedding_size)
            doc_number = E_d.shape[1]
            batch_size = E_d.shape[0]
            E_q = E_q.unsqueeze(1).expand(batch_size, doc_number, self.query_length, self.embedding_size)
            E_d = E_d.reshape(batch_size*doc_number, -1, self.document_length, self.embedding_size)
            E_q = E_q.reshape(batch_size*doc_number, -1, self.embedding_size)
            if q_mask is not None:
                q_mask = q_mask.unsqueeze(1).expand(batch_size, doc_number, self.query_length)
                q_mask = q_mask.reshape(-1, self.query_length)
            if d_mask is not None:
                d_mask = d_mask.reshape(-1, self.sentence_num, self.document_length)
            if sent_mask is not None:
                sent_mask = sent_mask.reshape(-1, self.sentence_num)
            # E_d dimension 4, shape (batch_size*doc_number, max_num_sent, max_seq_len, embedding_size)
            # E_q dimension 3, shape (batch_size*doc_number, max_seq_len, embedding_size)
        U_q = self.q_encoder(E_q)
        U_d = self.d_encoder(E_d)
        # score shape (batch_size * doc_number, 1) if use_pair else (batch_size, 1)
        score = self.har(U_d, U_q, d_mask, q_mask, sent_mask)
        if use_pair:
            score = score.view(batch_size, doc_number, -1)
        return score.squeeze(-1)


if __name__ == "__main__":
    # Har test
    # har = Har(32, 4, 10, 12, 20, 20).cuda()
    #
    # U_d = torch.rand(6, 4, 12, 32).cuda()
    # U_q = torch.rand(6, 10, 32).cuda()
    # d_mask = torch.rand(6, 4, 12) > 0.3
    # d_mask = d_mask.short()
    # q_mask = torch.rand(6, 10) > 0.2
    # q_mask = q_mask.short()
    # sent_mask = torch.rand(6, 4) > 0.15
    # sent_mask = sent_mask.short()
    # print(har(U_d, U_q, d_mask=d_mask, q_mask=q_mask, sent_mask=sent_mask))

    # HarEncoder test
    # model = HarEncoder(50, 32, 4, 10, 12, 20, 20, 0.1).cuda()
    # E_d = torch.rand(6, 4, 12, 50).cuda()
    # E_q = torch.rand(6, 10, 50).cuda()
    # print(model(E_d, E_q, d_mask=d_mask, q_mask=q_mask, sent_mask=sent_mask))

    # Embedder test
    embedder = Embedder(max_num_sent=6, max_seq_len=30, model_path="../embedding/model/model.bin")
    embedding_size = embedder.dim
    hidden_size = 60
    sentence_num = 6
    query_length = 30
    document_length = 30
    d_attn_size = 50
    q_attn_size = 50
    model = HarEncoder(embedding_size, hidden_size, sentence_num, query_length, document_length,
                       d_attn_size, q_attn_size, 0.1)
    # E_d, d_words, d_mask, s_mask = embedder.batch_embed(["我在说什么！", "你好啊！", "是的呢！"])
    # E_q, q_words, q_mask = embedder.embed("我说的不算数！")
    # E_d, E_q, d_mask, q_mask, s_mask = map(lambda l: l.unsqueeze(0), [E_d, E_q, d_mask, q_mask, s_mask])
    # print(model(E_d, E_q, d_mask, q_mask, s_mask))

    criterion = nn.MultiMarginLoss()
    data = HarData("../data/sample/sample.csv", 6, 30, "../embedding/model/model.bin")
    dataloader = DataLoader(dataset=data, batch_size=2, shuffle=True)
    for batch_idx, batch_data in enumerate(dataloader):
        E_d, E_q, d_mask, q_mask, s_mask, labels, d_words, q_words = batch_data
        out = model(E_d, E_q, d_mask, q_mask, s_mask, use_pair=True)
        print(criterion(out, labels))


