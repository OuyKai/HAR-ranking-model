import torch
import torch.nn as nn

from har.attentions import InnerAttention, DocumentHierachicalInnerAttention
from utils.score_layer import ScoreLayer


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


if __name__ == "__main__":
    # Har test
    har = Har(32, 4, 10, 12, 20, 20).cuda()
    U_d = torch.rand(6, 4, 12, 32).cuda()
    U_q = torch.rand(6, 10, 32).cuda()
    d_mask = torch.rand(6, 4, 12) > 0.3
    d_mask = d_mask.short()
    q_mask = torch.rand(6, 10) > 0.2
    q_mask = q_mask.short()
    sent_mask = torch.rand(6, 4) > 0.15
    sent_mask = sent_mask.short()

    print(har(U_d, U_q, d_mask=d_mask, q_mask=q_mask, sent_mask=sent_mask))

