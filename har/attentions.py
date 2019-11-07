import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.masked_softmax import MaskedSoftmax


class CrossAttention(nn.Module):
    """
    query-doc cross attention
    Remember: this module only calculates V_id
    """
    def __init__(self, hidden_size, query_length, document_length):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query_length = query_length
        self.document_length = document_length

        # wc is a R^3H trainable weight vector.
        # In PyTorch it is as same as a linear layer turn (*,3H) dimension to (*,1)
        self.wc = nn.Linear(3 * hidden_size, 1)
        nn.init.kaiming_normal_(self.wc.weight)
        self.d2q = MaskedSoftmax(dim=-1)
        self.q2d = MaskedSoftmax(dim=-2)

    def forward(self, U_d, U_q, q_mask=None, d_mask=None):
        # batch_size
        batch_size = U_d.size(0)
        shape = (batch_size, self.document_length, self.query_length, self.hidden_size)

        if q_mask is None:
            q_mask = torch.ones(batch_size, self.query_length).short()
        if d_mask is None:
            d_mask = torch.ones(batch_size, self.document_length).short()

        assert q_mask.shape == (batch_size, self.query_length)
        assert d_mask.shape == (batch_size, self.document_length)

        q_mask = q_mask.unsqueeze(1)  # (B, 1, m)
        d_mask = d_mask.unsqueeze(-1)  # (B , n, 1)
        
        S_mask = torch.bmm(d_mask, q_mask)  # (B, n, m)

        # sentence_encodings => {U_d}  B*n*H
        # query_encoding => U_q  B*m*H
        U_d_exp = U_d.unsqueeze(2).expand(shape)  # (B, n, m, H)
        U_q_exp = U_q.unsqueeze(1).expand(shape)  # (B, n, m, H)
        doc_query_dot = torch.mul(U_d_exp, U_q_exp)  # (B, n, m, H) elementwise dot
        cat_data = torch.cat((U_d_exp, U_q_exp, doc_query_dot), 3)  # (B, n, m, 3H)  [u_d, u_q, u_d dot u_q]

        # (B, n, m, 3H) => (B, n, m)
        S = self.wc(cat_data).view(batch_size, self.document_length, self.query_length)
        S_d2q = self.d2q(S, S_mask)  # softmax_q
        S_q2d = self.q2d(S, S_mask)  # softmax_d
        A_d2q = torch.bmm(S_d2q, U_q)  # batch_matrix_multiply((B, n, m), (B, m, H)) => (B, n, H)
        A_q2d = torch.bmm(torch.bmm(S_d2q, S_q2d.transpose(1, 2)), U_d)  # ((B,n,m),(B,m,n)),(B,n,H)) => (B,n,H)
        doc2queryAttn = torch.mul(U_d, A_d2q)
        query2docAttn = torch.mul(U_d, A_q2d)
        V_d = torch.cat((U_d, A_d2q, doc2queryAttn, query2docAttn), 2)
        return V_d


class InnerAttention(nn.Module):
    """
    self attention mechanism 1
    """
    def __init__(self, hidden_size, attn_size):
        super(InnerAttention, self).__init__()

        self.W = nn.Linear(hidden_size, attn_size)
        nn.init.kaiming_normal_(self.W.weight)

        self.wc = nn.Linear(attn_size, 1)
        nn.init.kaiming_normal_(self.wc.weight)
        self.tanh = nn.Tanh()
        self.softmax = MaskedSoftmax(dim=-1)

    def forward(self, U, mask=None):
        attn = self.wc(self.tanh(self.W(U))).squeeze(-1)  # (B, length)
        attn = self.softmax(attn, mask)
        return attn


if __name__ == "__main__":
    # CrossAttention test
    cross_model = CrossAttention(32, 10, 12)
    U_d = torch.rand(6, 12, 32)
    U_q = torch.rand(6, 10, 32)
    d_mask = torch.rand(6, 12) > 0.3
    d_mask = d_mask.short()
    print(cross_model(U_d, U_q, d_mask=d_mask)[0])

    # InnerAttention test
    inner_model = InnerAttention(32, 20)
    print(inner_model(U_d, d_mask))






