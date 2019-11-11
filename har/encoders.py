import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout):
        super(GRUEncoder, self).__init__()
        self.GRU = nn.GRU(embedding_size, hidden_size,
                          batch_first=True, bidirectional=True)
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        for p in self.GRU.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)
            else:
                torch.nn.init.normal_(p)

    def forward(self, embed):
        batch_size = embed.size(0)
        if embed.dim() == 3:  # E_q (B, m, E) => (B, 1, m, E)
            embed = embed.unsqueeze(1)
        assert embed.dim() == 4
        sentence_num = embed.size(1)
        embed = embed.view(batch_size*sentence_num, -1, self.embedding_size)  # => (B*l, seq, E)
        U, _ = self.GRU(embed)
        U = U.view(batch_size, sentence_num, -1, self.hidden_size*2)
        if sentence_num == 1:
            U = U.squeeze(1)
        return self.dropout(U)


if __name__ == "__main__":
    q_encoder = GRUEncoder(30, 20, 0.1)
    d_encoder = GRUEncoder(30, 20, 0.1)
    U_q = torch.rand(6, 12, 30)
    print(q_encoder(U_q))
    U_d = torch.rand(6, 4, 10, 30)
    print(d_encoder(U_d))

