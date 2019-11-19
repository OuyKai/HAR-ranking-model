import torch.nn as nn
import torch


class SelfAttention(nn.Module):
    def __init__(self, batch_size, seq_len, emb_size, attn_dim, mask=None):
        super().__init__()
        self.Wq_inner = nn.Linear(emb_size, attn_dim)
        self.Wq_outer = nn.Linear(attn_dim, 1)
        self.Tanh = nn.Tanh()
        self.Alpha = nn.Softmax(dim=0)
        if not mask:
            self.mask = torch.ones(seq_len)
        else:
            self.mask = mask

    def forward(self, Uq):
        # print(Uq.shape)
        Cq = self.Wq_outer(self.Tanh(self.Wq_inner(Uq)))
        # print(Cq.shape)
        Alpha = self.Alpha(Cq).mul(self.mask)
        # print(Alpha.shape)
        Zq = Uq.transpose(1, 0).mm(Alpha)
        # print(Zq.shape)
        return Zq


class CrossAttention(nn.Module):
    def __init__(self, batch_size, seq_len, emb_size, mask_Uq, mask_Uid):
        super().__init__()
        self.Wc = nn.Linear(3*emb_size, 1)
        self.S_D2Q = nn.Softmax(dim=0)
        self.S_Q2D = nn.Softmax(dim=1)
        self.mask = mask_Uq.transpose(1, 0).mm(mask_Uid)
        print(self.mask)
        self.batch_size = batch_size
        self.emb_size = emb_size

    def forward(self, Uq, Uid):
        stack = []
        batch_size = self.batch_size
        emb_size = self.emb_size
        shape = (batch_size, Uq.size(1), Uid.size(1), emb_size)
        Uq = Uq.unsqueeze(1).expand(shape)
        Uid = Uid.unsqueeze(2).expand(shape)
        ele_mul = torch.mul(Uq,Uid)
        cat_data = torch.cat((Uid, Uq, ele_mul),3)
        S = self.Wc(cat_data).reshape(batch_size, Uid.size(1), Uq.size(1))

        for uq in Uq:
            for uid in Uid:
                line = torch.cat((uid, uq, uid.mul(uq)), -1)
                line = line.unsqueeze(1)
                stack.append(line)
                # print(line.shape)
        Sxy = torch.cat(stack, 1).transpose(1, 0)
        print(Sxy.shape)
        # print(stack.size)

        S = self.Wc(Sxy).reshape(3, -1).mul(self.mask)
        print(S.shape)

        A_D2Q = self.S_D2Q(S).mm(Uq)
        A_Q2D = self.S_D2Q(S).mm(self.S_D2Q(S).transpose(1, 0)).mm(Uid)
        print(A_D2Q.shape)
        print(A_Q2D.shape)

        Vid = torch.cat((Uid, A_D2Q, Uid.mul(A_D2Q),Uid.mul(A_Q2D)),1)
        print(Vid.shape)
        return Vid


class Query(nn.Module):
    def __init__(self, seq_len, emb_size):
        super().__init__()
        self.RNN = nn.LSTM(input_size=seq_len, hidden_size=emb_size, num_layers=1, bidirectional=True)
        self.SelfAttn = SelfAttention(emb_size)
    pass


class Sentence(nn.Module):
    def __init__(self, seq_len, emb_size):
        super().__init__()


    pass


class Document(nn.Module):
    pass


class HAR(nn.Module):
    def __init__(self):
        super().__init__()

    pass


if __name__ == '__main__':
    attn_dim = 4
    seq_len, emb_size = 3, 5
    mask = torch.zeros(seq_len,1).unsqueeze(0)
    self_attn = SelfAttention(seq_len, emb_size, attn_dim, mask)
    input = torch.randn(seq_len, emb_size).unsqueeze(0)
    # input = torch.linspace(seq_len, emb_size)
    print(input)

    # Zq = self_attn(input)
    # print(Zq)

    crossattn = CrossAttention(seq_len, emb_size, mask, mask)
    crossattn(input, input)
