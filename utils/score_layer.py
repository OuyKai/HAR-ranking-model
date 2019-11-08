import torch
import torch.nn as nn


class ScoreLayer(nn.Module):

    def __init__(self, hidden_size):
        super(ScoreLayer, self).__init__()
        self.wd = nn.Linear(4 * hidden_size, hidden_size)
        self.wf = nn.Linear(hidden_size, 1)
        nn.init.kaiming_normal_(self.wd.weight)
        nn.init.kaiming_normal_(self.wf.weight)

    def forward(self, z_q, y_d):
        y_d_bar = self.wd(y_d)
        assert y_d_bar.shape == z_q.shape
        p = torch.mul(z_q, y_d_bar)
        r = self.wf(p)
        return r
