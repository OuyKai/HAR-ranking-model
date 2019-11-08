import torch
import torch.nn as nn


class MaskedSoftmax(nn.Module):
    """
    A masked version of softmax because of sentence length or doc length may be padded.
    """
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=self.dim)

    def forward(self, input, mask=None):
        if mask is None:
            mask = torch.ones_like(input).short()
        if mask.dim() == input.dim()-1:
            # in case of mask may not be padded in hidden feature space
            mask = mask.unsqueeze(-1).expand(input.shape)
        if input.is_cuda:
            mask = mask.cuda()
        # print(mask.shape)
        # print(input.shape)
        assert mask.shape == input.shape and mask.dtype == torch.short, "Mask shape %s should match input shape %s" % (mask.shape, input.shape)
        input.masked_fill_(mask == 0, -1e9)
        input = self.softmax(input)
        return input


if __name__ == "__main__":
    model = MaskedSoftmax(dim=1)
    mask = torch.rand(3, 5) >= 0.5
    mask = mask.short()
    print(model(torch.rand(3, 5, 2).cuda(), mask))
