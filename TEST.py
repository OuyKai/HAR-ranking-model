import torch.nn as nn
import torch

# tanh = nn.Tanh()
# input = torch.randn(1)
# print(input)
# output = tanh(input)
# print(output)
#
# linear = nn.Linear(3,3)
# input = torch.randn(5,3)
# print(input)
# output = linear(input)
# print(output)

# softmax = nn.Softmax(dim=1)
# input = torch.randn(5,3)
# print(input)
# output = softmax(input)
# print(output)

# input_1 = torch.ones(3,3,3)
# input_2 = torch.ones(3,3,3)
# input = input_1.mul(input_2)
# print(input)
# # input = torch.cat((input_1,input_2),0).unsqueeze(0)
# # print(input)
# # print(input.transpose(1,0).mm(input))

test = torch.linspace(1,20,20).unsqueeze(0)
print(test)
test = test.view(5,4)
print(test)
test = test.transpose(1,0).contiguous()
print(test)
print(test.view(4,5))
