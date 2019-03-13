
from __future__ import print_function
import torch
import numpy as np


# x = torch.empty(5, 3)
# print (x)

#x = torch.rand(5, 3)
#print (x)

# x = torch.zeros(5, 3, dtype=torch.long)
# print (x)

# x = torch.tensor([5.5, 3])
# print (x)

# x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
# print(x)

# x = torch.randn_like(x, dtype=torch.float)    # override dtype!
# print(x) 

#print(x.size())

# y = torch.rand(5, 3)
# print(y)

# print("---------------")

# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x)
# print(y)

# print(z)

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU

# x = torch.ones(2, 2, requires_grad=True)
# y = x + 2
# z = y * y * 3
# out = z.mean()

# print(z, out)


# out.backward()

# print(x.grad)

# print(x.size(), y.size(), z.size())


x = torch.randn(3, requires_grad=True)
print(x )

y = x * 2
print(y )

while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)