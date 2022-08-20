import torch
import torch.nn as nn


conv = nn.Conv1d(2, 1, 3)

ten = torch.tensor([[[1., 1, 1, 2], [1, 1, 1, 1]]])

conv(ten)