from .transformer import Transformer
import numpy as np
import torch
from typing import Tuple


class NumpyToTorchTransformer(Transformer):
    def transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        # unsqueeze 1 / view(...): returns n observations, each of dimension 1 x k (torch CNN expects each input to have
        # 2 dimensions), so the size of X is: n x 1 x k, of y: n x 1 x 1
        # besides, in torch it is more convenient to keep X and y in the same number of dimensions: 3, in this case
        return torch.tensor(X.astype(np.float32)).unsqueeze(1), torch.tensor(y.astype(np.float32)).view(len(y), 1, 1)
