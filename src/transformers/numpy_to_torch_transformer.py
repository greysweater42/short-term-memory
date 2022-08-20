from .transformer import Transformer
import numpy as np
import torch
from typing import Tuple


class NumpyToTorchTransformer(Transformer):
    def transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        X_transformed = torch.tensor(X.astype(np.float32))
        if X_transformed.ndim < 3:
            # torch CNN expects each observation to have 2 dimensions, so if an observation is 1-dimensional, we add a
            # dimension
            X_transformed = X_transformed.unsqueeze(1)
        # it is more convenient to have both X and y in the same number of dimensions: 3
        y_transformed = torch.tensor(y.astype(np.float32)).view(len(y), 1, 1)
        return X_transformed, y_transformed
