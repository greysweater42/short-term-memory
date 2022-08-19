from src.dataset import Dataset
import xgboost
import numpy as np
import pandas as pd

from .model import Model
from src.models.model import Model
from src.utils import timeit


class XGBClassifier(Model):
    # you can't simply inherit from sklearn. this is a well-know issue
    # https://github.com/scikit-learn/scikit-learn/issues/13555
    def __init__(self, dataset: Dataset, *args, **kwargs) -> None:
        self.dataset: Dataset = dataset
        self.model = xgboost.XGBClassifier(*args, **kwargs)

    # TODO train test split
    @timeit
    def train(self):
        self.model.fit(self.dataset.X, self.dataset.y)

    def predict(self, X: pd.DataFrame) -> np.array:
        return self.model.predict(X)

# def main(X, y):
#     from sklearn import svm
#     from sklearn.model_selection import train_test_split

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     model = svm.SVC(probability=True)
#     from sklearn.decomposition import PCA

#     pca = PCA(20)
#     pca.fit(X_train)
#     model.fit(pca.transform(X_train), y_train)
#     return calculate_metrics(model, pca.transform(X_train), y_train, pca.transform(X_test), y_test)
