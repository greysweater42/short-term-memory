from src.dataset import Dataset, SubDataset
import xgboost
import numpy as np

from .model import Model
from src.utils import timeit


class XGBClassifier(Model):
    # TODO since we inherit from Model, the name should be XGBClassifierModel
    # you can't simply inherit from sklearn. this is a well-know issue
    # https://github.com/scikit-learn/scikit-learn/issues/13555
    def __init__(self, dataset: Dataset, *args, **kwargs) -> None:
        self.dataset: Dataset = dataset
        self.model = xgboost.XGBClassifier(*args, **kwargs)

    @timeit
    def train(self):
        # TODO to be consistent with predict method, dataset could be provided as the arguments to train
        self.model.fit(self.dataset.train.X, self.dataset.train.y)

    def predict(self, sub_dataset: SubDataset) -> np.array:
        return self.model.predict(sub_dataset.X)

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
