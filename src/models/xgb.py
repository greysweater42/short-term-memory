from src.dataset import Dataset
from xgboost import XGBClassifier

from .model import Model


# TODO to rethink: this could potentially inherit from xgb, and train could get the dataset as parameter
class XGB(Model):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset: Dataset = dataset
        self.model = XGBClassifier(max_depth=2)

    # TODO train test split
    def train(self):
        self.model.fit(self.dataset.X, self.dataset.y)

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
