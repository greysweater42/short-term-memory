import numpy as np
from pathlib import Path
from dataset import Dataset
from xgboost import XGBClassifier


ds = Dataset(Path("./data"), ["train", "test"])
ds.prepare_datasets()

xgb = XGBClassifier(max_depth=2)
xgb.fit(ds.train.X, ds.train.y)

y_train_hat = xgb.predict(ds.train.X)
np.mean(y_train_hat == ds.train.y)
y_test_hat = xgb.predict(ds.test.X)
np.mean(y_test_hat == ds.test.y)
