from sklearn.base import BaseEstimator, TransformerMixin
from itertools import product
import numpy as np


class MultiBinaryToMultiClass(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, Y):
        _, self.n_cols = Y.shape
        self.mapping = list(product([0, 1], repeat=self.n_cols))
        return self
    def transform(self, Y):
        n_rows, _ = Y.shape
        out = np.empty(shape=(n_rows,))
        out[:] = np.nan
        for enc in range(len(self.mapping)):
            idx = np.all(Y == self.mapping[enc], axis=1)
            out[idx] = enc
        return out.astype(int)
    def inverse_transform(self, y):
        n_rows = len(y)
        out = []
        for enc in y:
            out.append(self.mapping[int(enc)])
        return np.vstack(out)


"""
%%time
m2m = MultiBinaryToMultiClass()
m2m.fit(Y_train)
Y_class = m2m.transform(Y_train)
Y_back = m2m.inverse_transform(Y_class)

print(np.all(Y_back == Y_train))
pd.DataFrame(data=Y_class).hist()

#from collections import Counter
#print(Counter(Y_class))
from sklearn.utils import class_weight
class_weights = dict(enumerate(class_weight.compute_class_weight(
    'balanced', np.unique(Y_class), Y_class)))
print(class_weights)
"""