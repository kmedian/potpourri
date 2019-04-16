import re
import itertools as it
import numpy as np
from seasalt import create_feature_names


def transformer_custom_settings(X, y=None, fit=True, names=True):
    r = re.compile(r"trans\d+")
    Z = []
    f = []
    for st in list(filter(r.match, globals())):
        sidx = st[5:]
        sm = "meta" + sidx
        if sidx.isdigit():
            print(sidx, st, sm)
            if fit:
                globals()[st].fit(X, y)
            Z_tmp = globals()[st].transform(X)
            Z.append(Z_tmp)
            if names:
                f.append(create_feature_names(
                    globals()[sm]['feature_names_prefix'], Z_tmp.shape[1]))

    return np.concatenate(Z, axis=1), list(it.chain.from_iterable(f))
