
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
import scipy.stats as ss

model = Pipeline(steps=[
    ('scl', StandardScaler()),
    ('lin', RidgeClassifier(
        solver = 'sparse_cg',
        tol = 0.001,  # optimizer termination criteria
        # alpha = 1.0,  # L2 regulization alpha=C^{-1}
        fit_intercept = True,
        class_weight = {0: 1, 1: 1},  # enforce binary [0,1]
        normalize = False,  # done in the pipeline
        copy_X = True,
        max_iter = 1000,  # for CG solver
    ))
])

hyper = {
    'lin__alpha': ss.gamma(a=1.5, loc=1e-5, scale=.7),  # alpha ~ [0.001, 10]
}

meta = {
    'id': "sbmi95",
    'name': 'Logistic Ridge',
    'descriptions': (
        "Logistic Regression, L2 penalty (Ridge), Conjugate Gradient solver, "
        "standard-normal transformed features."),
    'solver': 'Conjugate Gradient',
    'active': True,
    'keywords': [
        'binary classification', 'linear regression', 'RidgeClassifier',
        'scipy.sparse.linalg.cg'],
    'output_num': 'single',
    'output_scale': 'binary',
    'output_dtype': 'bool',
    'input_num': 'multi',
    'input_scale': 'interval',
    'input_dtype': 'float'
}
