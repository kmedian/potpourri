
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import scipy.stats as ss

model = Pipeline(steps=[
    ('scl', StandardScaler()),
    ('lin', Ridge(
        solver = 'lsqr',
        tol = 0.001,  # optimizer termination criteria
        # alpha = 1.0,  # L2 regulization alpha=C^{-1}
        fit_intercept = True,
        normalize = False,  # done in the pipeline
        copy_X = True,
    ))
])

hyper = {
    'lin__alpha': ss.gamma(a=1.5, loc=1e-5, scale=.7),  # alpha ~ [0.001, 10]
}

meta = {
    'id': "simi9",
    'name': 'LinReg Ridge',
    'descriptions': (
        "Ridge Regression (L2 penalty), Least-Square equations solver, "
        "standard-normal transformed features."),
    'solver': 'Least Square',
    'active': True,
    'keywords': [
        'linear regression', 'univariate regression', 'multiple regression',
        'ridge', 'l2 penalty', 'sklearn.linear_model.Ridge'],
    'output_num': 'single',
    'output_scale': 'interval',
    'output_dtype': 'float',
    'input_num': 'multi',
    'input_scale': 'interval',
    'input_dtype': 'float'
}
