
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import scipy.stats as ss

model = Pipeline(steps=[
    ('scl', StandardScaler()),
    ('lin', SGDRegressor(
        # Logistic Regression
        loss = 'squared_loss',
        penalty = 'elasticnet',
        fit_intercept = True,
        # solver settings
        max_iter = 1000,
        tol = 1e-3,
        shuffle = True,
        random_state = 42,
        # adaptive learning
        learning_rate = 'adaptive',
        eta0 = 0.5,
        # early stopping
        early_stopping = True,
        validation_fraction = 0.15,
        n_iter_no_change = 10,
        # other
        warm_start = True,
        average = False,  # disable for Lasso!
    ))
])

hyper = {
    'lin__alpha': ss.gamma(a=1.2, loc=1e-6, scale=.08),  # alpha ~ [1e-6, 1]
    'lin__l1_ratio': ss.uniform(0, 1),
}

meta = {
    'id': "simi3",
    'name': 'LinReg ElasticNet',
    'descriptions': (
        "ElasticNet Regression (L1/L2 penalty), SGD solver, "
        "squared loss function."),
    'solver': 'Stochastic Gradient Descent',
    'active': True,
    'keywords': [
        'linear regression', 'univariate regression', 'multiple regression',
        'elasticnet', 'sklearn.linear_model.SGDRegressor'],
    'output_num': 'single',
    'output_scale': 'interval',
    'output_dtype': 'float',
    'input_num': 'multi',
    'input_scale': 'interval',
    'input_dtype': 'float'
}
