
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import scipy.stats as ss

model = Pipeline(steps=[
    ('scl', StandardScaler()),
    ('lin', LogisticRegression())
])

hyper = {
    'lin__C': ss.expon(1e-5, 1)
}

meta = {
    'name': 'Std + Ridge',
    'descriptions': 'binary classification with logistic regression and standardized input variables.',
    'keywords': ['binary classification', 'linear regression'],
    'output_num': 'single',
    'output_scale': 'binary',
    'output_dtype': 'bool', # bool [0,1], sign [-1,+1], uint (x>=0), int, float, text, enc, ... 
    'input_num': 'multi',
    'input_scale': 'interval',
    'input_dtype': 'float'
}
