## This repository is in support of our paper "A novel Deep Learning approach for one-step Conformal Prediction approximation".

We show both the code used for results generation, and the specific results presented in the paper. For method descriptions and results interpretation, please see our paper here: https://arxiv.org/abs/2207.12377


### If you would like to generate your own results:

_The nonconformist package for ACP is not officially compatible with Python 3 or tensorflow models. Please make the following changes, filling in values for \<placeholders\>:_ (works as of 30/12/2022).

1. Change ```IS_LOAD_RESULTS''' to ```False'''.
1. Open the file ~/anaconda3/envs/\<env-name\>/lib/python3.9/site-packages/nonconformist/acp.py. (file path might be different if you're using a different environment manager)
1. In lines 10 and 11, update the lines ```from sklearn.cross_validation import ...``` to ```from sklearn.model_selection import ...```
1. In line 78, comment out the entire ```def gen_samples(self, y, n_samples, problem_type): ...``` function, and replace with

```
def gen_samples(self, y, n_samples, problem_type):
    if problem_type == 'classification':
        splitter = StratifiedShuffleSplit(n_splits=n_samples,
                                       test_size=self.cal_portion)
        splits = splitter.split(y, y) # 'X' is ignored, but checked for length
    else:
        print("NOT compatible for regression")

    for train, cal in splits:
        yield train, cal
```
4. Add these two lines to the import section:
```
import tensorflow as tf
from tensorflow.keras.models import clone_model
```
5. After line ```predictor = clone(self.predictor)``` (196 in the AggregatedCp function ```def fit(self, x, y): ...```), include:
```
predictor.nc_function.model.model.model = clone_model(self.predictor.nc_function.model.model.model)
predictor.nc_function.model.model.model.compile(optimizer='adam',
      loss = predictor.nc_function.model.model.loss,
      metrics=['accuracy'])
```
