
The stability selection method for lasso linear regression and logistic regression.


```python
import numpy as np
from sklearn.linear_model import Lasso, LogisticRegression
import utils

import barber_candes_selection
```


```python

class StabilitySelection(object):
    """docstring for StabilitySelection"""
    def __init__(self, base_estimator, method_name="logistic regression", lambda_grid=np.logspace(0, 1, 20), n_iterations=100, sample_portion=.5, with_replacement=False, threshold=.7, **kwargs):
        super(StabilitySelection, self).__init__()

        self.base_estimator = base_estimator
        self.method_name = method_name
        self.kwargs = kwargs
        self.lambda_grid = lambda_grid
        self.n_iterations = n_iterations
        self.sample_portion = sample_portion
        self.with_replacement = with_replacement # If with_replacement=True, we'll run bootstrap and in each iteration we get n samples with replacements.
        self.threshold = threshold

    def fit(self, x, y):

        n, p = x.shape
        probs = []

        for hyperparam in self.lambda_grid:
            S_temp = []

            for k in range(self.n_iterations):

                if self.with_replacement:
                    idx = np.random.choice(n, n, replace=self.with_replacement)
                else:
                    idx = np.random.choice(n, int(n*self.sample_portion), replace=self.with_replacement)

                x_temp, y_temp = x[idx, :], y[idx]

                if self.method_name in ["linear regression"]:
                    self.method = self.base_estimator(alpha=hyperparam, **self.kwargs)
                elif self.method_name in ["logistic regression"]:
                    self.method = self.base_estimator(C=hyperparam, **self.kwargs)

                self.method.fit(x_temp, y_temp.ravel())

                non_zero_index = list(np.arange(p)[self.method.coef_.flatten() != 0.])
                S_temp += [non_zero_index]

            all_indeces_temp = np.array([j for lst in S_temp for j in lst])

            probs += [[1.*np.sum(all_indeces_temp==j)/float(self.n_iterations) for j in range(p)]]

        probs_ = np.array(probs)
        self.max_probs = np.max(probs_, axis=0)
        
        self.S_final = np.arange(p)[self.max_probs >= self.threshold]

        return(self)

```

We simulate data from a linear regression and then apply lasso linear regression:


```python
n = 3000
mean = 0.
sd = 1.
error_std = 1.
corr = "AR(1)" 
problem_type = "regression"
q = .1

p = 1000
p1 = 20
rho = .0
r = ["uniform", .0, 1.]

x, y, TrueIndex = utils.simulate_data(n, p1, p, error_std, rho, mean, sd, r, problem_type, corr=corr)
```


```python
SS = StabilitySelection(base_estimator=Lasso, method_name="linear regression", lambda_grid=np.linspace(0.1, 2, 50), n_iterations=2, sample_portion=.5, with_replacement=False, threshold=.7)

SS.fit(x, y)

S_SS = SS.S_final
```


```python
print("-----------Stability Selection------------------")
fdr_SS = 100*utils.FDR(S_SS, TrueIndex)
power_SS = 100*utils.power(S_SS, TrueIndex)
fnp_SS = 100*utils.FNP(S_SS, TrueIndex, p)

print('------------SS method-------------')
print("FDR:  " +str(fdr_SS) + "%")
print("power:  "+str(power_SS) + "%")
print("FNP:  "+str(fnp_SS) + "%")
```

We simulate data from a logistic regression and then apply lasso logistic regression:


```python
n = 3000
mean = 0.
sd = 1.
error_std = 1.
corr = "AR(1)" 
problem_type = "classification"
q = .1

p = 1000
p1 = 20
rho = .0
r = ["uniform", .0, 1.]

x, y, TrueIndex = utils.simulate_data(n, p1, p, error_std, rho, mean, sd, r, problem_type, corr=corr)
```


```python
SS = StabilitySelection(base_estimator=LogisticRegression, method_name="logistic regression", lambda_grid=np.linspace(.1, 2, 20), n_iterations=100, sample_portion=.5, with_replacement=False, threshold=.7)

SS.fit(x, y)

S_SS = SS.S_final
```


```python
print("-----------Stability Selection------------------")
fdr_SS = 100*utils.FDR(S_SS, TrueIndex)
power_SS = 100*utils.power(S_SS, TrueIndex)
fnp_SS = 100*utils.FNP(S_SS, TrueIndex, p)

print('------------SS method-------------')
print("FDR:  " +str(fdr_SS) + "%")
print("power:  "+str(power_SS) + "%")
print("FNP:  "+str(fnp_SS) + "%")
```
