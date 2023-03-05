import pandas as pd
import numpy as np

from barmpy.barn import BARN

# adapted from BartPy by Jake Coltman
def run(alpha, beta, num_nets, n_regressors, n_burn=50, n_obsv=1000):
    b_true = np.random.uniform(-2, 2, size = n_regressors)
    x = np.random.normal(0, 1, size=n_obsv * n_regressors).reshape(n_obsv, n_regressors)
    x[:50, 1] = 4
    X = pd.DataFrame(x)
    y = np.random.normal(0, 0.1, size=n_obsv) + np.array(X.multiply(b_true, axis = 1).sum(axis=1))
    model = BARN(num_nets=num_nets)
    model.setup_nets()
    model.train(X, y)
    # predictions = model.predict()
    return model, x, y


if __name__ == "__main__":
    import cProfile
    from datetime import datetime as dt
    print(dt.now())
    # model, x, y = run(0.95, 2., 200, 50, n_obsv=100000)
    cProfile.run("run(0.95, 2., 200, 40)", "restats")

    print(dt.now())
