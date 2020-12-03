import numpy as np
from scipy.linalg import sqrtm, inv
from sklearn.linear_model import LogisticRegression


def estimate_coordinate(x, y, mu = np.repeat(1, 4).reshape((-1,1)), sigma = np.identity(4) * 0.9 + 0.1):
    return (((x - mu[:3].T) @ inv(sigma[:3, :3]) @ sigma[:3, 3])\
         + y.reshape((-1)) * mu[3]).reshape((-1,1))

def errors(r = 0.1, d = 4, n_source = 200, n_target = 200, pi_source = 0.5, pi_target = 0.75):
    
    # Distributional parameters
    sigma = np.identity(d) * (1-r) + r
    M = sqrtm(sigma)
    mu = np.repeat(1, d).reshape((-1,1))

    # Source and target data
    y_source = np.random.binomial(1, pi_source, (n_source,1))
    x_source = np.random.normal(0, 1, (n_source, d)) @ M + y_source @ mu.T

    y_target = np.random.binomial(1, pi_target, (n_target,1))
    x_target = np.random.normal(0, 1, (n_target, d)) @ M + y_target @ mu.T
    x_target = x_target[:, :3]

    ## Ignore absent co-ordinates in source data
    cl = LogisticRegression(max_iter= 1000)
    cl.fit(x_source[:, :3], y_source)
    coef, intercept = cl.coef_, cl.intercept_
    # Shift in intercept to accomodate label shift
    intercept = intercept + np.log(pi_target/(1-pi_target)) - np.log(pi_source/(1-pi_source)) 
    logit = (x_target @ coef.T).reshape((-1,)) + intercept
    y_target_estimate = (logit > 0).astype('float32').reshape(-1, 1)
    error_deleted = np.mean(np.absolute(y_target - y_target_estimate))

    # Estimate absent co-ordinates in target dat
    last_coord_estimated = estimate_coordinate(x_target, y_target)
    x_target_estimated = np.concatenate([x_target, last_coord_estimated], axis = 1)

    ## Fit logistic model with full source data
    cl = LogisticRegression(max_iter= 1000)
    cl.fit(x_source, y_source)
    coef, intercept = cl.coef_, cl.intercept_
    # Shift in intercept for label shift
    intercept = intercept + np.log(pi_target/(1-pi_target)) - np.log(pi_source/(1-pi_source))
    logit = (x_target_estimated @ coef.T).reshape((-1,)) + intercept
    y_target_estimate = (logit > 0).astype('float32').reshape(-1, 1)
    error_estimated = np.mean(np.absolute(y_target - y_target_estimate))
    
    return [error_deleted, error_estimated]